import argparse
import pandas as pd
from tqdm import tqdm
import os
import random
import rasterio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
from PIL import Image    
import re

import torch.nn as nn
import imageio
from sklearn.model_selection import train_test_split    
from torch.optim import Adam
from torch.nn import L1Loss


"""
Finetuning the DinoV2 model using spatial data
This spatial data is stored in survey_processing/processed_data and is split into folds
View survey_processing/main.py for more information on this

Dinov2's input is a RGB satellite image converted to a tensor
The target for each satellite image is several dhs variables (predict_target variable)
These variables once one-hot encoded form a larger dimension vector
The DinoV2 model outputs a 768 dimension vector, so we add an additional linear layer with sigmoid activation function
in order to get an output the size of our target vector

After each epoch we save the models weights to our 'last model' file and if the error is sufficiently low -
we save it to 'best model' file also

Satellite imagery is saved in the following file structure
Sub directories should be of the form country code + year + satellite
Filenames are the CENTROID_ID

- imagery parent directory
--- ET2018S2
------ ET2000000090.tif
------ ET2000000213.tif
------ ...
--- RW2018S2
------ ...
--- ...
"""


def main(fold, model_name, imagery_path, imagery_source, emb_size, batch_size, num_epochs):
    """
    Finetunes and validates Dinov2 model using one fold of data
    Saves the last and the best model weights to file 
        
    Parameters:
        fold (integer): fold number
        model_name (string): model name (i.e dinov2_vitb14)
        imagery_path (string): parent directory of imagery
        imagery_source (string): Landsat (L) or Sentinel (S)
        emb_size (int): size of model output, default is 768
        batch_size (int): batch size
        num_epochs (int): number of epochs
        
    Returns:
        None
    """

    if imagery_source == 'L':
        normalization = 30000.
        imagery_size = 336
    elif imagery_source == 'S':
        normalization = 3000.
        imagery_size = 994
    else:
        raise Exception("Unsupported imagery source")
    data_folder = r'survey_processing/processed_data'

    # load preprocessed dhs data of the fold we are considering, we will take target columns from this
    train_df = pd.read_csv(f'{data_folder}/train_fold_{fold}.csv', index_col=0)
    test_df = pd.read_csv(f'{data_folder}/test_fold_{fold}.csv', index_col=0)

    # store file paths of all available imagery in following list
    available_imagery = []
    for d in os.listdir(imagery_path):
        # d[-2] will either be S or L, refer to top comment to understand file structure of saved images
        if d[-2] == imagery_source:
            for f in os.listdir(os.path.join(imagery_path, d)):
                available_imagery.append(os.path.join(imagery_path, d, f))
    
    # gets filename of each image without the .fileformat
    available_centroids = [f.split('/')[-1][:-4] for f in available_imagery]
    # filter df to remove rows with no corresponding satellite image
    train_df = train_df[train_df['CENTROID_ID'].isin(available_centroids)]
    test_df = test_df[test_df['CENTROID_ID'].isin(available_centroids)]


    def filter_contains(query):
        """
        Returns a list of items that contain the given query substring.
        
        Parameters:
            query (str): The substring to search for in each item of the list.
            
        Returns:
            list of str: A list containing all items that have the query substring.
        """
        # Use a list comprehension to filter items
        for item in available_imagery:
            if query in item:
                return item
            

    # add file path of satellite imagery corresponding to each row
    train_df['imagery_path'] = train_df['CENTROID_ID'].apply(filter_contains)
    test_df['imagery_path'] = test_df['CENTROID_ID'].apply(filter_contains)

    # dhs variables to use as target data
    # vaccination status, wealth index, height for age s.d, level of education, water access, sleeping arrangements etc
    predict_target = ['h10', 'h3', 'h31', 'h5', 'h7', 'h9', 'hc70', 'hv109', 'hv121', 'hv106', 'hv201', 'hv204', 'hv205', 'hv216', 'hv225', 'hv271', 'v312']

    # find one hot encoded columns associated with each of the categorical targets using regex
    filtered_predict_target = []
    for col in predict_target:
        filtered_predict_target.extend(
            [c for c in train_df.columns if c == col or re.match(f"^{col}_[^a-zA-Z]", c)]
        )
    # Drop rows with NaN values in the filtered subset of columns
    train_df = train_df.dropna(subset=filtered_predict_target)
    predict_target = sorted(filtered_predict_target)


    def load_and_preprocess_image(path):
        with rasterio.open(path) as src:
            # Read the specific bands (4, 3, 2 for RGB)
            r = src.read(4)  # Band 4 for Red
            g = src.read(3)  # Band 3 for Green
            b = src.read(2)  # Band 2 for Blue
            
            # Stack and normalize the bands
            img = np.dstack((r, g, b))
            img = img / normalization*255.  # Normalize to [0, 1] (if required)
            
        img = np.nan_to_num(img, nan=0, posinf=255, neginf=0)
        img = np.clip(img, 0, 255)  # Clip values to be within the 0-255 range
        
        return img.astype(np.uint8)  # Convert to uint8


    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    # set your desired seed
    seed = 42
    set_seed(seed)

    train, validation = train_test_split(train_df, test_size=0.2, random_state=42)


    class CustomDataset(Dataset):
        """
        Stores dataframe and transform (collection of image transforms)
        When object is indexed, returns image_tensor, target
        """

        def __init__(self, dataframe, transform):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            item = self.dataframe.iloc[idx]
            image = load_and_preprocess_image(item['imagery_path'])
            # Apply feature extractor if necessary, might need adjustments
            image_tensor = self.transform(Image.fromarray(image))
            
            # Assuming your target is a single scalar
            target = torch.tensor(item[predict_target], dtype=torch.float32)
            return image_tensor, target  # Adjust based on actual output of feature_extractor


    # convert image to tensor of the correct size
    transform = transforms.Compose([
        transforms.Resize((imagery_size, imagery_size)),  # Resize the image to the input size expected by the model
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet's mean and std
    ])

    train_dataset = CustomDataset(train, transform)
    val_dataset = CustomDataset(validation, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=batch_size+4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=batch_size+4)

    base_model = torch.hub.load('facebookresearch/dinov2', model_name)


    def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, filename)


    torch.cuda.empty_cache()


    class ViTForRegression(nn.Module):
        """
        Parent class is nn.Module (i.e DinoV2 model)
        Adds additional linear layer with sigmoid activation function in order to get output of length len(predict_target)
        """

        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            # Assuming the original model outputs 768 features from the transformer
            self.regression_head = nn.Linear(emb_size, len(predict_target))  # Output one continuous variable

        def forward(self, pixel_values):
            outputs = self.base_model(pixel_values)
            # We use the last hidden state
            return torch.sigmoid(self.regression_head(outputs))


    # load last and best model for comparison of loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForRegression(base_model).to(device)
    best_model = f'modelling/dino/model/{model_name}_{fold}_all_cluster_best_{imagery_source}.pth'
    last_model = f'modelling/dino/model/{model_name}_{fold}_all_cluster_last_{imagery_source}.pth'
    if os.path.exists(last_model):
        last_state_dict = torch.load(last_model)
        best_error = torch.load(best_model)['loss']
        epoch_ran = last_state_dict['epoch']
        model.load_state_dict(last_state_dict['model_state_dict'])
        print('Found existing model')
    else:
        epochs_ran = 0
        best_error = np.inf

    # Move model to appropriate device
    model.to(device)

    base_model_params = {'params': model.base_model.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6}
    head_params = {'params': model.regression_head.parameters(), 'lr': 1e-6, 'weight_decay': 1e-6}

    # Setup the optimizer
    optimizer = torch.optim.Adam([base_model_params, head_params])
    loss_fn = L1Loss()

    # training and validation
    for epoch in range(epochs_ran+1, num_epochs):
        torch.cuda.empty_cache()
        model.train()
        print('Training...')

        for batch in tqdm(train_loader):
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, targets)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.cuda.empty_cache()

        # Validation phase
        model.eval()
        val_loss = []
        indiv_loss = []
        print('Validating...')
        for batch in val_loader:
            images, targets = batch
            images, targets = images.to(device), targets.to(device)
            
            # Forward pass
            with torch.no_grad():
                outputs = model(images)
            batch_loss = loss_fn(outputs, targets)
            val_loss.append(batch_loss.item())
            indiv_loss.append(torch.mean(torch.abs(outputs-targets), axis=0))
        
        # Compute mean validation loss
        mean_val_loss = np.mean(val_loss)   
        mean_indiv_loss = torch.stack(indiv_loss).mean(dim=0)

        # save best and last model if appropriate
        if mean_val_loss< best_error:
            save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=best_model)
            best_error = mean_val_loss
        print(f'Epoch [{epoch+1}/{num_epochs}], Validation Loss: {mean_val_loss}, Individual Loss: {mean_indiv_loss}')
        save_checkpoint(model, optimizer, epoch, mean_val_loss, filename=last_model)


# handle command line inputs, note we have to run a seperate command to train on each fold
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run satellite image processing model training.')
    parser.add_argument('--fold', type=int, help='CV fold')
    parser.add_argument('--model_name', type=str, help='Name of the model')
    parser.add_argument('--imagery_path', type=str, help='The parent directory of all imagery')
    parser.add_argument('--imagery_source', type=str, default='L', help='L for Landsat and S for Sentinel')
    parser.add_argument('--emb_size', type=int, default=768, help='Size of the model output')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs for training')
    args = parser.parse_args()
    main(args.fold, args.model_name, args.imagery_path, args.imagery_source,args.emb_size, args.batch_size, args.num_epochs)