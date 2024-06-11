import argparse
import numpy as np
import rasterio
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
import os
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV

from sklearn.metrics import mean_absolute_error
import numpy as np


def evaluate(fold, use_checkpoint = False, imagery_path = None, imagery_source = None, mode = 'temporal'):
    if use_checkpoint:
        if mode == 'temporal':
            checkpoint = f'dinov2_vitb14_temporal_best_{imagery_source}.pth'
        elif mode == 'spatial':
            checkpoint = f'dinov2_vitb14_{fold}_all_cluster_best_{imagery_source}.pth'
        else:
            raise Exception()
    model_output_dim = 768

    if imagery_source == 'L':
        normalization = 30000.
        transform_dim = 336
    elif imagery_source == 'S':
        normalization = 3000.
        transform_dim = 994

    import pandas as pd
    from tqdm import tqdm

    if mode == 'spatial':
        train_df = pd.read_csv(f'train_fold_{fold}.csv', index_col=0)
        test_df = pd.read_csv(f'test_fold_{fold}.csv', index_col=0)
    elif mode == 'temporal':
        train_df = pd.read_csv(f'before_2020.csv', index_col=0)
        test_df = pd.read_csv(f'after_2020.csv', index_col=0)
    available_imagery = []
    import os
    for d in os.listdir(imagery_path):
        if d[-2] == imagery_source:
            for f in os.listdir(os.path.join(imagery_path, d)):
                available_imagery.append(os.path.join(imagery_path, d, f))
    available_centroids = [f.split('/')[-1][:-4] for f in available_imagery]
    train_df = train_df[train_df['CENTROID_ID'].isin(available_centroids)]
    test_df = test_df[test_df['CENTROID_ID'].isin(available_centroids)]

    def filter_contains(query):
        """
        Returns a list of items that contain the given query substring.
        
        Parameters:
            items (list of str): The list of strings to search within.
            query (str): The substring to search for in each item of the list.
            
        Returns:
            list of str: A list containing all items that have the query substring.
        """
        # Use a list comprehension to filter items
        for item in available_imagery:
            if query in item:
                return item
    train_df['imagery_path'] = train_df['CENTROID_ID'].apply(filter_contains)
    train_df = train_df[train_df['deprived_sev'].notna()]
    test_df['imagery_path'] = test_df['CENTROID_ID'].apply(filter_contains)
    test_df = test_df[test_df['deprived_sev'].notna()]

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = torch.hub.load('facebookresearch/dinov2', f'dinov2_vitb14')
    class ViTForRegression(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.base_model = base_model
            # Assuming the original model outputs 768 features from the transformer
            self.regression_head = nn.Linear(model_output_dim, 99)  # Output one continuous variable

        def forward(self, pixel_values):
            outputs = self.base_model(pixel_values)
            # We use the last hidden state
            return torch.sigmoid(self.regression_head(outputs))
        
        def forward_encoder(self, pixel_values):
            return self.base_model(pixel_values)
    model = ViTForRegression(base_model)
    if use_checkpoint:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['model_state_dict'])

    model.to(device)
    model.eval()
    
    def get_features(df):
        dino_features = []
        transform = transforms.Compose([
            transforms.Resize((transform_dim, transform_dim)),  # Resize the image to the input size expected by the model
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet's mean and std
        ])
        
        for idx in tqdm(range(len(df))):
            image = load_and_preprocess_image(df.iloc[idx]['imagery_path'])
            image_tensor = transform(Image.fromarray(image))
            with torch.no_grad():
                features = model.forward_encoder(torch.stack([image_tensor]).to(device))
            for f in features:
                dino_features.append(f.cpu())

                # with open(filename, 'wb') as file:
                #     pickle.dump(dino_features, file)
        return dino_features
    train_sat_features = get_features(train_df)
    test_sat_features = get_features(test_df)

    train_features_df = pd.DataFrame([f.tolist() for f in train_sat_features])
    test_features_df = pd.DataFrame([f.tolist() for f in test_sat_features])
    train_target = train_df['deprived_sev']
    test_target = test_df['deprived_sev']
    # Create a list of alphas to consider for RidgeCV
    alphas = np.logspace(-6, 6, 13)

    # Creating the pipeline with StandardScaler and RidgeCV
    # RidgeCV is initialized with a list of alphas
    pipeline = make_pipeline(StandardScaler(), RidgeCV(alphas=alphas))

    # Fit the model
    pipeline.fit(train_features_df, train_target)


    # Predict on test data
    y_pred = pipeline.predict(test_features_df)

    # Evaluate the model using Mean Absolute Error (MAE)
    mae = mean_absolute_error(test_target, y_pred)

    print("Mean Absolute Error on Test Set:", mae)
    return mae




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run satellite image processing model training.')
    parser.add_argument('--imagery_source', type=str, default='L', help='L for Landsat and S for Sentinel')
    parser.add_argument('--imagery_path', type=str, help='The parent directory of all imagery')
    parser.add_argument('--mode', type=str, default='temporal', help='Evaluating temporal model or spatial model')
    parser.add_argument('--use_checkpoint', action='store_true', help='Whether to use checkpoint file. If not, use raw model.')

    
    args = parser.parse_args()
    maes = []
    if args.mode == 'temporal':
        print(evaluate(1, args.use_checkpoint, args.imagery_path, args.imagery_source, args.mode))
    else:
        for i in range(5):
            fold = i + 1
            mae = evaluate(fold, args.use_checkpoint,args.imagery_path, args.imagery_source, args.mode)
            maes.append(mae)
        print(np.mean(maes), np.std(maes)/np.sqrt(5))
  
