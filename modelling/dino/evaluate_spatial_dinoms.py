import argparse
import numpy as np
import rasterio
import numpy as np
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import RidgeCV
from torch.utils.data import Dataset, DataLoader
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
from tqdm import tqdm

def evaluate(fold, model_name, target = "", use_checkpoint = False, model_not_named_target = True, imagery_path = None, imagery_source = None, mode = 'temporal', model_output_dim = 768):
    model_par_dir = r'modelling/dino/model/'

    import os
    best_model = f'{model_name}ms_cat_{fold}_all_cluster_best_{imagery_source}{target}_.pth'
    checkpoint = os.path.join(model_par_dir, best_model)

    print(f"Evaluating fold {fold} with target {target} using checkpoint {checkpoint}")

    if target == '':
        eval_target = 'deprived_sev'
        target_size = 99
    else:
        eval_target = target
        if model_not_named_target:
            target_size = 1
        else:
            target_size = 99

        
    if imagery_source == 'L':
        normalization = 30000.
        transform_dim = 336
    elif imagery_source == 'S':
        normalization = 3000.
        transform_dim = 994

    data_folder = r'survey_processing/processed_data/'
    if 'spatial' in mode:
        train_df = pd.read_csv(f'{data_folder}train_fold_{fold}.csv')
        test_df = pd.read_csv(f'{data_folder}test_fold_{fold}.csv')
    elif 'temporal' in mode:
        train_df = pd.read_csv(f'{data_folder}before_2020.csv')
        test_df = pd.read_csv(f'{data_folder}after_2020.csv')
    elif mode == 'one_country':
        train_df = pd.read_csv(f'{data_folder}train_fold_{fold}.csv')
        test_df = pd.read_csv(f'{data_folder}test_fold_{fold}.csv')
    
    available_imagery = []
    import os
    for d in os.listdir(imagery_path):
        if d[-2] == imagery_source:
            for f in os.listdir(os.path.join(imagery_path, d)):
                available_imagery.append(os.path.join(imagery_path, d, f))
    def is_available(centroid_id):
        for centroid in available_imagery:
            if centroid_id in centroid:
                return True
        return False
    train_df = train_df[train_df['CENTROID_ID'].apply(is_available)]
    test_df = test_df[test_df['CENTROID_ID'].apply(is_available)]
    if test_df.empty:
        raise Exception("Empty test set")
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
            bands = src.read()
            img = bands[:13]
            img = img / normalization  # Normalize to [0, 1] (if required)
        
        img = np.nan_to_num(img, nan=0, posinf=1, neginf=0)
        img = np.clip(img, 0, 1)  # Clip values to be within the 0-1 range
        img = np.transpose(img, (1, 2, 0))
        # Scale back to [0, 255] for visualization purposes
        img = (img * 255).astype(np.uint8)

        return img

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_models = [torch.hub.load('facebookresearch/dinov2', model_name).to(device) for _ in range(4)]
    class ViTForRegression(nn.Module):
        def __init__(self, base_models, grouped_bands=[[4, 3, 2], [8, 4, 2], [13, 1, 3], [12, 8, 2]], emb_size=768, output_size=1):
            super().__init__()
            self.base_models = base_models
            self.grouped_bands = torch.tensor(grouped_bands) - 1
            self.cross_attention = nn.MultiheadAttention(embed_dim=emb_size, num_heads=8)
            
            # Update the input size of the regression head to handle concatenation of 4 embeddings
            self.regression_head = nn.Linear(emb_size * len(grouped_bands), output_size)

        def forward_encoder(self, pixel_values):
            # Extract outputs from each base model with specific band groups
            outputs = [self.base_models[i](pixel_values[:, self.grouped_bands[i], :, :]) for i in range(len(self.base_models))]
            
            # Stack and permute outputs for multihead attention
            outputs = torch.stack(outputs, dim=0)  # Shape: [num_views, batch_size, emb_size]
            
            # Apply cross-attention
            attn_output, _ = self.cross_attention(outputs, outputs, outputs)  # Shape: [num_views, batch_size, emb_size]
            
            # Concatenate the attention output across all views
            concat_output = torch.cat([attn_output[i] for i in range(attn_output.size(0))], dim=-1)  # Shape: [batch_size, emb_size * num_views]
            
            return concat_output
        
        def forward(self, pixel_values):
            # Extract outputs from each base model with specific band groups
            concat_output = self.forward_encoder(pixel_values)
            # Pass through regression head
            return torch.sigmoid(self.regression_head(concat_output))
        
    model = ViTForRegression(base_models, output_size=target_size).to(device)
    if use_checkpoint:
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict['model_state_dict'])

    class CustomDataset(Dataset):
        def __init__(self, dataframe, transform):
            self.dataframe = dataframe
            self.transform = transform

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            item = self.dataframe.iloc[idx]
            image = load_and_preprocess_image(item['imagery_path'])
            # Apply feature extractor if necessary, might need adjustments
            image_tensor = self.transform(image)
            
            # Assuming your target is a single scalar
            target = torch.tensor(item[eval_target], dtype=torch.float32)
            return image_tensor, target  # Adjust based on actual output of feature_extractor
        
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Resize((transform_dim, transform_dim)),  # Resize the image to the input size expected by the model
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet's mean and std
    ])

    train_dataset = CustomDataset(train_df, transform)
    val_dataset = CustomDataset(test_df, transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    model.to(device)
    model.eval()
    
    X_train = []
    y_train = []
    for batch in tqdm(train_loader):
        images, targets = batch
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model.forward_encoder(images)
        X_train.append(outputs.cpu()[0].numpy())
        y_train.append(targets.cpu()[0].numpy())

    torch.cuda.empty_cache()
    # Validation phase
    X_test = []
    y_test = []
    for batch in tqdm(val_loader):
        images, targets = batch
        images, targets = images.to(device), targets.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model.forward_encoder(images)
        X_test.append(outputs.cpu()[0].numpy())
        y_test.append(targets.cpu()[0].numpy())

    # Convert lists to numpy arrays
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Convert to pandas DataFrames
    df_X_train = pd.DataFrame(X_train)
    df_y_train = pd.DataFrame(y_train, columns=['target'])
    df_X_test = pd.DataFrame(X_test)
    df_y_test = pd.DataFrame(y_test, columns=['target'])

    results_folder = f'modelling/dino/results/split_{mode}{imagery_source}_{fold}/'
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # Save to CSV files
    df_X_train.to_csv(results_folder+'X_train.csv', index=False)
    df_y_train.to_csv(results_folder+'y_train.csv', index=False)
    df_X_test.to_csv(results_folder+'X_test.csv', index=False)
    df_y_test.to_csv(results_folder+'y_test.csv', index=False)

    alphas = np.logspace(-6, 6, 20)
    # Define the model and pipeline
    ridge_pipeline = Pipeline([
        # ('scaler', StandardScaler()),
        ('ridge', RidgeCV(alphas=alphas, cv=5, scoring='neg_mean_absolute_error'))
    ])

    # Define the cross-validation strategy
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform cross-validation
    cv_scores = cross_val_score(ridge_pipeline, X_train, y_train, cv=kf, scoring='neg_mean_absolute_error')

    # Print the cross-validation scores
    print("Cross-validation scores (negative MAE):", cv_scores)
    print("Mean cross-validation score (negative MAE):", cv_scores.mean())

    ridge_pipeline.fit(X_train, y_train)
    test_score= np.mean(np.abs(ridge_pipeline.predict(X_test)- y_test))
    print("Test Score (negative MAE):", test_score)

    return test_score





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run satellite image processing model training.')
    parser.add_argument('--fold', type=str, default='1', help='The fold number')
    parser.add_argument('--model_name', type=str, default='dinov2_vitb14', help='The model name')
    parser.add_argument('--target', type=str,default='', help='The target variable')
    parser.add_argument('--imagery_source', type=str, default='L', help='L for Landsat and S for Sentinel')
    parser.add_argument('--imagery_path', type=str, help='The parent directory of all imagery')
    parser.add_argument('--mode', type=str, default='temporal', help='Evaluating temporal model or spatial model')
    parser.add_argument('--model_output_dim', type=int, default=768, help='The output dimension of the model')
    parser.add_argument('--use_checkpoint', action='store_true', help='Whether to use checkpoint file. If not, use raw model.')
    parser.add_argument('--model_not_named_target', action='store_false', help='Whether the model name contains the target variable')

    
    args = parser.parse_args()
    maes = []
    if args.mode == 'temporal':
        print(evaluate("1", args.model_name,args.target, args.use_checkpoint,args.model_not_named_target, args.imagery_path, args.imagery_source, args.mode,  args.model_output_dim))
    elif  'spatial' in args.mode:
        for i in range(5):
            fold = i + 1
            mae = evaluate(str(fold), args.model_name, args.target, args.use_checkpoint,args.model_not_named_target,args.imagery_path, args.imagery_source, args.mode, args.model_output_dim)
            maes.append(mae)
        print(np.mean(maes), np.std(maes)/np.sqrt(5))
    elif args.mode == 'one_country':
        COUNTRIES = ['Madagascar', 'Burundi', 'Uganda', 'Mozambique', 'Rwanda',
                    'Zambia', 'Tanzania', 'Malawi', 'Ethiopia', 'Kenya', 'Zimbabwe',
                    'Lesotho', 'South Africa', 'Angola', 'Eswatini', 'Comoros']
        
        n_samples = len(COUNTRIES)
        for country in COUNTRIES:
            try:
                mae = evaluate(country, args.model_name, args.target, args.use_checkpoint,args.model_not_named_target,args.imagery_path, args.imagery_source, args.mode, args.model_output_dim)
                maes.append(mae)
            except Exception as e:
                print(f"Error in {country}: {e}")
                n_samples -= 1
        print(np.mean(maes), np.std(maes)/np.sqrt(n_samples))
    else:
        raise Exception("Invalid mode")
