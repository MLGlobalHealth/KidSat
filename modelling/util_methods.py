import rasterio
import numpy as np
import torch
import pandas as pd
import os
import re
import random
from sklearn.model_selection import train_test_split, KFold
import torchvision.transforms.v2 as transforms
from torch.utils.data import Dataset
from PIL import Image


def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, filename)

def load_and_preprocess_image(path, landsat=True):
    with rasterio.open(path) as src:
        img = src.read([4,3,2]).transpose(1,2,0)
        if landsat:
            img = img / 30000. *255.  # Normalize to [0, 1] (if required)
        else:
            img = img / 3000. * 255.
        
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


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform, predict_target, landsat=True):
        self.dataframe = dataframe
        self.transform = transform
        self.target = predict_target
        self.landsat = landsat
        self.cache = {}
        print("Using", "landsat" if landsat else "sentinel")

    def __len__(self):
        return len(self.dataframe)

    def load_item(self, idx):
        if idx in self.cache:
            return
        item = self.dataframe.iloc[idx]

        image = load_and_preprocess_image(item["imagery_path"], landsat=self.landsat)
        # Apply feature extractor if necessary, might need adjustments
        image_tensor = self.transform(Image.fromarray(image))
        # Assuming your target is a single scalar
        target = torch.tensor(item[self.target], dtype=torch.float32)
        self.cache[idx] = (image_tensor, target)

    def __getitem__(self, idx):
        self.load_item(idx)
        return self.cache[idx]  # Adjust based on actual output of feature_extractor

class CustomTemporalDataset(CustomDataset):
    def load_item(self, idx):
        if idx in self.cache:
            return
            
        item = self.dataframe.iloc[idx]

        image = load_and_preprocess_image(item["imagery_path"], landsat=self.landsat)
        # Apply feature extractor if necessary, might need adjustments
        image_tensor = self.transform(Image.fromarray(image))
        # Assuming your target is a single scalar
        target = torch.tensor(item[self.target], dtype=torch.float32)
        timestamp = torch.tensor([item["YEAR"] - 2002, 0, 0])

        image_tensor = image_tensor[None].expand((3,) + image_tensor.shape)
        timestamp = timestamp[None].expand((3, 3))
        self.cache[idx] = (image_tensor, timestamp, target)  # Adjust based on actual output of feature_extractor



def get_datasets(dhs_path, imagery_path, predict_target, temporal=False, split=True, seed=42, landsat=True, train=True):
    print("Dataset path:", dhs_path)
    df = pd.read_csv(dhs_path, index_col=0)
    if temporal:
        df = df[df["YEAR"] <= 2019] if train else df[df["YEAR"] >= 2020]
    available_imagery = []
    for d in os.listdir(imagery_path):
        if (landsat and d[-2] == 'L') or (not landsat and d[-2] == 'S'):
            for f in os.listdir(os.path.join(imagery_path, d)):
                available_imagery.append(os.path.join(imagery_path, d, f))
    print(available_imagery[:2])
    available_centroids = [f.split('/')[-1][:-4] for f in available_imagery]

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

    df = df[df['CENTROID_ID'].isin(available_centroids)]
    print(len(df))
    

    df['imagery_path'] = df['CENTROID_ID'].apply(filter_contains)
    print(len(df))


    filtered_predict_target = []
    for col in predict_target:
        filtered_predict_target.extend(
            [c for c in df.columns if c == col or re.match(f"^{col}_[^a-zA-Z]", c)]
        )
    # Drop rows with NaN values in the filtered subset of columns
    df = df.dropna(subset=filtered_predict_target)
    print(len(df))
    
    predict_target = sorted(filtered_predict_target)
    assert len(predict_target) in [99, 1]
    # assert len(df) == 14068

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the image to the input size expected by the model
        transforms.ToTensor(),  # Convert the image to a PyTorch tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize with ImageNet's mean and std
    ])

    dataset_class = CustomTemporalDataset if temporal else CustomDataset
    if split:
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=seed)
    
        train_dataset = dataset_class(train_df, transform, predict_target, landsat=landsat)
        val_dataset = dataset_class(val_df, transform, predict_target, landsat=landsat)
        return train_dataset, val_dataset, len(predict_target)
    else:
        return dataset_class(df, transform, predict_target, landsat=landsat), len(predict_target)
