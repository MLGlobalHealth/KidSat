import numpy as np
import rasterio

if imagery_source == 'L':
    normalization = 30000.
    imagery_size = 336
elif imagery_source == 'S':
    normalization = 3000.
    imagery_size = 994
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