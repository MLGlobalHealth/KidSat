"""
This script is designed for various operations using data collections from Google Earth Engine,
specifically focusing on Landsat 8 data. It includes imports for handling data arrays (numpy),
Earth Engine operations (ee), JSON file parsing, data manipulation (pandas), and utility functions
like warnings and time tracking. It also utilizes tqdm for progress tracking during loops.

The Landsat 8 collections included are primarily raw image data (Tier 1, Tier 1 + Real-Time, and Tier 2).

Author: Fan Yang
Date: 2024-02-17

Reference:
- Google Earth Engine Dataset Catalog for Landsat: https://developers.google.com/earth-engine/datasets/catalog/landsat
"""

import numpy as np
import ee
import json
import pandas as pd
import geopandas as gpd
import warnings
import time
from tqdm import tqdm
from collections import Counter

# data collections from google earth
# from https://developers.google.com/earth-engine/datasets/catalog/landsat
LANDSAT8_COLLECTIONS = [ # 2013 - Now
    "LANDSAT/LC08/C02/T1_L2",        # Raw Image Tier 1
    "LANDSAT/LC08/C01/T1_RT",     # Raw Image Tier 1 + Read-Time
    "LANDSAT/LC08/C01/T2",        # Raw Image Tier 2
    # 'LANDSAT/LC08/C01/T1_L2',     # Surface reflecttance Tier 1
    # "LANDSAT/LC08/C01/T2_L2",     # Surface reflecttance Tier 2
    # "LANDSAT/LC08/C01/T1_TOA",    # Top of Atomsphere Tier 1
    # "LANDSAT/LC08/C01/T1_RT_TOA", # Top of Atomsphere Tier 1 + Read-Time
    # "LANDSAT/LC08/C01/T2_TOA",    # Top of Atomsphere Tier 2
]

LANDSAT7_COLLECTIONS = [ # 1999 - 2021
    "LANDSAT/LE07/C02/T1_L2",        # Raw Image Tier 1
    'LANDSAT/LE07/C01/T2',        # Raw Image Tier 2
    # 'LANDSAT/LE07/C01/T1_L2',     # Surface Reflectance Tier 1
    # 'LANDSAT/LE07/C01/T2_L2',     # Surface Reflectance Tier 2
    # 'LANDSAT/LE07/C01/T1_TOA',    # Top of Atmosphere Tier 1
    # 'LANDSAT/LE07/C01/T2_TOA',    # Top of Atmosphere Tier 2
]

LANDSAT5_COLLECTIONS = [ # 1984 - 2012
    "LANDSAT/LT05/C02/T1_L2",        # Raw Image Tier 1 (Collection 1)
    'LANDSAT/LT05/C01/T2',        # Raw Image Tier 2 (Collection 1)
    # 'LANDSAT/LT05/C01/T1_L2',     # Surface Reflectance Tier 1 (Collection 1)
    # 'LANDSAT/LT05/C01/T2_L2',     # Surface Reflectance Tier 2 (Collection 1)
    # 'LANDSAT/LT05/C01/T1_TOA',    # Top of Atmosphere Tier 1 (Collection 1)
    # 'LANDSAT/LT05/C01/T2_TOA',    # Top of Atmosphere Tier 2 (Collection 1)
]

LANDSAT9_COLLECTIONS = [ # 2021 - Now
    'LANDSAT/LC09/C01/T1',        # Raw Image Tier 1 (Collection 1)
    'LANDSAT/LC09/C01/T2',        # Raw Image Tier 2 (Collection 1)
]

SENTINEL2_COLLECTIONS = [ # 2015 - Now
    "COPERNICUS/S2",
    "COPERNICUS/S2_HARMONIZED",
    "COPERNICUS/S2_SR_HARMONIZED",
]
SENSORS = {
    'L5' : LANDSAT5_COLLECTIONS,
    'L7' : LANDSAT7_COLLECTIONS,
    'L8' : LANDSAT8_COLLECTIONS,
    'L9' : LANDSAT9_COLLECTIONS,
    'S2' : SENTINEL2_COLLECTIONS
}

def create_rectangle(center_lat, center_lon, width_km, height_km):
    """
    Creates a rectangle around a center point.

    Parameters:
    - center_lat (float): Latitude of the center point.
    - center_lon (float): Longitude of the center point.
    - width_km (float): Width of the rectangle in kilometers.
    - height_km (float): Height of the rectangle in kilometers.

    Returns:
    - ee.Geometry.Rectangle: A rectangle represented as an Earth Engine Geometry object.
    """
    km_per_degree = 111
    delta_lat = height_km / 2 / km_per_degree
    delta_lon = width_km / 2 / (km_per_degree * np.cos(np.radians(center_lat)))

    lower_left = [center_lon - delta_lon, center_lat - delta_lat]
    upper_right = [center_lon + delta_lon, center_lat + delta_lat]

    return ee.Geometry.Rectangle([lower_left, upper_right])

def create_square(center_lat, center_lon, side_length_km):
    """
    Creates a square around a center point.

    Parameters:
    - center_lat (float): Latitude of the center point.
    - center_lon (float): Longitude of the center point.
    - side_length_km (float): Side length of the square in kilometers.

    Returns:
    - ee.Geometry.Rectangle: A square represented as an Earth Engine Geometry object.
    """
    return create_rectangle(center_lat, center_lon, side_length_km, side_length_km)

def get_project_name(config_filepath='config/google_config.json'):
    """
    Retrieves the project name from a configuration file.

    Parameters:
    - config_filepath (str): Path to the configuration file.

    Returns:
    - str: The project name.
    """
    with open(config_filepath, 'r') as file:
        data = json.load(file)
    return data['project']

def get_column_name(df, substring, exclude_pattern = None):
    """
    Finds a column name in a DataFrame that contains a given substring.

    Parameters:
    - df (pd.DataFrame): The DataFrame to search.
    - substring (str): The substring to look for in column names.

    Returns:
    - str: The name of the column that contains the substring.
    """
    columns = df.columns
    for c in columns:
        if substring.lower() in c.lower():
            if exclude_pattern == None or not exclude_pattern.lower() in c.lower():
                return c
    return None

def download_imagery(filepath, drive, year, sensor, range_km, rgb_only, parallel = True, verbose = False):
    """
    Downloads satellite imagery for specified locations and parameters.

    Parameters:
    - filepath (str): Path to the CSV file containing locations.
    - drive (str): Google Drive folder name where images will be saved.
    - year (str): Year for which to download imagery.
    - sensor (str): Sensor code ('L5', 'L7', 'L8', 'L9', 'S2') indicating the imagery source.
    - range_km (float): Range in kilometers to define the area around each location.
    - rgb_only (bool): Whether get only RBG bands for the image

    Raises:
    - NotImplementedError: If an unsupported sensor is requested.
    """
    ee.Authenticate()
    project_name = get_project_name()
    ee.Initialize(project = project_name)

    is_csv = filepath[-4:] == '.csv'

    if is_csv:
        target_df = pd.read_csv(filepath)
    else:
        target_df = gpd.read_file(filepath)
    # in case someone uses latitude as colname
    lat_colname = get_column_name(target_df, 'lat')
    lon_colname = get_column_name(target_df, 'lon')

    if (lat_colname != 'lat' or lon_colname != 'lon') and verbose:
        warnings.warn(lat_colname +" and "+ lon_colname + ' columns are used as lat and lon inputs. Please check whether this is correct')
    if is_csv:
        name_colname = get_column_name(target_df, 'name', exclude_pattern='Unnamed')
    else:
        name_colname = get_column_name(target_df, 'DHSID')
    start_date = year + '-01-01'
    end_date = year + '-12-30'
    if sensor not in SENSORS:
        raise NotImplementedError('The requested sensor has not been implemented.')
    else:
        image_collection = SENSORS[sensor][0] # default raw image
    
    for i in tqdm(range(len(target_df))):
        
        region = create_square(
            target_df[lat_colname][i],
            target_df[lon_colname][i],
            range_km
        )
        if sensor[0] == 'L':
            resolution_m = 30
            cloud_filter = 'CLOUD_COVER'
        elif sensor[0] == 'S':
            resolution_m = 10
            cloud_filter = 'CLOUDY_PIXEL_PERCENTAGE'
        else:
            raise (NotImplementedError)
        cloudy_pixel_percentage_threshold = 20
        collection_size = 0
        while collection_size == 0 and cloudy_pixel_percentage_threshold<=100:
            collection = ee.ImageCollection(image_collection) \
                .filterBounds(region) \
                .filterDate(start_date, end_date) \
                .filter(ee.Filter.lt(cloud_filter, cloudy_pixel_percentage_threshold))
            cloudy_pixel_percentage_threshold+=10
            # Check if the collection is empty
            collection_size = collection.size().getInfo()
        image = collection.median()
        # if cloudy_pixel_percentage_threshold==110:
        #     print(cloudy_pixel_percentage_threshold)

        if rgb_only:
            if 'T2' in image_collection:
                image = image.select(['SR_B4', 'SR_B3', 'SR_B2'])
            else:
                image = image.select(['B4', 'B3', 'B2'])

        
 
        export_params = {
            'description': str(target_df[name_colname][i]),
            'folder': drive,
            'scale': resolution_m,  # This is the resolution in meters
            'region': region,
            'fileFormat': 'GeoTIFF',
            'maxPixels': 1e10
        }

        export_task = ee.batch.Export.image.toDrive(image, **export_params)
        export_task.start()
        
        if not parallel:
            while export_task.status()['state'] in ['READY', 'RUNNING']:
                time.sleep(1)
            if export_task.status()['state'] == 'FAILED':
                print(export_task.status())
                error_msg = 'Image.clipToBoundsAndScale: Parameter \'input\' is required.'
                if export_task.status()['error_message'] == error_msg:
                    warnings.warn("The dataset does not have the imagery given the filters. Try another timespan, coordinates, or sensor.")
