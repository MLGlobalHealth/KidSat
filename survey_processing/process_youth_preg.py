"""
Author: Luke Yang
Date: 2024-08-18

This script processes DHS data to generate indicators of poverty and deprivation.

Usage:
    python main.py <parent_dir> [config_file]

Arguments:
    parent_dir: The parent directory enclosing all DHS folders.
    config_file (optional): The configuration parameters for preprocessing. Defaults to 'processing_params.json' if not provided.

Functions:
    find_sub_file(directory, pattern: str) -> str:
        Finds and returns the filename in a directory that matches a given pattern.

    get_poverty(source_path: str, save_csv: bool = False) -> pd.DataFrame:
        Processes DHS data files to generate various deprivation indicators and optionally saves the result as a CSV file.

        Parameters:
            source_path (str): The path to the directory containing the DHS data files.
            save_csv (bool): If True, saves the resulting DataFrame to a CSV file in the source directory.

        Returns:
            pd.DataFrame: The processed data with deprivation indicators.
"""

import argparse
import os
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
import json
import geopandas as gpd
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import KFold

# Ignore all warnings
warnings.filterwarnings('ignore')
par_dir = r'survey_processing/'
def find_sub_file(directory, pattern:str):
    for f in os.listdir(directory):
      if pattern.lower() in f.lower():
        return f

def make_string(integer, length = 8):
    return str(integer).zfill(length)

# process data
def get_youth_preg(source_path):
   
    for f in os.listdir(source_path):
        if 'IR' in f:
            individual_datafile = os.path.join(source_path, f, find_sub_file(source_path+f,'dta'))
    survey_year = source_path.split('/')[-2][3:7]
    country_code = source_path.split('/')[-2][:2]

    dhs_ir = pd.read_stata(individual_datafile, convert_categoricals=False)
    dhs_ir['id'] = country_code[:2]+survey_year+ dhs_ir['v004'].apply(make_string)

    youth_df = dhs_ir[dhs_ir['v013'] == 1] ## age group 15-19
    youth_df['exposed_preg'] = ((youth_df['v213'] > 0) | (youth_df['v201'] > 0) | (youth_df['v228'] == 1)).astype(int)
    youth_df = youth_df.groupby('id').mean(numeric_only=True).reset_index()
    return youth_df


def process_dhs(parent_dir, config_file):
    if parent_dir[-1]!='/':
        parent_dir+=r'/'
    with open(config_file, 'r') as file:
        config_data = json.load(file)
    with open(f'{par_dir}dhs_country_code.json', 'r') as file:
        dhs_cc = json.load(file)
    root_grid = parent_dir
    youth_dfs = []
    print('Summarizing youth pregnancy...')
    for f in tqdm(os.listdir(root_grid)):
        if 'DHS' in f:
            try:
                youth_df = get_youth_preg(root_grid+f+'/')
                # youth_df = youth_df.loc[:, ~youth_df.columns.str.contains('_')]
                youth_df=youth_df[['id','exposed_preg']]
                youth_dfs.append(youth_df)
            except Exception as e:
                pass
                # print(e)
                


    dhs_df_all = pd.concat(youth_dfs)

    ## Calculating Centroids
    print('Calculating geospatial variables...')
    gdfs = []
    for f in tqdm(os.listdir(root_grid)):
        if 'DHS' in f:
            for sub_f in os.listdir(os.path.join(root_grid,f)):
                if sub_f.__contains__('GE'):
                    shape_file = os.path.join(root_grid, f, sub_f)
                    gdf = gpd.read_file(shape_file)
                    # Append to the list of GeoDataFrames
                    gdfs.append(gdf)
                    # print(gdf.shape)
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    def country_code_to_name(country_code):
        return dhs_cc[country_code]
    print('Generating columns...')
    combined_gdf['COUNTRY'] = combined_gdf['DHSCC'].apply(country_code_to_name)
    combined_gdf['SURVEY_NAME'] = [combined_gdf.iloc[i]['COUNTRY']+'_DHS_'+str(int(combined_gdf.iloc[i]['DHSYEAR'])) for i in range(combined_gdf.shape[0])]
    combined_gdf['YEAR'] =combined_gdf['DHSYEAR'].apply(int)
    combined_gdf['CENTROID_ID']  = combined_gdf['DHSID']

    centroid_df = combined_gdf[['CENTROID_ID', 'SURVEY_NAME', 'COUNTRY','YEAR', 'LATNUM', 'LONGNUM']]
    centroid_df = centroid_df[~((centroid_df['LATNUM'] == 0) & (centroid_df['LONGNUM'] == 0))]
    centroid_df.drop_duplicates(inplace=True)
    dhs_df_all.drop_duplicates(inplace=True)
    centroid_df = centroid_df.reset_index()
    merged_centroid_df = pd.merge(centroid_df, dhs_df_all, left_on='CENTROID_ID', right_on='id', how='left')
    # print(merged_centroid_df.shape)
    print('Spliting data...')

    save_split(merged_centroid_df)
    

def save_split(df):
    save_par_dir = r'survey_processing/processed_data/'
    df.to_csv(f'{save_par_dir}dhs_variables.csv', index=False)
    df = df.sample(frac=1, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for train_index, test_index in tqdm(kf.split(df)):
        # Generate train and test subsets
        train_df = df.iloc[train_index]
        test_df = df.iloc[test_index]
        
        # Save to CSV files
        train_df.to_csv(f'{save_par_dir}train_fold_{fold}.csv', index=False)
        test_df.to_csv(f'{save_par_dir}test_fold_{fold}.csv', index=False)
        
        fold += 1
    
    old_df = df[df['YEAR'] < 2020]
    new_df = df[df['YEAR'] >= 2020]
    new_df.to_csv(f'{save_par_dir}after_2020.csv', index=False)
    old_df.to_csv(f'{save_par_dir}before_2020.csv', index=False)

def main():

    # Setup argument parser
    parser = argparse.ArgumentParser(description="Process DHS data to a single CSV file.")
    parser.add_argument("parent_dir", help="The parent directory enclosing all DHS folders")
    parser.add_argument("config_file", nargs='?', default=f'{par_dir}processing_params.json', help="The configuration parameters for preprocessing (default: processing_params.json)")
    # Parse arguments
    args = parser.parse_args()

    # Call the download function with the parsed arguments
    process_dhs(args.parent_dir, args.config_file)

if __name__ == "__main__":
    main()
