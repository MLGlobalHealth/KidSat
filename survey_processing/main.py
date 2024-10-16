"""
Author: Luke Yang
Date: 2024-06-11

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

# process data
def get_poverty(source_path, save_csv = False):
   
    for f in os.listdir(source_path):
        if 'KR' in f:
            child_datafile = os.path.join(source_path, f, find_sub_file(source_path+f,'dta'))
        elif 'PR' in f:
            household_datafile = os.path.join(source_path, f, find_sub_file(source_path+f,'dta'))
        elif 'IR' in f:
            individual_datafile = os.path.join(source_path, f, find_sub_file(source_path+f,'dta'))
    survey_year = source_path.split('/')[-2][3:7]
    country_code = source_path.split('/')[-2][:2]

    dhs_kr = pd.read_stata(child_datafile, convert_categoricals=False)
    # Save the final merged dataset
    if save_csv:
        output_path = source_path+'dhs_variables.csv'
        dhs_kr.to_csv(output_path, index=False)
    # Filter out certain cases
    dhs_kr = dhs_kr[(dhs_kr['b16'].notna()) & (dhs_kr['b16'] != 0)]
    dhs_kr['chweight'] = dhs_kr['v005'] / 1000000

    # Check if b19 exists, otherwise take information of age of child from hw1
    if 'b19' not in dhs_kr.columns:
        dhs_kr['b19'] = dhs_kr['hw1']
    # Prepare for merge with HH member module
    dhs_kr = dhs_kr.drop_duplicates(subset=['v001', 'v002', 'b16'])
    dhs_kr['hv001'] = dhs_kr['v001']
    dhs_kr['hv002'] = dhs_kr['v002']
    dhs_kr['hvidx'] = dhs_kr['b16']

    # Save the final merged dataset
    if save_csv:
        output_path = source_path+'dhs_variables.csv'
        dhs_kr.to_csv(output_path, index=False)

    dhs_pr = pd.read_stata(household_datafile, convert_categoricals=False)
    dhs_pr.set_index(['hv001', 'hv002', 'hvidx'], inplace=True)
    dhs_kr = dhs_kr.merge(dhs_pr[['hv000', 'hv005', 'hv007', 'hv009', 'hv024', 'hv025', 'hv104', 'hv105', 'hv109', 'hv121', 'hv122', 'hv201', 'hv204', 'hv205', 'hv216', 'hv225', 'hv270', 'hv271', 'hc70']], left_on=['hv001', 'hv002', 'hvidx'], right_index=True, how='outer')
    dhs_kr['hhweight'] = dhs_kr['hv005'] / 1000000


    # Save the final merged dataset
    if save_csv:
        output_path = source_path+'dhs_variables.csv'
        dhs_kr.to_csv(output_path, index=False)

    # Prepare for another merge with individual member data
    dhs_kr.drop(['v001', 'v002', 'v003'], axis=1, inplace=True)
    dhs_kr['v001'] = dhs_kr['hv001']
    dhs_kr['v002'] = dhs_kr['hv002']
    dhs_kr['v003'] = dhs_kr['hvidx']

    dhs_ir = pd.read_stata(individual_datafile, convert_categoricals=False)
    dhs_ir.set_index(['v001', 'v002', 'v003'], inplace=True)
    dhs_kr = dhs_kr.merge(dhs_ir[['v626a', 'v312']], left_on=['v001', 'v002', 'v003'], right_index=True,  how='outer')
    dhs_kr['v626a']=dhs_kr['v626a_x']
    dhs_kr['v312']=dhs_kr['v312_x']
    # Save the final merged dataset
    if save_csv:
        output_path = source_path+'dhs_variables.csv'
        dhs_kr.to_csv(output_path, index=False)
    df = dhs_kr
    print(df.v)
    ## 1
    # Assuming 'df' is your DataFrame
    df['personsperroom'] = df['hv009'] / df['hv216']
    df = df[df['hv105'] < 18]  # Dropping adults

    # Generate severe housing deprivation flag
    df['dep_housing_sev'] = (df['personsperroom'] >= 5).astype(int)

    # Generate moderate housing deprivation flag
    df['dep_housing_mod'] = (df['personsperroom'] >= 3).astype(int)

    ## 2
    df['dep_water_sev'] = 0  # Default to 0 (no severe deprivation)
    df.loc[df['hv201'].isin([32, 42, 43, 96]), 'dep_water_sev'] = 1  # Recode specific values to 1
    df.loc[df['hv201'] == 99, 'dep_water_sev'] = pd.NA  # Set specific values to NaN

    # Generate 'dep_water_mod' as a copy of 'dep_water_sev'
    df['dep_water_mod'] = df['dep_water_sev']

    # Update 'dep_water_mod' for moderate water deprivation conditions
    # Here it's important to ensure no overwrite of previously set severe conditions
    mask_mod = (df['dep_water_mod'] == 0) & (~df['hv201'].isin([32, 42, 43, 96])) & (df['hv204'] > 30) & (df['hv204'] <= 900)
    df.loc[mask_mod, 'dep_water_mod'] = 1

    ## 3
    df['dep_sanitation_sev'] = 0  # Default to 0 (no severe deprivation)
    df.loc[df['hv205'].isin([23, 31, 42, 43, 96]), 'dep_sanitation_sev'] = 1  # Recode specific values to 1
    df.loc[df['hv205'] == 99, 'dep_sanitation_sev'] = pd.NA  # Set specific values to NaN

    # Generate 'dep_sanitation_mod' as a copy of 'dep_sanitation_sev'
    df['dep_sanitation_mod'] = df['dep_sanitation_sev']

    # Update 'dep_sanitation_mod' for moderate sanitation deprivation conditions
    # Here it's important to ensure no overwrite of previously set severe conditions
    mask_mod_sanitation = (df['dep_sanitation_mod'] == 0) & (~df['hv205'].isin([23, 31, 42, 43, 96])) & (df['hv225'] == 1)
    df.loc[mask_mod_sanitation, 'dep_sanitation_mod'] = 1

    ## 4

    df['dep_nutrition_sev'] = 0  # Initialize with 0 (no severe deprivation)
    df.loc[df['hc70'] <= -300, 'dep_nutrition_sev'] = 1  # Set to 1 where severe stunting occurs

    # Generate moderate nutrition deprivation based on stunting more than -2 standard deviations
    df['dep_nutrition_mod'] = 0  # Initialize with 0 (no moderate deprivation)
    df.loc[df['hc70'] <= -200, 'dep_nutrition_mod'] = 1  # Set to 1 where moderate stunting occurs


    ## 5

    # Assume df is your DataFrame
    age_filter = (df['b19'] >= 12) & (df['b19'] <= 35)  # Children between 12 to 35 months

    # DPT 1 Deprivation
    df['dpt1deprived'] = 0  # Initialize column
    df.loc[age_filter & ((df['h10'] == 0) | (df['h3'] == 0)), 'dpt1deprived'] = 1
    df.loc[age_filter & (df['h3'].between(1, 3)), 'dpt1deprived'] = 0

    # DPT 2 Deprivation
    df['dpt2deprived'] = 0  # Initialize column
    df.loc[age_filter & ((df['h10'] == 0) | (df['h5'] == 0)), 'dpt2deprived'] = 1
    df.loc[age_filter & (df['h5'].between(1, 3)), 'dpt2deprived'] = 0

    # DPT 3 Deprivation
    df['dpt3deprived'] = 0  # Initialize column
    df.loc[age_filter & ((df['h10'] == 0) | (df['h7'] == 0)), 'dpt3deprived'] = 1
    df.loc[age_filter & (df['h7'].between(1, 3)), 'dpt3deprived'] = 0

    # Measles Deprivation
    df['measlesdeprived'] = 0  # Initialize column
    df.loc[age_filter & ((df['h10'] == 0) | (df['h9'] == 0)), 'measlesdeprived'] = 1
    df.loc[age_filter & (df['h9'].between(1, 3)), 'measlesdeprived'] = 0

    # Before aggregating, reorder columns to send new variables to the end
    column_order = [col for col in df.columns if col not in ['dpt1deprived', 'dpt2deprived', 'dpt3deprived', 'measlesdeprived']] + \
                  ['dpt1deprived', 'dpt2deprived', 'dpt3deprived', 'measlesdeprived']
    df = df[column_order]

    # Count missing values across the immunization indicators
    df['hasmissvaccines'] = df[['dpt1deprived', 'dpt2deprived', 'dpt3deprived', 'measlesdeprived']].isnull().sum(axis=1)

    # Sum up the indicators to get total vaccines missed
    df['sumvaccines'] = df[['dpt1deprived', 'dpt2deprived', 'dpt3deprived', 'measlesdeprived']].sum(axis=1)

    # Adjust for rows where any vaccines data is missing
    df.loc[df['hasmissvaccines'].between(1, 4), 'sumvaccines'] = pd.NA

    # Generate moderate deprivation based on missing any of the four immunizations
    df['moderatevaccinesdeprived'] = 0  # Initialize with 0
    df.loc[df['sumvaccines'].between(1, 4), 'moderatevaccinesdeprived'] = 1

    # Generate severe deprivation if all four vaccines are missing
    df['severevaccinesdeprived'] = 0  # Initialize with 0
    df.loc[df['sumvaccines'] == 4, 'severevaccinesdeprived'] = 1

    # Identifying ARI symptoms in children aged 36 to 59 months
    df['arisymptoms'] = 0  # Initialize the column
    # Set arisymptoms to 1 based on UNICEF's definition of ARI
    df.loc[(df['h31'] == 2) & (df['h31b'] == 1) & (df['h31c'].isin([1, 3])) & (df['b19'].between(36, 59)), 'arisymptoms'] = 1

    # Severe threshold: Child had ARI symptoms and no treatment was sought
    df['ariseverelydeprived'] = 0  # Initialize the column
    df.loc[(df['arisymptoms'] == 1) & (df['h32y'].isin([1, 8, pd.NA])), 'ariseverelydeprived'] = 1
    df.loc[(df['arisymptoms'] == 1) & (df['h32y'] == 0), 'ariseverelydeprived'] = 0

    # Moderate (+severe) threshold: Child had ARI symptoms, and no treatment was sought at an appropriate medical facility
    df['arimoderatedeprived'] = 0  # Initialize the column
    df.loc[(df['arisymptoms'] == 1) & (df['h32z'] == 0), 'arimoderatedeprived'] = 1
    df.loc[(df['arisymptoms'] == 1) & (df['h32z'] == 1), 'arimoderatedeprived'] = 0

    # Treatment at inappropriate facilities could be handled as below if specifics were provided:
    # Assuming h32z values for inappropriate facilities are explicitly defined or derived elsewhere in your data context
    # Define inappropriate treatment facilities (example values)
    inappropriate_facilities = [5110, 9995, 2300, 9998]
    df.loc[(df['ariseverelydeprived'] == 0) & df['h32z'].isin(inappropriate_facilities), 'arimoderatedeprived'] = 1

    # Filter for girls aged 15 to 17 who have unmet needs for family planning
    age_filter = (df['hv105'] >= 15) & (df['hv105'] <= 17)
    need_filter = (df['v626a'] >= 1) & (df['v626a'] <= 4)  # Assuming these codes indicate the need for family planning

    # Severe deprivation: Girls who do not want to become pregnant but are not using contraception
    df['contramethodseverelydep'] = 0  # Initialize column
    df.loc[age_filter & need_filter & (df['v312'] == 0), 'contramethodseverelydep'] = 1
    df.loc[age_filter & need_filter & (df['v312'] == 99), 'contramethodseverelydep'] = pd.NA  # Handling missing data as NaN

    # Moderate (+ severe) deprivation: Includes girls using traditional methods of contraception
    df['contramethodmoderatedep'] = 0  # Initialize column
    traditional_methods = [8, 9, 10]  # Assuming these codes indicate traditional methods
    df.loc[age_filter & need_filter & (df['v312'].isin([0] + traditional_methods)), 'contramethodmoderatedep'] = 1
    df.loc[age_filter & need_filter & (df['v312'] == 99), 'contramethodmoderatedep'] = pd.NA  # Handling missing data as NaN

    # Reorder columns to place certain deprivation indicators at the end (optional, mostly for visualization)
    df = df[['severevaccinesdeprived', 'contramethodseverelydep', 'ariseverelydeprived'] + [col for col in df.columns if col not in ['severevaccinesdeprived', 'contramethodseverelydep', 'ariseverelydeprived']]]

    # Calculate the number of missing indicators for severe health deprivation
    df['hasmissseverehealth'] = df[['severevaccinesdeprived', 'contramethodseverelydep', 'ariseverelydeprived']].isnull().sum(axis=1)

    # Aggregate severe health deprivation indicators
    df['sumseverehealth'] = df[['severevaccinesdeprived', 'contramethodseverelydep', 'ariseverelydeprived']].sum(axis=1, min_count=1)  # min_count=1 ensures NaN if all are NaN

    # Exclude children missing in all severe health indicators
    df.loc[df['hasmissseverehealth'] == 3, 'sumseverehealth'] = pd.NA  # Set to NaN if all indicators are missing

    # Generate the severe health deprivation index
    df['severehealthdep'] = 0  # Default to 0 (not deprived)
    df.loc[df['sumseverehealth'] == 1, 'severehealthdep'] = 1  # Set to 1 if deprived in one or more indicators

    # Reorder columns for easier handling (optional)
    df = df[['moderatevaccinesdeprived', 'contramethodmoderatedep', 'arimoderatedeprived'] + [col for col in df.columns if col not in ['moderatevaccinesdeprived', 'contramethodmoderatedep', 'arimoderatedeprived']]]

    # Calculate the number of missing indicators for moderate health deprivation
    df['hasmissmoderatehealth'] = df[['moderatevaccinesdeprived', 'contramethodmoderatedep', 'arimoderatedeprived']].isnull().sum(axis=1)

    # Aggregate moderate health deprivation indicators
    df['summoderatehealth'] = df[['moderatevaccinesdeprived', 'contramethodmoderatedep', 'arimoderatedeprived']].sum(axis=1, min_count=1)  # min_count=1 ensures NaN if all are NaN

    # Exclude children missing in both indicators
    df.loc[df['hasmissmoderatehealth'] == 3, 'summoderatehealth'] = pd.NA  # Set to NaN if all indicators are missing

    # Generate the moderate health deprivation index
    df['moderatehealthdep'] = 0  # Default to 0 (not deprived)
    df.loc[df['summoderatehealth'] == 1, 'moderatehealthdep'] = 1  # Set to 1 if deprived in one or more indicators

    ## 6

    # Filter for the young cohort (5 to 14 years old)
    age_filter = (df['hv105'] >= 5) & (df['hv105'] <= 14)

    # Severe Educational Deprivation
    # Initial severe deprivation setup based on not attending school and no schooling level reached
    df['severeedudeprivedbelow15'] = 0
    df.loc[age_filter & (df['hv109'] == 0) & (df['hv121'] == 0), 'severeedudeprivedbelow15'] = 1
    df.loc[age_filter & (df['hv109'] == 0) & (df['hv121'] == 2), 'severeedudeprivedbelow15'] = 0
    df.loc[age_filter & (df['hv109'] >= 1) & (df['hv109'] <= 5), 'severeedudeprivedbelow15'] = 0

    # Moderate Educational Deprivation
    # Starts with severe deprivation setup
    df['moderateedudeprivedbelow15'] = df['severeedudeprivedbelow15']
    df.loc[age_filter & (df['hv121'] == 0), 'moderateedudeprivedbelow15'] = 1
    df.loc[age_filter & (df['hv121'] == 0) & (df['hv109'].between(4, 5)), 'moderateedudeprivedbelow15'] = 0
    df.loc[age_filter & (df['hv109'] == 0) & (df['hv122'] == 1), 'moderateedudeprivedbelow15'] = 1
    df.loc[age_filter & (df['hv109'] == 0) & ((df['hv121'] == 98) | df['hv121'].isna() | (df['hv122'] == 8)), 'moderateedudeprivedbelow15'] = 1

    # Handling missing cases - setting to NaN where there's insufficient data to determine deprivation
    missing_conditions = (df['hv109'].isna() | df['hv109'].isin([7, 8])) & df['hv121'].isna()
    df.loc[missing_conditions, 'moderateedudeprivedbelow15'] = pd.NA

    ## older cohort
    # Filter for the older cohort (15 to 17 years old)
    older_cohort_filter = (df['hv105'] >= 15) & (df['hv105'] <= 17)

    # Severe Educational Deprivation for older cohort
    df['severeedudeprived15older'] = 0  # Initialize with 0
    df.loc[older_cohort_filter & (df['hv121'] == 2), 'severeedudeprived15older'] = 0
    df.loc[older_cohort_filter & (df['hv109'] == 0) & (df['hv121'] == 0), 'severeedudeprived15older'] = 1
    df.loc[older_cohort_filter & (df['hv121'] == 0) & (df['hv109'] <= 1), 'severeedudeprived15older'] = 1
    df.loc[older_cohort_filter & (df['hv121'] == 0) & (df['hv109'] >= 2), 'severeedudeprived15older'] = 0
    df.loc[df['hv109'].isna() | (df['hv109'] == 8), 'severeedudeprived15older'] = pd.NA

    # Moderate Educational Deprivation for older cohort
    df['moderateedudeprived15older'] = df['severeedudeprived15older'].copy()
    df.loc[older_cohort_filter & (df['hv121'] == 2) & (df['hv122'] <= 1), 'moderateedudeprived15older'] = 1
    df.loc[older_cohort_filter & (df['hv121'] == 0) & (df['hv109'] < 4), 'moderateedudeprived15older'] = 1
    df.loc[older_cohort_filter & (df['hv121'] == 0) & (df['hv109'] >= 4), 'moderateedudeprived15older'] = 0
    df.loc[df['hv109'].isna() | (df['hv109'] == 8), 'moderateedudeprived15older'] = pd.NA

    # Aggregate severe deprivation across both age groups
    df['severeedudeprivedgroup'] = 0  # Initialize with 0
    df.loc[(df['severeedudeprivedbelow15'] == 1) | (df['severeedudeprived15older'] == 1), 'severeedudeprivedgroup'] = 1

    # Aggregate moderate deprivation across both age groups
    df['moderateedudeprivedgroup'] = 0  # Initialize with 0
    df.loc[(df['severeedudeprivedgroup'] == 1) | (df['moderateedudeprivedbelow15'] == 1) | (df['moderateedudeprived15older'] == 1), 'moderateedudeprivedgroup'] = 1

    ## final agg

    # Identifying missing data in moderate deprivation indicators
    moderate_columns = [col for col in df.columns if 'dep_' in col and '_mod' in col]
    df['hasmissmoderatepoor'] = df[moderate_columns].isnull().sum(axis=1)

    # Aggregating moderate deprivation indicators
    df['summoderatepoor'] = df[moderate_columns].sum(axis=1, min_count=1)  # Use min_count=1 to require at least one non-NA value

    # Discounting children missing in all moderate dimensions
    df.loc[df['hasmissmoderatepoor'] == 6, 'summoderatepoor'] = pd.NA

    # Determining final incidence of moderate child poverty
    df['moderatelydeprived'] = 0  # Default to not deprived
    df.loc[df['summoderatepoor'] >= 1, 'moderatelydeprived'] = 1

    # Identifying missing data in severe deprivation indicators
    severe_columns = [col for col in df.columns if 'dep_' in col and '_sev' in col]
    df['hasmissseverepoor'] = df[severe_columns].isnull().sum(axis=1)

    # Aggregating severe deprivation indicators
    df['sumseverepoor'] = df[severe_columns].sum(axis=1, min_count=1)

    # Discounting children missing in all severe dimensions
    df.loc[df['hasmissseverepoor'] == 6, 'sumseverepoor'] = pd.NA

    # Determining final incidence of severe child poverty
    df['severelydeprived'] = 0  # Default to not deprived
    df.loc[df['sumseverepoor'] >= 1, 'severelydeprived'] = 1

    # Set survey characteristics
    df['countrycode'] = country_code
    df['year'] = survey_year
    df['year_interview'] = df['hv007']
    df['survey'] = 'DHS'
    df['version'] = survey_year

    # Generate additional variables
    df['cluster'] = df['hv001']
    df['hhid'] = df['hv002']
    df['indid'] = df['hvidx']

    # Extract substrings for country code and survey round from a hypothetical variable hv000
    df['countrycode2'] = df['hv000'].str[:2]
    # Uncomment and adjust the following line if hv000 contains round information at a specific position
    # df['round'] = df['hv000'].str[-1]

    # Summarize hv007 to get the range
    year2_min = df['hv007'].min()
    year2_max = df['hv007'].max()
    df['year2'] = f"{year2_min}-{year2_max}"

    # Rename variables to make them more intuitive
    df.rename(columns={
        'hv025': 'location',
        'hv104': 'sex',
        'hv270': 'wealth',
        'hv271': 'wealthscore',
        'hv024': 'region',
        'hv105': 'age'
    }, inplace=True)

    # Optionally, handle renaming with try-except to catch errors if columns don't exist
    try:
        df.rename(columns={'b19': 'age_mo', 'hw1': 'age_mo'}, inplace=True)
    except KeyError as e:
        print(f"One or more columns not found: {e}")


    # Housing
    df.rename(columns={'severecrowdingdep': 'dep_housing_sev', 'moderatecrowdingdep': 'dep_housing_mod'}, inplace=True, errors='ignore')

    # Water
    df.rename(columns={'severewaterdep': 'dep_water_sev', 'moderatewaterdep': 'dep_water_mod'}, inplace=True, errors='ignore')

    # Sanitation
    df.rename(columns={'severesanitationdeprived': 'dep_sanitation_sev', 'moderatesanitationdeprived': 'dep_sanitation_mod'}, inplace=True, errors='ignore')

    # Nutrition
    df.rename(columns={'severestunting': 'dep_nutrition_sev', 'moderatestunting': 'dep_nutrition_mod'}, inplace=True, errors='ignore')

    # Initialize new columns for nutrition HAZ scores
    df['nutrition_haz'] = pd.NA
    df['nutrition_hazflag'] = pd.NA

    # Check if 'hc70' is in the DataFrame and use it to replace values in 'nutrition_haz'
    if 'hc70' in df.columns:
        df['nutrition_haz'] = df['hc70']  # Update nutrition_haz only if hc70 is within the specified range
        df.loc[~df['hc70'].between(-300, 900), 'nutrition_haz'] = pd.NA  # Assume the range condition needs to be applied

    # Renaming HAZ and HAZFLAG if they exist
    df.rename(columns={'HAZ': 'haz', 'HAZFLAG': 'hazflag'}, inplace=True, errors='ignore')

    # If 'haz' is in the DataFrame, update 'nutrition_haz' and optionally 'nutrition_hazflag'
    if 'haz' in df.columns:
        df['nutrition_haz'] = df['haz']
        if 'hazflag' in df.columns:
            df['nutrition_hazflag'] = df['hazflag']

    # Health Related Renaming and Variable Generation
    df.rename(columns={
        'severehealthdep': 'dep_health_sev',
        'moderatehealthdep': 'dep_health_mod',
        'severevaccinesdeprived': 'health_vac_sevdep',
        'moderatevaccinesdeprived': 'health_vac_moddep',
        'ariseverelydeprived': 'health_ari_sevdep',
        'arimoderatedeprived': 'health_ari_moddep',
        'contramethodseverelydep': 'health_con_sevdep',
        'contramethodmoderatedep': 'health_con_moddep'
    }, inplace=True, errors='ignore')

    # Initialize new health-related columns
    df['health_polio'] = pd.NA
    df['health_measles'] = pd.NA

    # Update measles based on measlesdeprived, if it exists
    if 'measlesdeprived' in df.columns:
        df['health_measles'] = df['measlesdeprived'].apply(lambda x: 0 if x == 1 else 1 if x == 0 else pd.NA)

    # Generate and update DPT related columns
    for i in range(1, 4):
        col_name = f'health_dpt{i}'
        deprived_col = f'dpt{i}deprived'
        df[col_name] = pd.NA
        if deprived_col in df.columns:
            df[col_name] = df[deprived_col].apply(lambda x: 0 if x == 1 else 1 if x == 0 else pd.NA)

    # Education Related Renaming
    df.rename(columns={
        'severeedudeprivedgroup': 'dep_education_sev',
        'moderateedudeprivedgroup': 'dep_education_mod',
        'severeedudeprivedbelow15': 'education_b15_sevdep',
        'moderateedudeprivedbelow15': 'education_b15_moddep',
        'severeedudeprived15older': 'education_15o_sevdep',
        'moderateedudeprived15older': 'education_15o_moddep'
    }, inplace=True, errors='ignore')

    # Renaming columns for summary indicators of severe and moderate poverty
    df.rename(columns={
        'sumseverepoor': 'sumpoor_sev',
        'summoderatepoor': 'sumpoor_mod'
    }, inplace=True, errors='ignore')

    # Rename severely and moderately deprived indicators
    df.rename(columns={
        'severelydeprived': 'deprived_sev',
        'moderatelydeprived': 'deprived_mod'
    }, inplace=True, errors='ignore')

    # Check if 'deprived_sev' exists in the DataFrame and create or update if not
    if 'deprived_sev' not in df.columns:
        # Create the 'deprived_sev' based on 'sumpoor_sev' if it exists
        if 'sumpoor_sev' in df.columns:
            df['deprived_sev'] = (df['sumpoor_sev'] >= 1).astype(int)
        else:
            # If 'sumpoor_sev' also doesn't exist, you might need to calculate it based on a pattern
            # Assuming dep_*_sev pattern for deprivation columns
            sev_columns = [col for col in df.columns if 'dep_' in col and '_sev' in col]
            df['sumpoor_sev'] = df[sev_columns].sum(axis=1, min_count=1)  # min_count=1 to require at least one non-NA value
            df['deprived_sev'] = (df['sumpoor_sev'] >= 1).astype(int)

    # Similar check and creation for 'deprived_mod'
    if 'deprived_mod' not in df.columns:
        if 'sumpoor_mod' in df.columns:
            df['deprived_mod'] = (df['sumpoor_mod'] >= 1).astype(int)
        else:
            # Calculate 'sumpoor_mod' if not already present, based on dep_*_mod pattern
            mod_columns = [col for col in df.columns if 'dep_' in col and '_mod' in col]
            df['sumpoor_mod'] = df[mod_columns].sum(axis=1, min_count=1)
            df['deprived_mod'] = (df['sumpoor_mod'] >= 1).astype(int)
    # Keep only relevant variables
    columns_to_keep = [
        col for col in df.columns if (
            'countrycode' in col or 'year' in col or 'survey' in col or 'version' in col or
            'round' in col or 'cluster' in col or 'hhid' in col or 'indid' in col or
            'chweight' in col or 'hhweight' in col or 'location' in col or 'sex' in col or
            'wealth' in col or 'region' in col or 'age' in col or
            col.startswith('dep_') or col.startswith('education_') or
            col.startswith('health_') or col.startswith('nutrition_') or
            'sumpoor_' in col or 'deprived_' in col
        )
    ]
    df = df[columns_to_keep]


    # Sort and order DataFrame
    df.sort_values(by=['cluster', 'hhid', 'indid'], inplace=True)

    # Optionally compress the DataFrame before saving
    df = df.convert_dtypes()

    # Optionally save DataFrame
    file_name = f"poverty_variables.csv"
    if save_csv:
        df.to_csv(os.path.join(source_path,file_name), index=False)
    return df

def make_string(integer, length = 8):
    return str(integer).zfill(length)

def check_file_integrity(parent_dir, all_files, country_code):
    complete = True
    for f in all_files:
        if not any(f in string for string in os.listdir(parent_dir)):
            print(f'{country_code[f[:2]]}\'s data in year {f[-4:]} is missing.')
            complete = False
    return complete

def process_dhs(parent_dir, config_file):
    if parent_dir[-1]!='/':
        parent_dir+=r'/'
    with open(config_file, 'r') as file:
        config_data = json.load(file)
    with open(f'{par_dir}dhs_country_code.json', 'r') as file:
        dhs_cc = json.load(file)
    root_grid = parent_dir
    save_to_csv_dir = os.path.join(parent_dir, "dhs_variables.csv")
    print('Checking file integrity...')
    # if not check_file_integrity(parent_dir, config_data['all_DHS'], dhs_cc):
    #     raise FileNotFoundError('DHS data incomplete')

    pov_dfs = []
    print('Summarizing poverty...')
    for f in tqdm(os.listdir(root_grid)):
        if 'DHS' in f:
            try:
                pov_cache = os.path.join(root_grid,f,'poverty_variables.csv')
                if os.path.exists(pov_cache):
                    pov_df = pd.read_csv(pov_cache)
                else:
                    pov_df = get_poverty(root_grid+f+'/', save_csv=True)
                
                pov_dfs.append((f[3:7],pov_df))
            except Exception as e:
                pass
                
    dhs_dfs = []
    print('Extracting DHS variables...')
    for f in tqdm(os.listdir(root_grid)):
        if 'DHS' in f:
            dhs_dfs.append((f[3:7], pd.read_csv(os.path.join(root_grid,f,'dhs_variables.csv'))))

    thresholds = config_data['dhs_variable_lim']
    columns_to_encode = config_data['categorical_columns']
    matches = config_data['matches']

    dhs_dfs_agg = []
    print('Aggregating DHS variables...')
    for year,df in tqdm(dhs_dfs):
        ccode = df['v000'][0]
        for column, threshold in thresholds.items():

            if column in df.columns:
                df = df[(df[column] <= threshold) | (df[column].isna())]
        filtered_columns_to_encode = [col for col in columns_to_encode if col in df.columns]
        for col in df.columns:
            if col in filtered_columns_to_encode:
                df[col] = df[col].astype('Int64')
        df = pd.get_dummies(df, columns=filtered_columns_to_encode)
        df_agg = df.select_dtypes(include=[np.number, bool]).groupby('v004').agg('mean').reset_index()
        df_agg['survey_year'] = [year] * df_agg.shape[0]
        df_agg['id'] = ccode[:2]+df_agg['survey_year']+ df_agg['v004'].apply(make_string)
        dhs_dfs_agg.append(df_agg)

    dhs_df_all = pd.concat(dhs_dfs_agg)

    existing_cols = [col for col in matches if col in dhs_df_all.columns]

    additional_cols = []
    for col in matches:
        if col not in dhs_df_all.columns:
            pattern_cols = [c for c in dhs_df_all.columns if c.startswith(f"{col}_")]
            additional_cols.extend(pattern_cols)
    cols_to_select = existing_cols + additional_cols + ['id']
    dhs_df_all = dhs_df_all[list(set(cols_to_select))]

    print('Aggregating poverty data...')
    poverty_dfs_agg = []
    for df in tqdm(pov_dfs):
        ccode = df[1].countrycode2[0]
        df_agg = df[1].select_dtypes(include=[np.number]).groupby('cluster').agg('mean').reset_index()
        df_agg['id'] = ccode[:2]+df[0]+ df_agg['cluster'].apply(make_string)
        poverty_dfs_agg.append(df_agg)

    pov_df_all = pd.concat(poverty_dfs_agg)

    ## Calculating Centroids
    gdfs = []
    for f in os.listdir(root_grid):
        if 'DHS' in f:
            for sub_f in os.listdir(os.path.join(root_grid,f)):
                if sub_f.__contains__('GE'):
                    shape_file = os.path.join(root_grid, f, sub_f)
                    gdf = gpd.read_file(shape_file)
                    # Append to the list of GeoDataFrames
                    gdfs.append(gdf)
    combined_gdf = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    def country_code_to_name(country_code):
        return dhs_cc[country_code]
    combined_gdf['COUNTRY'] = combined_gdf['DHSCC'].apply(country_code_to_name)
    combined_gdf['SURVEY_NAME'] = [combined_gdf.iloc[i]['COUNTRY']+'_DHS_'+str(int(combined_gdf.iloc[i]['DHSYEAR'])) for i in range(combined_gdf.shape[0])]
    combined_gdf['YEAR'] =combined_gdf['DHSYEAR'].apply(int)
    combined_gdf['CENTROID_ID']  = combined_gdf['DHSID']

    centroid_df = combined_gdf[['CENTROID_ID', 'SURVEY_NAME', 'COUNTRY','YEAR', 'LATNUM', 'LONGNUM']]
    centroid_df = centroid_df[~((centroid_df['LATNUM'] == 0) & (centroid_df['LONGNUM'] == 0))]
    centroid_df.drop_duplicates(inplace=True)
    pov_df_all.drop_duplicates(inplace=True)
    dhs_df_all.drop_duplicates(inplace=True)
    centroid_df = centroid_df.reset_index()

    merged_centroid_df = pd.merge(centroid_df, pov_df_all, left_on='CENTROID_ID', right_on='id', how='left')
    merged_centroid_df = pd.merge(merged_centroid_df, dhs_df_all, left_on='CENTROID_ID', right_on='id', how='left')

    # Ignore the first 7 columns
    df_subset = merged_centroid_df.iloc[:, 8:]
    # Remove columns if all values are NaN
    df_subset = df_subset.dropna(axis=1, how='all')

    # Dictionary to store min-max values
    min_max_dict = {}

    # Function to scale columns and record min-max values
    def scale_column(col):
        if col.dtype in [np.float64, np.float32, np.int64, np.int32]:
            if col.min() < 0 or col.max() > 1:
                min_val = col.min()
                max_val = col.max()
                scaler = MinMaxScaler()
                scaled_col = scaler.fit_transform(col.values.reshape(-1, 1)).flatten()
                min_max_dict[col.name] = {'min': min_val, 'max': max_val}
                return scaled_col
        return col

    # Apply scaling to appropriate columns
    df_scaled = df_subset.apply(scale_column)

    # Combine the original first 7 columns with the processed subset
    df_processed = pd.concat([merged_centroid_df.iloc[:, :8], df_scaled], axis=1)

    # Save min-max dictionary locally
    with open('min_max_values.json', 'w') as f:
        json.dump(min_max_dict, f, indent=4)
    import re
    col_pattern = r"^[a-zA-Z]*\d*_[^a-zA-Z]"
    matching_columns = [col for col in df_processed.columns if re.match(f"^{col_pattern}", col)]

    # Fill NaN values with 0 in the matched columns
    df_processed[matching_columns] = df_processed[matching_columns].fillna(0)
    
    df_processed.to_csv(save_to_csv_dir)

    save_split(df_processed)
    

def save_split(df):
    save_par_dir = r'survey_processing/processed_data/'
    df.to_csv(f'{save_par_dir}dhs_variables.csv', index=False)
    df = df.sample(frac=1, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold = 1
    for train_index, test_index in kf.split(df):
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
