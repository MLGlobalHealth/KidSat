# KidSat: satellite imagery to map childhood poverty dataset and benchmark

## Introduction

This is a repository for the work **KidSat: satellite imagery to map childhood poverty**.

![Figure 1](https://i.imgur.com/xLbiFwq.png)


## Getting All DHS Data

The Demographic and Health Surveys (DHS) program gathers and shares vital data on population, health, and nutrition in developing countries to inform public health policies. Their collection procedures and methods are listed [here](https://dhsprogram.com/data/data-collection.cfm).

To access DHS data, please follow these steps:

1. **Register for DHS Access:**
   - Visit the registration page [here](https://dhsprogram.com/data/new-user-registration.cfm) and apply for access to the DHS data.


2. **Obtain the Data for Following Countries and Years**
    For the following country and years, select ALL STATA and Geographic Data.
    | Country      | Year(s) |
    |--------------|---------|
    | Zambia       | 2007, 2013, 2018|
    | Malawi       | 2000, 2004, 2010, 2015|
    | Uganda       | 2000, 2006, 2011, 2016|
    | Comoros      | 2012|
    | Tanzania     | 1999, 2010, 2015, 2022|
    | Kenya        | 2003, 2008, 2014, 2022|
    | Angola       | 2015    |
    | Ethiopia     | 2000, 2005, 2011, 2016, 2019|
    | Rwanda       | 2005, 2007, 2010, 2014, 2019|
    | Lesotho      | 2004, 2009, 2014    |
    | Madagascar   | 1997, 2008, 2021|
    | Zimbabwe     | 1999, 2005, 2010, 2015|
    | Burundi      | 2010, 2016    |
    | Mozambique   | 2011    |
    | Eswatini     | 2006    |
    | South Africa | 2016    |

    The folders should be unzipped and store in `survey_processing/dhs_data/` (e.g. `survey_processing/dhs_data/` should contain subfolders of "ET_20XX_DHS_XXX..." etc. ).
---

## Usage for Imagery Scraping

This section provides step-by-step instructions on how to use this repository to achieve its intended functionality.

### Prerequisites

Before you start, make sure you have registered a Google Earth Engine project for academic purposes. You will need your project name to query the API. The sign-up page is [here](https://signup.earthengine.google.com).


1. **Set Up Environment**

    Example:

    ```bash
    pip install -r requirements.txt
    ```

2. **Configuration**

    You need to update your Google Earth Engine project name to `imagery_scraping/config/google_config.json`. The format (for me) was `ee-YOUR_GMAIL_NAME`. Note, please do not push your project name to GitHub.

3. **Query File (Optional)**

    The file `imagery_scraping/config/query.json` contains an example of how you should query imageries. You need to provide the latitude and longitude in WGS84 format. In our work, we mainly use shapefile from DHS directly.

4. **Running the Application**
    You first need to go to the `imagery_scraping` directory

    Example:


    An example of usage is shown below:

    ```bash
    python main.py "config/query.csv" "EarthImagery" 2021 "L8" -r 5
    ```

    It will prompt you to authenticate for Google. If all goes well, it will download the images to your Google Drive under a folder called `EarthImagery`. The images will be collected from the 2021 LandSat8 dataset and will be centered around the coordinates you provided in the query file with a 5 km square window.

    If you have a shapefile from DHS, you can also use for example

    ```bash
    python main.py "ETGE81FL" "Ethiopia2021Imagery" 2021 "S2" -r 5
    ```
    
    to extract the imagery.

5. **Visualization (Optional)**

    To see the imagery, you need to download the imagery data from Google Drive first. We provide sample data in `imagery_scraping/data` and a [notebook](imagery_scraping/visualization.ipynb) to see the imagery you queried in true color. Note that this is only a visualization; the original data is much richer and contains more than the three RGB channels. For training, we should use the original data instead of the true-color image alone.

6. **Getting All Imagery**

    We recommend using this [notebook](imagery_scraping/get_imagery.ipynb) to download all imagery and keep track of progress as GEE has a upper limit of 3000 jobs at the same time. You will need to download the imagery and save to an accessible location (we will refer to `path_to_parent_imagery_folder` in later sections), each of its subdirectory should be country code + year + source (e.g. ET2019S2 for Ethiopia 2019 Sentinel 2). The notebook should already be formatting the export using this naming convention.


## Summarizing the Dataset

Collect all DHS data to `survey_processing/dhs_data`. The following command

```bash
python survey_processing/main.py survey_processing/dhs_data
```

would create 5 splits of the training and test data for spatial analysis and before/after 2020 split for temporal analysis.

## Experiment with MOSAIKS

The MOSAIKS features were extracted using [IDinsight](https://github.com/IDinsight/mosaiks#mosaiks-satellite-imagery-featurization) package. A [notebook](modelling/mosaiks/main.ipynb) is provided in this repository for getting all features for MOSAIKS.

## Experiment with DINOv2

After having the splits in `survey_processing/processed_data`, you can finetune DINOv2 using the following commands. For the spatial experiment with Landsat imagery, you can use the following code.


```bash
python modelling/dino/finetune_spatial.py --fold 1 --model_name dinov2_vitb14 --imagery_path {path_to_parent_imagery_folder} --batch_size 8 --imagery_source L --num_epochs 20
```

Finetuning sentinel imagery, the normal command is 

```bash
python modelling/dino/finetune_spatial.py --fold 1 --model_name dinov2_vitb14 --imagery_path {path_to_parent_imagery_folder} --batch_size 1 --imagery_source S --num_epochs 10
```

Note that to get a cross-validated result, you should use fold 1 to 5.

For temporal finetuning, the command for Landsat is 

```bash
python modelling/dino/finetune_temporal.py --model_name dinov2_vitb14 --imagery_path {path_to_parent_imagery_folder} --batch_size 8 --imagery_source L
```

and replace `L` to `S` for sentinel finetuning.

For evaluation, make sure the all 1-5 finetuned spatial models  (or the finetuned temporal model for temporal evaluation) are in `modelling/dino/model` and run 

```bash
python modelling/dino/evaluate.py --use_checkpoint --imagery_path {path_to_parent_imagery_folder} --imagery_source L --mode spatial
```

Change the `--mode` to `temporal` for temporal evaluation, and change `L` to `S` for imagery sources.
Remove the `--use_checkpoint` for evaluating on raw DINO models.

## Experiment with SatMAE
### Finetuning
To run the finetuning process, you first need to download the checkpoints for fMoW-SatMAE [non-temporal](https://zenodo.org/record/7369797/files/fmow_pretrain.pth) or [temporal](https://zenodo.org/record/7369797/files/pretrain_fmow_temporal.pth). Then run the following:

```sh
python -m modelling.satmae.satmae_finetune --pretrained_ckpt $CHECKPOINT_PATH --dhs_path ./survey_processing/processed_data/train_fold_1.csv --output_dir $OUTPUT_DIR --imagery_path $IMAGERY_PATH
```
Arguments:
- `--pretrained_ckpt`: Checkpoint of pretrained SatMAE model.
- `--imagery_path`: Path to imagery folder
- `--dhs_path`: Path to DHS `.csv` file
- `--output_path`: Path to export the output. A unique subdirectory will be created.
- `--batch_size`
- `--random_seed`
- `--sentinel`: Landsat is used by default. Turn this on to use Sentinel imagery
- `--temporal`: Add this flag to use the temporal mode
- `--epochs`: Number of epochs
- `--stopping_delta`: Delta for early stopping
- `--stopping_patience`: Early stopping patience
- `--loss`: Either `l1` (default) or `l2`.
- `--lr`: Learning rate
- `--weight_decay`: Weight decay for Adam optimizer
- `--enable_profiling`: Enable reporting of loading/inference time.


### Evaluation
Evaluation consists of 2 steps: exporting the model output, and perform Ridge Regression. Since exporting the model output is expensive, we split it into 2 separate modules:

To carry out the first step, edit the file `modelling/satmae/satmae_finetune` and change the `SATMAE_PATHS` variable accordingly. For each entry, you can put all the model checkpoints you need to evaluate or `None` to use the pretrained checkpoint, along with their fold (1-5). You do not have to put the entries in any order, nor need to put all the folds, but the script caches the data from different folds in memory, which helps significantly reduce the time for loading and preprocessing the satellite images.
```sh
python -m modelling.satmae.satmae_finetune --output_dir $OUTPUT_DIR --imagery_path $IMAGERY_PATH
```
Arguments
- `--imagery_path`: Path to imagery folder
- `--output_path`: Path to export the output. A unique subdirectory will be created.
- `--batch_size`
- `--sentinel`: Landsat is used by default. Turn this on to use Sentinel imagery
- `--temporal`: Add this flag to use the temporal mode

This will export data as Numpy arrays in `.npy` files in the output location, which has the shape `(num_samples, 1025)`. The first 1024 columns (i.e `arr[:, :1024]`) is the predicted feature vector from the model, and the last column (i.e `arr[:, 1024]`) is the target. You can then adapt the script `modelling/satmae/eval_dhs.py` to conduct Ridge Regression or more advanced regression.
