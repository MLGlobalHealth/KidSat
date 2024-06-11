# KidSat: satellite imagery to map childhood poverty dataset and benchmark

## Introduction

This is a repository for code of NeurIPS benchmark and dataset submission 2024.

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

3. **Query File**

    The file `imagery_scraping/config/query.json` contains an example of how you should query imageries. You need to provide the latitude and longitude in WGS84 format.

3. **Running the Application**

    Example:

    You first need to go to the `imagery_scraping` directory

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

4. **Visualization**

    To see the imagery, you need to download the imagery data from Google Drive first. We provide sample data in `imagery_scraping/data` and a [notebook](imagery_scraping/visualization.ipynb) to see the imagery you queried in true color. Note that this is only a visualization; the original data is much richer and contains more than the three RGB channels. For training, we should use the original data instead of the true-color image alone.

## Summarizing the dataset

Collect all DHS data to `survey_processing/dhs_data`. The following command

```bash
python survey_processing/main.py survey_processing/dhs_data
```

would create 5 splits of the training and test data for spatial analysis and before/after 2020 split for temporal analysis.

## Experiment with DINOv2

After having the splits in `survey_processing/processed_data`, you can finetune DINOv2 using the following commands. For the spatial experiment with Landsat imagery, you can use the following code.


```bash
python modelling/dino/finetune_spatial.py --fold 1 --model_name dinov2_vitb14 --imagery_path {path_to_parent_imagery_folder} --batch_size 8 --imagery_source L
```

Finetuning sentinel imagery, the normal command is 

```bash
python modelling/dino/finetune_spatial.py --fold 1 --model_name dinov2_vitb14 --imagery_path {path_to_parent_imagery_folder} --batch_size 1 --imagery_source S
```

Note that to get a cross-validated result, you should use fold 1 to 5.

For temporal finetuning, the command for Landsat is 

```bash
python modelling/dino/finetune_temporal.py --model_name dinov2_vitb14 --imagery_path {path_to_parent_imagery_folder} --batch_size 8 --imagery_source L
```

and replace `L` to `S` for sentinel finetuning.

For evalution, make sure the finetuned models are in `modelling/dino/model` and run 

```bash
python modelling/dino/evaluate.py --use_checkpoint --imagery_source L --mode spatial
```

Change the `--mode` to `temporal` for temporal evaluation, and change `L` to `S` for imagery sources.
Remove the `--use_checkpoint` for evaluating on raw DINO models.

## Experiment with SatMAE (To be updated)
- `cd` to `modelling/finetuning`, and download fMoW-non-temporal SatMAE checkpoint from [here](https://zenodo.org/record/7369797/files/fmow_pretrain.pth). (There is a checkpoint at `/data/coml-satellites/orie4868/SatMAE/fmow-pretrain.pth`)
- Run the following:
```sh
python -m finetune --model_name satmae --pretrained_model $CHECKPOINT_PATH --training_df_path ./DHS_2019_Image_Path.csv --output_dir $OUTPUT_DIR --img_size=224
```
- (There will be `IncompatibleKeys` warning, but this is expected since we are discarding the decoder part of SatMAE)

Currently the temporal SatMAE is not supported yet. This is because the data doesn't really make sense to use in SatMAE's temporal context. SatMAE uses 3 different timestamps of the same location to improve performance.
