"""
Author: Fan Yang
Date: 2024-02-17

This script is designed to download satellite imagery based on parameters provided by the user.
It requires the path to a CSV file containing the names, latitudes, and longitudes of the locations
for which the imagery is to be downloaded, the target Google Drive folder for storing the downloaded images,
the year of the imagery, the satellite sensor (Landsat 5, 7, or 8), and an optional range in kilometers that
defines the area around the specified coordinates to be covered by the imagery.

Usage:
    python main.py <filepath> <drive> <year> <sensor> [-r <range_km>] [rgb_only]

Arguments:
    filepath: Path to a CSV file containing columns for name, latitude (lat), and longitude (lon) of the locations.
    drive: The ID or name of the Google Drive folder where the downloaded images will be stored.
    year: The year for which the satellite images are to be collected.
    sensor: Specifies the satellite sensor to use for the images. Acceptable values are 'L5', 'L7', and 'L8', corresponding to Landsat 5, 7, and 8, respectively.
    range_km (optional): The range around the specified coordinates, in kilometers, to be covered by the satellite imagery. Defaults to 10 km if not provided.

The script uses the 'download_imagery' function from the 'download_imagery' module to perform the download.
Ensure that 'download_imagery.py' is in the same directory as this script and contains the required 'download_imagery' function.
"""

import argparse
from download_imagery import download_imagery  # Assuming download.py is in the same directory and has a function named 'download'

def main():
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Download satellite images based on provided parameters.")
    parser.add_argument("filepath", help="CSV file for scraping, containing name, lat, and lon")
    parser.add_argument("drive", help="Google Drive folder")
    parser.add_argument("year", help="Year for which satellite images are collected")
    parser.add_argument("sensor", help="Sensor. Can take values L5, L7, and L8 for Landsat 5, 7, and 8 respectively")
    parser.add_argument("-r", "--range_km", type=int, help="The range of the imagery in kilometers.", default=10)  # Assuming a default value if not provided
    parser.add_argument("--rgb_only", action='store_true', help="Whether to only obtain RGB bands")

    # Parse arguments
    args = parser.parse_args()

    # Call the download function with the parsed arguments
    download_imagery(args.filepath, args.drive, args.year, args.sensor, args.range_km, args.rgb_only)

if __name__ == "__main__":
    main()
