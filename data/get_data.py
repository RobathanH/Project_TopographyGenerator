import requests
import rasterio
from rasterio.plot import show

import numpy as np
import matplotlib.pyplot as plt
import json
import math
import os
import tqdm
from functools import lru_cache

# Constants / Defaults
OPEN_TOPOGRAPHY_URL = "https://portal.opentopography.org/API/globaldem"
API_KEY = "9563347f2787bf85dda0ce8f8b6b8c9e"
OUTPUT_FORMAT = "GTiff"

# Dataset Format
DATASET = "SRTM15Plus"

# Length of windows saved individually
LNG_LEN = 10
LAT_LEN = 10

# Bounds of data saved
LNG_MIN = -180
LNG_MAX = 180
LAT_MIN = -80
LAT_MAX = 80

# Multiply values by this factor before storing
MAGNITUDE_MULTIPLIER = 1e-3


'''
Requests a particular region of height data from the OpenTopography API
Returns a rasterio file object, which should be closed after use
'''
def get_bounded_data(south, north, west, east, verbose=False):
    params = {
        'demtype': DATASET,
        'south': south,
        'north': north,
        'west': west,
        'east': east,
        'outputFormat': OUTPUT_FORMAT,
        'API_Key': API_KEY
    }

    if verbose:
        print(f"Requesting...\n{json.dumps(params, indent=2)}")
    resp = requests.get(OPEN_TOPOGRAPHY_URL, params=params)
    if verbose:
        print(f"Request Complete: {resp.status_code}")

    if resp.status_code != 200:
        raise ValueError(resp.content.decode('utf-8'))
    
    with open("data.tmp", "wb") as fp:
        fp.write(resp.content)
    data = rasterio.open("data.tmp")
    
    return data

'''
Helper functions to determine save location for data of given format and coords
'''
def data_savefolder(create_folder=False):
    script_folder = os.path.dirname(os.path.realpath(__file__))
    save_folder = f"{DATASET}.{LNG_LEN}x{LAT_LEN}.{LNG_MIN}_{LNG_MAX}.{LAT_MIN}_{LAT_MAX}"
    save_folder_path = os.path.join(script_folder, save_folder)
    if create_folder:
        os.makedirs(save_folder_path, exist_ok=True)
    return save_folder_path

def data_savename(lng, lat, create_folder=False):
    save_folder = data_savefolder(create_folder=create_folder)
    save_file = f"heights.lng_{lng}.lat_{lat}.npy"
    return os.path.join(save_folder, save_file)


'''
Requests 10 lat, 10 lng patches over the entire default dataset and saves them for easy retrieval.
Uses the default data config constants at the top of this file.
'''
def save_dataset():
    total_windows = math.ceil((LNG_MAX - LNG_MIN) / LNG_LEN) * math.ceil((LAT_MAX - LAT_MIN) / LAT_LEN)
    lat = LAT_MIN
    lng = LNG_MIN
    for i in tqdm.trange(total_windows):
        # Sanity check
        if lng >= LNG_MAX:
            print(f"Iterating through too many windows, reached lat = {lat}, lng = {lng}")
            return

        # Skip if file already exists
        if not os.path.exists(data_savename(lng, lat)):
            # Download image
            data = get_bounded_data(south = lat, north = min(lat + LAT_LEN, LAT_MAX), west = lng, east = min(lng + LNG_LEN, LNG_MAX))

            # Add to img array
            # Default pixel order:
            #   west -> east = axis 1 low to high
            #   south -> north = axis 0 high to low
            img = data.read(1)

            # Convert pixel order:
            #   west -> east = axis 0 low to high
            #   south -> north = axis 1 low to high
            img = np.tranpose(np.flip(img, axis=0))
            
            # Save img to coord-labeled file
            np.save(data_savename(lng, lat, create_folder=True), img)
        
        # Increment window location
        lat += LAT_LEN
        if lat >= LAT_MAX:
            lat = LAT_MIN
            lng += LNG_LEN

'''
Cache to load particular lat/lng window files
Returns np.ndarray if the requested window exists,
and None if it doesn't.
'''
@lru_cache(maxsize=50)
def load_window(lng, lat):
    save_path = data_savename(lng, lat)
    if os.path.exists(save_path):
        return np.load(save_path)
    else:
        return None

'''
Loads an arbitrary region from the saved dataset, putting together
data from multiple adjacent windows to cover the desired region, 
and optionally downsampling to a particular resolution.
'''
def load_region(region_lng, region_lat, region_lng_len, region_lat_len, output_shape=None):
    # Check coords are valid
    if region_lng_len < 0 or region_lng_len > LNG_MAX - LNG_MIN:
        raise ValueError(f"Region longitude length ({region_lng_len}) is invalid.")
    if region_lng_len < 0 or region_lat_len > LAT_MAX - LAT_MIN:
        raise ValueError(f"Region latitude length ({region_lat_len}) is invalid.")
    if region_lat < LAT_MIN or region_lat + region_lat_len > LAT_MAX:
        raise ValueError(f"Region latitude measures (start = {region_lat}, len = {region_lat_len}) are invalid.")

    # Convert measurements into window-relative coords
    window_x_len = region_lng_len / LNG_LEN
    window_y_len = region_lat_len / LAT_LEN
    window_x_start = (region_lng - LNG_MIN) / LNG_LEN
    window_y_start = (region_lat - LAT_MIN) / LAT_LEN

    # Collect parts of saved windows covered by region
    total_x_windows = math.ceil((LNG_MAX - LNG_MIN) / LNG_LEN)
    total_y_windows = math.ceil((LAT_MAX - LAT_MIN) / LAT_LEN)
    window_x_ind_start = math.floor(window_x_start)
    window_x_ind_end = math.ceil(window_x_start + window_x_len)
    window_y_ind_start = math.floor(window_y_start)
    window_y_ind_end = math.ceil(window_y_start + window_y_len)

    sub_arrays = []
    for window_x_ind in range(window_x_ind_start, window_x_ind_end):
        sub_arrays.append([])
        for window_y_ind in range(window_y_ind_start, window_y_ind_end):
            window_lng = (window_x_ind % total_x_windows) * LNG_LEN + LNG_MIN
            window_lat = window_y_ind * LAT_LEN + LAT_MIN
            sub_arrays[-1].append(load_window(window_lng, window_lat))

    # Trim sub_arrays at edges of region
    window_shape = sub_arrays[0][0].shape
    x_start_trim_ind = math.floor((window_x_start % 1) * window_shape[0])
    x_end_trim_ind = math.ceil(((window_x_start + window_x_len) % 1) * window_shape[0])
    for i in range(window_y_ind_end - window_y_ind_start):
        sub_arrays[0][i] = sub_arrays[0][i][x_start_trim_ind:, :]
        if x_end_trim_ind != 0: # If this is 0, we want full end window
            sub_arrays[-1][i] = sub_arrays[-1][i][:x_end_trim_ind, :]

    y_start_trim_ind = math.floor((window_y_start % 1) * window_shape[1])
    y_end_trim_ind = math.ceil(((window_y_start + window_y_len) % 1) * window_shape[1])
    for i in range(window_x_ind_end - window_x_ind_start):
        sub_arrays[i][0] = sub_arrays[i][0][:, y_start_trim_ind:]
        if y_end_trim_ind != 0: # If this is 0, we want full end window
            sub_arrays[i][-1] = sub_arrays[i][-1][:, :y_end_trim_ind]

    # Stitch together trimmed windows
    vertical_strips = [np.concatenate(sub_arrays[i], axis=1) for i in range(len(sub_arrays))]
    region = np.concatenate(vertical_strips, axis=0)

    # Optional Downsampling to desired resolution
    if output_shape is not None:
        if region.shape[0] > output_shape[0]:
            x_inds = np.round(np.linspace(0, region.shape[0], num=output_shape[0], endpoint=False)).astype(int)
            region = region[x_inds, :]
        if region.shape[1] > output_shape[1]:
            y_inds = np.round(np.linspace(0, region.shape[1], num=output_shape[1], endpoint=False)).astype(int)
            region = region[:, y_inds]

    return region


if __name__ == '__main__':
    save_dataset()