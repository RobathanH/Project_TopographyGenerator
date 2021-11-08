import numpy as np
import tqdm
import os
from collections import deque

from .get_data import *
from .util import *


class DataLoader:
    def __init__(self, img_shape, region_dims):
        self.shape = img_shape
        self.region_lng_len, self.region_lat_len = region_dims

        # Constants for data filtering and augmentation
        self.FILTER_ENABLED = False
        self.FILTER_MIN_VAR = 0.01**2 # filter out images below this variance threshold
        self.FILTER_MIN_LANDMASS = 0.05 # filter out images with relative landmass area below this threshold

        self.TRANSLATION_RELATIVE_DISTANCE = 0.1 # shift input images by this fraction of width/height between samples

        self.AUGMENT_LNG_FLIP = True
        self.AUGMENT_LAT_FLIP = True

    def cache_path(self):
        identifiers = ["dataset"]
        identifiers += [f"output_{self.shape[0]}x{self.shape[1]}"]
        identifiers += [f"input_{self.region_lng_len}x{self.region_lat_len}"]
        identifiers += [f"translation_{self.TRANSLATION_RELATIVE_DISTANCE}"]
        if self.FILTER_ENABLED:
            identifiers += [f"filter_var_{self.FILTER_MIN_VAR}"]
            identifiers += [f"filter_landmass_{self.FILTER_MIN_LANDMASS}"]
        if self.AUGMENT_LNG_FLIP:
            identifiers += [f"augment_lngflip"]
        if self.AUGMENT_LAT_FLIP:
            identifiers += [f"augment_latflip"]

        return os.path.join(data_savefolder(), ".".join(identifiers) + ".npy")

    def get_data(self):
        # Load cache if it exists
        if os.path.exists(self.cache_path()):
            print("Loading cached formatted dataset")
            self.data = np.load(self.cache_path())
        else:
            print("Creating formatted dataset")
            self.load_full_data()

    '''
    Plots and shows some randomly chosen samples from the dataset
    '''
    def show_example_data(self):
        inds = np.random.choice(range(self.data.shape[0]), size=10)
        save_images(self.data[inds], title="Example Data")

    '''
    Stitch and load a full dataset with the given config, and save cache version
    '''
    def load_full_data(self):
        self.data = None
        data_sequence = deque()
        lng = LNG_MIN
        lat = LAT_MIN

        lat_iterations = (LAT_MAX - LAT_MIN - self.region_lat_len) / (self.region_lat_len * TRANSLATION_RELATIVE_DISTANCE) + 1
        lng_iterations = (LNG_MAX - LNG_MIN) / (self.region_lng_len * self.TRANSLATION_RELATIVE_DISTANCE)
        total_iterations = lng_iterations * lat_iterations
        pbar = tqdm.tqdm(total=total_iterations, desc="Loading regions into compiled dataset")

        # Increment helper to pass through all possible regions
        def increment():
            nonlocal lng, lat
            pbar.update(1)
            lat += self.region_lat_len * self.TRANSLATION_RELATIVE_DISTANCE
            if (lat + self.region_lat_len) > LAT_MAX:
                lat = LAT_MIN
                lng += self.region_lng_len * self.TRANSLATION_RELATIVE_DISTANCE

        while lng < LNG_MAX:
            # Get region
            region = load_region(lng, lat, self.region_lng_len, self.region_lat_len, output_shape=self.shape)

            # Filters
            if self.FILTER_ENABLED:
                var = np.var(region)
                landmass = np.mean(region > 0)

                if var < self.FILTER_MIN_VAR or landmass < self.FILTER_MIN_LANDMASS:
                    increment()
                    continue

            # Add to data
            data_sequence.append(region)

            # Add augmented versions
            if self.AUGMENT_LNG_FLIP:
                augmented = np.flip(region, axis=0)
                data_sequence.append(augmented)
            if self.AUGMENT_LAT_FLIP:
                augmented = np.flip(region, axis=1)
                data_sequence.append(augmented)
            if self.AUGMENT_LNG_FLIP and self.AUGMENT_LAT_FLIP:
                augmented = np.flip(np.flip(region, axis=0), axis=1)
                data_sequence.append(augmented)

            # Increment
            increment()

        # Save final dataset
        print("Saving formatted dataset")
        self.data = np.array(data_sequence)
        np.save(self.cache_path(), self.data)

if __name__ == "__main__":
    default_loader = DataLoader((128, 128), 30, 30)