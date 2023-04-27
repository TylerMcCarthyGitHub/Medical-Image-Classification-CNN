#data loader for  dataset

import os
import csv
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class DataLoader(keras.utils.Sequence):
    def __init__(self, data_dir, target_size=(32,32), batch_size=32, uids=None):
        self.data_dir = data_dir
        self.target_size = target_size
        self.batch_size = batch_size
        self.uids = uids
        self.image_paths = self.get_image_paths()
        self.num_images = len(self.image)
        self.indexes = np.arrange(self.num_images)

        if self.uids is None and self.uid_path is None:
            raise ValueError("Either uids or uid_path must be provided")
        
        if self.uids is None:
            self.uids = self._load_uids_from_csv()

        if len(self.uids)!= self.num_images:
            raise ValueError("Number of uids doesn't match number of images")

    def __len__(self):
        return  int(np.ceil(self.num_images/self.batch_size))
    
    #def __getitem__(self, idx):