import os
import csv
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class DataLoader(keras.utils.Sequence):
    def __init__(self, data_dir, csv_path, target_size=(256, 256), batch_size=32):
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.target_size = target_size
        self.batch_size = batch_size
        self.image_paths, self.labels = self.load_data_from_csv()

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_paths = self.image_paths[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_labels = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]

        batch_images = []
        for image_path in batch_paths:
            image = load_img(image_path, target_size=self.target_size)
            image = img_to_array(image)
            batch_images.append(image)

        batch_images = np.array(batch_images)
        batch_labels = np.array(batch_labels)

        return batch_images, batch_labels

    def load_data_from_csv(self):
        image_paths = []
        labels = []

        with open(self.csv_path, "r") as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader)  # Skip header row

            for row in csv_reader:
                image_name = row[0]
                label = row[1]

                image_path = os.path.join(self.data_dir, image_name)
                image_paths.append(image_path)
                labels.append(label)

        return image_paths, labels