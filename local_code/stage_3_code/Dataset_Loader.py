'''
Concrete IO class for image datasets (MNIST, ORL, CIFAR-10)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.dataset import dataset
import pickle
import numpy as np


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None
    # 'MNIST', 'ORL', or 'CIFAR'
    dataset_type = None

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load(self):
        print('loading data...')
        file_path = self.dataset_source_folder_path + self.dataset_source_file_name
        with open(file_path, 'rb') as f:
            raw = pickle.load(f)

        train_X, train_y = self._process_split(raw['train'])
        test_X, test_y = self._process_split(raw['test'])

        return {
            'train': {'X': train_X, 'y': train_y},
            'test':  {'X': test_X,  'y': test_y}
        }

    def _process_split(self, instances):
        X, y = [], []
        for item in instances:
            img = np.array(item['image'], dtype=np.float32)
            label = item['label']

            if self.dataset_type == 'MNIST':
                # img is shape (28, 28) — add channel dim → (1, 28, 28)
                img = img / 255.0
                img = img[np.newaxis, :, :]

            elif self.dataset_type == 'ORL':
                # img is shape (112, 92, 3) but all channels equal (grayscale)
                # take R channel only → (112, 92), then add channel → (1, 112, 92)
                if img.ndim == 3:
                    img = img[:, :, 0]
                img = img / 255.0
                img = img[np.newaxis, :, :]

            elif self.dataset_type == 'CIFAR':
                # img is shape (32, 32, 3) colored — transpose HWC → CHW → (3, 32, 32)
                img = img / 255.0
                img = img.transpose(2, 0, 1)

            X.append(img)
            y.append(label)

        return X, y
