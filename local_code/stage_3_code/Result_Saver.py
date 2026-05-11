'''
Concrete ResultModule class for saving experiment output (stage 3)
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.result import result
import pickle
import os


class Result_Saver(result):
    data = None
    fold_count = None
    result_destination_folder_path = None
    result_destination_file_name = None

    def save(self):
        print('saving results...')
        os.makedirs(self.result_destination_folder_path, exist_ok=True)
        path = os.path.join(
            self.result_destination_folder_path,
            self.result_destination_file_name + '_' + str(self.fold_count)
        )
        with open(path, 'wb') as f:
            pickle.dump(self.data, f)
