'''
Concrete SettingModule for stage 3: data already pre-split into train/test
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from local_code.base_class.setting import setting


class Setting_Train_Test_Split(setting):

    def load_run_save_evaluate(self):
        # Dataset.load() returns {'train': {'X':..,'y':..}, 'test': {'X':..,'y':..}}
        # The pickle files are already partitioned — no sklearn split needed
        loaded_data = self.dataset.load()

        self.method.data = loaded_data
        learned_result   = self.method.run()

        self.result.data = learned_result
        self.result.save()

        self.evaluate.data = learned_result
        return self.evaluate.evaluate(), None
