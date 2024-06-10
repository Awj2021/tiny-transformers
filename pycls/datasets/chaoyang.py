import os
import json
from PIL import Image

from .base import BaseDataset
from pycls.core.io import pathmgr


class Chaoyang(BaseDataset):

    def __init__(self, data_path, split):
        """
        data_path: /home/wenjie/dataset/chaoyang/
        split: 'train' or 'test'
        """
        super(Chaoyang, self).__init__(split)
        self.split = split
        assert pathmgr.exists(data_path), "Data path '{}' not found".format(data_path)
        # TODO: check if the 'train.json' and 'test.json' are the correct file names.
        # splits = {"train": "train_classification.json", "test": "test_split_2.json"} # version0, should be test_split_2.json
        # splits = {"train": "train_split_2.json", "test": "test_split_2.json"}                          # version_original 
        # splits = {"train": "train_version1.json", "test": "test_split_2.json"}     
                              # version1
        
        ############## Some matching between train_folder and splits ##############
        # version_split_2_words: data_mix.json, test_split_2.json
        # version_cy_ori_split_text: data_mix.json, test_ori.json 
        # version_cy_ori_split_words: data_mix.json, test_ori.json
        # version_cy_ori_split_wo_da: train_ori.json, test_ori.json
        # cy_split_2_text_diff_ratio_ax2_dax4: train_classification.json, test_split_2.json

        # ----------------------------- to be modified -----------------------------
        self.train_folder = 'version_cy_ori_split_text_epoch_30'  # json | version_cy_ori_split_text | version_cy_ori_split_text_epoch_40

        # ----------------------------- to be modified -----------------------------
        splits = {"train": "data_mix.json", "test": "test.json"}        # version_cy_ori_split_text

        assert self.split in splits, f"Split '{self.split}' not supported for Chaoyang"
        self.data_path = data_path

        if split == 'test':
            with open(os.path.join(data_path, 'json', splits[self.split]), 'r') as f:
                anns = json.load(f)
        elif split == 'train':
            with open(os.path.join(data_path, self.train_folder, splits[self.split]), 'r') as f:
                anns = json.load(f)
        else:
            raise ValueError(f"Split '{self.split}' not supported for Chaoyang")

        self.data = anns
        

    def __len__(self):
        return len(self.data)

    def _get_data(self, index):
        ann = self.data[index]
        
        if self.split == 'train':
            # if train and test with original chaoyang dataset, please remove the train_folder in the below line.
            # img = Image.open(os.path.join(self.data_path, self.train_folder, ann['name']))
            if os.path.exists(os.path.join(self.data_path, self.train_folder, ann['name'])):
                img = Image.open(os.path.join(self.data_path, self.train_folder, ann['name']))
            elif os.path.exists(os.path.join(self.data_path, ann['name'])):
                img = Image.open(os.path.join(self.data_path, ann['name']))
            else:
                raise ValueError(f"Image '{ann['name']}' not found")
            # img = Image.open(os.path.join(self.data_path, ann['name']))
            return img, ann['label']
        else:   # test. keep returning ann['label'].
            img = Image.open(os.path.join(self.data_path, ann['name']))
            return img, ann['label']
