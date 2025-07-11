# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    # dataset_dir = 'market1501_to_RGBNT201_dark'
    dataset_dir = 'Market1501'

    def __init__(self, root='', verbose=True, i_modality=None, c_modality=None, **kwargs):
        super(Market1501, self).__init__()
        self.i_modality = i_modality if i_modality is not None else ["RGB"]
        self.c_modality = c_modality if c_modality is not None else []

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self.caption_dir = osp.join(root, 'cap_predictions', 'Market1501 Captions')
        self.train_caption_prefix = 'Market1501-bbox-train'
        self.query_caption_prefix = 'Market1501-query'
        self.test_caption_prefix = 'Market1501-bbox-test'

        self._check_before_run()
        
        train = self._process_dir(self.train_dir, prefix=self.train_caption_prefix, relabel=True)
        query = self._process_dir(self.query_dir, prefix=self.query_caption_prefix, relabel=False)
        gallery = self._process_dir(self.gallery_dir, prefix=self.test_caption_prefix, relabel=False)


        if verbose:
            print("=> market1501_to_RGBNT201_dark loaded")
            self.print_dataset_statistics(train, query, gallery, self.c_modality)

        self.train = train
        self.query = query
        self.gallery = gallery



        self.num_train_pids, self.num_train_items, self.num_train_cams, self.num_train_vids, self.num_train_images, self.num_train_captions = self.get_imagedata_info(
            self.train)
        self.num_train_pids, self.num_train_items, self.num_train_cams, self.num_train_vids, self.num_train_images, self.num_train_captions = self.get_imagedata_info(
            self.query)
        self.num_train_pids, self.num_train_items, self.num_train_cams, self.num_train_vids, self.num_train_images, self.num_train_captions = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _load_caption_dict(self, caption_file):
        """
        Load a caption file with format:
        <filename>\t<caption text>
        Returns: dict of {filename: caption}
        """
        caption_dict = {}
        with open(caption_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if '\t' not in line:
                    continue  # skip malformed lines
                filename, caption = line.split('\t', 1)
                filename = filename.strip()
                caption = caption.strip()
                caption_dict[filename] = caption
        return caption_dict

    def _load_caption_dicts(self, prefix):
        """
        Loads multiple caption dicts (1 per modality) given a prefix.
        e.g., prefix = 'Market1501-bbox-train'
        """
        caption_dicts = []
        for mod in self.c_modality:
            mod_suffix = 'visible' if mod == 'RGB' else mod  # fix naming
            filename = f"{prefix}-{mod_suffix}.txt"
            filepath = osp.join(self.caption_dir, filename)
            caption_dicts.append(self._load_caption_dict(filepath))
        return caption_dicts
    
    # def _process_dir_train(self, dir_path, relabel=False):
    #     img_paths_RGB = glob.glob(osp.join(dir_path, 'visible', '*.jpg'))
    #     pid_container = set()
    #     for img_path_RGB in img_paths_RGB:
    #         jpg_name = img_path_RGB.split('/')[-1]
    #         pid = int(jpg_name.split('_')[0][0:4])
    #         pid_container.add(pid)
        
    #     pid2label = {pid: label for label, pid in enumerate(pid_container)}
    #     data = []
        
    #     for img_path_RGB in img_paths_RGB:
    #         jpg_name = img_path_RGB.split('/')[-1]
    #         img_path_NI = osp.join(dir_path, 'IR', jpg_name)
    #         img_path_TI = osp.join(dir_path, 'TI', jpg_name)

    #         # Check if all three modalities exist
    #         if osp.exists(img_path_RGB) and osp.exists(img_path_NI) and osp.exists(img_path_TI):
    #             img = []
    #             img.append(img_path_RGB)
    #             img.append(img_path_NI)
    #             img.append(img_path_TI)
    #             pid = int(jpg_name.split('_')[0][0:4])
    #             camid = int(jpg_name.split('_')[1][1]) - 1  # Adjust camera ID to start from 0

    #             timeid = int(jpg_name.split('_')[3][1])
    #             if timeid != 2:
    #                 timeid = 0
    #             else:
    #                 timeid = 1
    #             if relabel:
    #                 pid = pid2label[pid]
    #             data.append((img, pid, camid, timeid))
    #         else:
    #             print(f"Skipping {jpg_name}: missing one or more modalities in training set")

    #     return data

    def _process_dir(self, dir_path, prefix, relabel=False):
        img_paths_RGB = glob.glob(osp.join(dir_path, 'visible', '*.jpg'))
        pid_container = set()

        for img_path_RGB in img_paths_RGB:
            jpg_name = osp.basename(img_path_RGB)
            pid = int(jpg_name.split('_')[0][0:4])
            pid_container.add(pid)

        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        # Only load caption dicts if c_modality is provided
        caption_dicts = self._load_caption_dicts(prefix) if self.c_modality else None
        data = []

        for img_path_RGB in img_paths_RGB:
            jpg_name = osp.basename(img_path_RGB)
            pid = int(jpg_name.split('_')[0][0:4])
            camid = int(jpg_name.split('_')[1][1]) - 1
            timeid = 0 if int(jpg_name.split('_')[3][1]) != 2 else 1

            img_paths = []
            valid = True

            # Collect image paths based on i_modality
            for mod in self.i_modality:
                folder = 'visible' if mod == 'RGB' else mod
                img_path = osp.join(dir_path, folder, jpg_name)
                if not osp.exists(img_path):
                    valid = False
                    break
                img_paths.append(img_path)

            # Collect captions based on c_modality (if enabled)
            caption_list = []
            if valid and self.c_modality:
                for cap_dict in caption_dicts:
                    caption_list.append(cap_dict.get(jpg_name, ""))

            if valid:
                if relabel:
                    pid = pid2label[pid]
                if self.c_modality:
                    data.append((img_paths, caption_list, pid, camid, timeid))
                else:
                    data.append((img_paths, pid, camid, timeid))
            else:
                print(f"Skipping {jpg_name}: missing image file for selected modalities")

        return data

    # def _process_dir_test(self, dir_path, relabel=False):
    #     img_paths_RGB = glob.glob(osp.join(dir_path, 'visible', '*.jpg'))
    #     pid_container = set()
    #     for img_path_RGB in img_paths_RGB:
    #         jpg_name = img_path_RGB.split('/')[-1]
    #         pid = int(jpg_name.split('_')[0][0:4])
    #         pid_container.add(pid)
    #     pid2label = {pid: label for label, pid in enumerate(pid_container)}

    #     data = []
    #     for img_path_RGB in img_paths_RGB:
    #         jpg_name = img_path_RGB.split('/')[-1]
    #         img_path_NI = osp.join(dir_path, 'IR', jpg_name)
    #         img_path_TI = osp.join(dir_path, 'TI', jpg_name)

    #         # Check if all three modalities exist
    #         if osp.exists(img_path_RGB) and osp.exists(img_path_NI) and osp.exists(img_path_TI):
    #             img = []
    #             img.append(img_path_RGB)
    #             img.append(img_path_NI)
    #             img.append(img_path_TI)
    #             pid = int(jpg_name.split('_')[0][0:4])
    #             camid = int(jpg_name.split('_')[1][1]) - 1  # Adjust camera ID to start from 0
    #             timeid = int(jpg_name.split('_')[3][1])
    #             if relabel:
    #                 pid = pid2label[pid]
    #             data.append((img, pid, camid, timeid))
    #         else:
    #             print(f"Skipping {jpg_name}: missing one or more modalities")

    #     return data

    