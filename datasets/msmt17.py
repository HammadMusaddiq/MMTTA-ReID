
import glob
import re

import os.path as osp

from .bases import BaseImageDataset


class MSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'MSMT17_V2'

    def __init__(self, root='', verbose=True, pid_begin=0, modality=None, **kwargs):
        super(MSMT17, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'mask_train_v2')
        self.test_dir = osp.join(self.dataset_dir, 'mask_test_v2')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path)
        val = self._process_dir(self.train_dir, self.list_val_path)
        train += val
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)
        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    # def _process_dir(self, dir_path, list_path):
    #     with open(list_path, 'r') as txt:
    #         lines = txt.readlines()
    #     dataset = []
    #     pid_container = set()
    #     cam_container = set()
    #     for img_idx, img_info in enumerate(lines):
    #         img_path, pid = img_info.split(' ')
    #         pid = int(pid)  # No need to relabel
    #         camid = int(img_path.split('_')[2])

    #         # Extract filename and PID folder from the image path
    #         filename = osp.basename(img_path)
    #         pid_folder = img_path.split('/')[0]  # The <pid> folder

    #         # Construct paths for all three modalities
    #         img_paths = [
    #             osp.join(dir_path, pid_folder, 'visible', filename),
    #             osp.join(dir_path, pid_folder, 'IR', filename),
    #             osp.join(dir_path, pid_folder, 'TI', filename),
    #         ]

    #         # Append the tuple with the list of image paths
    #         # print(f"PID type, {type(pid)}, PID: {pid}, list_path {list_path}")
    #         dataset.append((img_paths, self.pid_begin + pid, camid-1, 1))
    #         pid_container.add(pid)
    #         cam_container.add(camid)

    #     print(cam_container, 'cam_container')
    #     # print(pid_container, 'pid_container')
    #     # Check if PIDs start from 0 and increment by 1
    #     for idx, pid in enumerate(sorted(pid_container)):
    #         assert idx == pid, "PIDs must be consecutive starting from 0"
    #     return dataset
    
    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # No need to relabel
            camid = int(img_path.split('_')[2])

            # Extract filename and PID folder from the image path
            filename = osp.basename(img_path)
            pid_folder = img_path.split('/')[0]  # The <pid> folder

            # Initialize a list to hold the selected modalities
            img_paths = []

            # Only add paths for modalities present in the provided list
            if "RGB" in self.modality:
                img_paths.append(osp.join(dir_path, pid_folder, 'visible', filename))
            if "IR" in self.modality:
                img_paths.append(osp.join(dir_path, pid_folder, 'IR', filename))
            if "TI" in self.modality:
                img_paths.append(osp.join(dir_path, pid_folder, 'TI', filename))


            # Check if all three modality images exist
            if all(osp.exists(path) for path in img_paths):
                # Append the tuple with the list of image paths
                dataset.append((img_paths, self.pid_begin + pid, camid - 1, 1))
                pid_container.add(pid)
                cam_container.add(camid)

        # print(cam_container, 'cam_container')
        return dataset
    