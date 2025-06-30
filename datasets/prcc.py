import os.path as osp
from .bases import BaseImageDataset
import os

class PRCC(BaseImageDataset):
    dataset_dir = 'prcc/rgb'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(PRCC, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.pid_begin = pid_begin

        self.cam_map = {'A': 0, 'B': 1, 'C': 2}

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.val_dir = osp.join(self.dataset_dir, 'val')
        self.query_dir = osp.join(self.dataset_dir, 'test/A')
        self.gallery_dir = osp.join(self.dataset_dir, 'test/C')

        train, train_id_list = self.init_dataset(self.train_dir)
        val, _ = self.init_dataset(self.val_dir)
        train = self.relabel_dataset(train, train_id_list)

        query, query_id_list = self.init_test_dataset(self.query_dir, is_query=True)
        gallery, gallery_id_list = self.init_test_dataset(self.gallery_dir, is_query=False)

        self.train = train
        self.query = query
        self.gallery = gallery
        self.num_train_pids = len(train_id_list)
        self.num_train_cams = 3
        self.num_train_vids = 1
        if verbose:
            print("=> PRCC loaded")
            self.print_dataset_statistics(train, query, gallery)

    def relabel_dataset(self, dataset, id_list):
        new_dataset = []
        for img_paths, id, cam, index in dataset:
            new_dataset.append((img_paths, id_list.index(id) + self.pid_begin, cam, index))
        return new_dataset

    def init_dataset(self, root, must_in_ids=None):
        id_list = []
        dataset = []

        for _id in os.listdir(root):
            if int(_id) not in id_list:
                id_list.append(int(_id))
            if osp.isdir(osp.join(root, _id)):
                # Assuming 'visible' contains the filenames to iterate over
                for filename in os.listdir(osp.join(root, _id, 'visible')):
                    # Construct paths for all modalities
                    visible_path = osp.join(root, _id, 'visible', filename)
                    ir_path = osp.join(root, _id, 'IR', filename)
                    ti_path = osp.join(root, _id, 'TI', filename)
                    img_paths = [visible_path, ir_path, ti_path]
                    
                    # Extract camera information from filename (e.g., 'A' from 'A_001.jpg')
                    cam = self.cam_map[filename.split('_')[0]]
                    dataset.append((img_paths, int(_id), cam, 1))

        return dataset, id_list

    @staticmethod
    def init_test_dataset(root, is_query=False):
        id_list = []
        dataset = []
        per_cam = 0 if is_query else 1  # Camera index for query and gallery
        for _id in os.listdir(root):
            if int(_id) not in id_list:
                id_list.append(int(_id))
            if osp.isdir(osp.join(root, _id)):
                for filename in os.listdir(osp.join(root, _id, 'visible')):
                    # Construct paths for all modalities
                    visible_path = osp.join(root, _id, 'visible', filename)
                    ir_path = osp.join(root, _id, 'IR', filename)
                    ti_path = osp.join(root, _id, 'TI', filename)
                    img_paths = [visible_path, ir_path, ti_path]
                    dataset.append((img_paths, int(_id), per_cam, 1))
        return dataset, id_list