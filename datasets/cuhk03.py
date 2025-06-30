import os.path as osp
import json
from .bases import BaseImageDataset

class CUHK03(BaseImageDataset):
    dataset_dir = 'cuhk03'

    def __init__(self, root='', split_id=0, verbose=True, **kwargs):
        super(CUHK03, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.data_dir = osp.join(self.dataset_dir, 'images_labeled')
        self.split_path = osp.join(self.dataset_dir, 'splits_new_labeled.json')

        with open(self.split_path, 'r') as f:
            splits = json.load(f)
        split = splits[split_id]
        train_names = split['train']
        query_names = split['query']
        gallery_names = split['gallery']

        train = self._process_names(train_names, relabel=True)
        query = self._process_names(query_names, relabel=False)
        gallery = self._process_names(gallery_names, relabel=False)

        if verbose:
            print("=> CUHK03 loaded")
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
        if not osp.exists(self.data_dir):
            raise RuntimeError("'{}' is not available".format(self.data_dir))
        if not osp.exists(self.split_path):
            raise RuntimeError("'{}' is not available".format(self.split_path))

    def _process_names(self, names, relabel=False):
        data = []
        pid_container = set()
        for name in names:
            # Parse filename, assuming format "PID_index_camID_timeID.png"
            image_path, pid, camid = name
            filename = image_path.split('\\')[-1]
            img_paths = [
                osp.join(self.data_dir, 'visible', filename),
                osp.join(self.data_dir, 'IR', filename),
                osp.join(self.data_dir, 'TI', filename)
            ]
            data.append((img_paths, pid, camid, 1))  # View ID set to 1
            pid_container.add(pid)
        if relabel:
            pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
            data = [(img_paths, pid2label[pid], camid, 1) for img_paths, pid, camid, _ in data]
        return data