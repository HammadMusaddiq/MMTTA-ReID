import os.path as osp
from .bases import BaseImageDataset
import os

class Celebrity(BaseImageDataset):
    dataset_dir = 'Celeb-reID'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(Celebrity, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'gallery')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, False)
        query = self._process_dir(self.query_dir, True)
        gallery = self._process_dir(self.gallery_dir, False)

        if verbose:
            print("=> Celebrity loaded with multi-modality support")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(
            self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(
            self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(
            self.gallery)

    def _check_before_run(self):
        """Check if all required directories are available before proceeding."""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, is_query):
        """
        Process the directory to handle multi-modality images.
        Assumes subdirectories 'visible', 'IR', and 'TI' exist within dir_path.
        Returns a list of tuples, each containing a list of image paths (for visible, IR, TI),
        a person ID, a camera ID, and a constant 1.
        """
        dataset = []
        modality = 'visible'  # Use 'visible' to list the base filenames
        modality_dir = osp.join(dir_path, modality)
        for img_name in os.listdir(modality_dir):
            if img_name.endswith('.jpg'):
                pid = int(img_name.split('_')[0]) - 1
                # Construct paths for all three modalities
                img_path_RGB = osp.join(dir_path, 'visible', img_name)
                img_path_NI = osp.join(dir_path, 'IR', img_name)
                img_path_TI = osp.join(dir_path, 'TI', img_name)

                if osp.exists(img_path_RGB) and osp.exists(img_path_NI) and osp.exists(img_path_TI):
                    img_paths = []
                    img_paths.append(img_path_RGB)
                    img_paths.append(img_path_NI)
                    img_paths.append(img_path_TI)
                    
                    cam_id = 0 if is_query else 1
                    dataset.append((img_paths, self.pid_begin + pid, cam_id, 1))
        return dataset