from collections import defaultdict
import glob
import os.path as osp

from .bases import BaseImageDataset


class Market1501_MM(BaseImageDataset):
    dataset_dir = 'market1501_to_RGBNT201_dark'

    # ---------- helper -----------------------------------------------------

    @staticmethod
    def _folder_for_mod(mod: str) -> str:
        """Convert logical modality → sub-folder name on disk."""
        if mod == 'RGB':
            return 'RGB'
        if mod == 'IR':
            return 'NI'      # IR images are stored under NI
        return mod           # 'TI' stays TI

    @staticmethod
    def _load_caption_file(path: str) -> dict:
        """Load a TSV caption file -> {filename: caption}."""
        d = {}
        if not osp.exists(path):
            return d
        with open(path, 'r') as f:
            for line in f:
                if '\t' not in line:
                    continue
                fname, cap = line.rstrip('\n').split('\t', 1)
                d[fname] = cap.strip()
        return d

    # ----------------------------------------------------------------------

    def __init__(
        self,
        root='',
        verbose=True,
        i_modality=None,
        c_modality=None,
        **kwargs,
    ):
        super().__init__()

        # which modalities to **load as images** / **captions**
        self.i_modality = i_modality if i_modality is not None else ['RGB']
        self.c_modality = c_modality if c_modality is not None else []

        # ---------- paths --------------------------------------------------
        self.dataset_dir  = osp.join(root, self.dataset_dir)
        self.train_dir    = osp.join(self.dataset_dir, 'train')
        self.query_dir    = osp.join(self.dataset_dir, 'query')
        self.gallery_dir  = osp.join(self.dataset_dir, 'gallery')

        self.caption_dir  = osp.join(root, 'cap_predictions', 'Market1501 Captions')
        self.train_prefix = 'Market1501-bbox-train'
        self.query_prefix = 'Market1501-query'
        self.test_prefix  = 'Market1501-bbox-test'
        # -------------------------------------------------------------------

        self._check_before_run()

        train   = self._process_split(self.train_dir,   self.train_prefix, relabel=True)
        query   = self._process_split(self.query_dir,   self.query_prefix, relabel=False)
        gallery = self._process_split(self.gallery_dir, self.test_prefix,  relabel=False)

        if verbose:
            print('=> market1501_to_RGBNT201_dark (multi-modal) loaded')
            self.print_dataset_statistics(train, query, gallery, self.c_modality)

        # save to instance
        self.train, self.query, self.gallery = train, query, gallery

        # stats
        for subset in [self.train, self.query, self.gallery]:
            (self.num_train_pids,
             self.num_train_items,
             self.num_train_cams,
             self.num_train_vids,
             self.num_train_images,
             self.num_train_captions) = self.get_imagedata_info(subset)

    # ---------------- core -------------------------------------------------

    def _process_split(self, dir_path, prefix, *, relabel=False):
        """Generic loader for train / query / gallery."""
        # scan filenames in RGB folder (serves as master index)
        rgb_files = glob.glob(osp.join(dir_path, 'RGB', '*.jpg'))
        pid_set = {int(osp.basename(p).split('_')[0][:4]) for p in rgb_files}
        pid2label = {pid: idx for idx, pid in enumerate(sorted(pid_set))}

        # build caption dicts once (one per caption-modality)
        caption_dicts = None
        if self.c_modality:
            caption_dicts = []
            for mod in self.c_modality:
                suf  = 'visible' if mod == 'RGB' else mod
                file = osp.join(self.caption_dir, f'{prefix}-{suf}.txt')
                caption_dicts.append(self._load_caption_file(file))

        data = []
        for rgb_path in rgb_files:
            fname   = osp.basename(rgb_path)
            pid     = int(fname.split('_')[0][:4])
            camid   = int(fname.split('_')[1][1]) - 1
            timeid  = int(fname.split('_')[3][1])

            if relabel:
                pid = pid2label[pid]

            # collect images for requested modalities
            img_paths, valid = [], True
            for mod in self.i_modality:
                folder = self._folder_for_mod(mod)
                path   = osp.join(dir_path, folder, fname)
                if not osp.exists(path):
                    valid = False
                    break
                img_paths.append(path)
            if not valid:
                continue

            # captions
            if caption_dicts:
                caps = [cap_d.get(fname, '') for cap_d in caption_dicts]
                data.append((img_paths, caps, pid, camid, timeid))
            else:
                data.append((img_paths, pid, camid, timeid))

        return data

    # ---------------- sanity checks ---------------------------------------

    def _check_before_run(self):
        for p in [self.dataset_dir, self.train_dir, self.query_dir, self.gallery_dir]:
            if not osp.exists(p):
                raise RuntimeError(f"Dataset path not found: {p}")