# msmt17.py
import os, os.path as osp
from   .bases import BaseImageDataset

class MSMT17(BaseImageDataset):
    """
    MSMT17 V2 with multi-modality images and optional caption loading.
    """
    dataset_dir   = "MSMT17_V2"
    _train_prefix = "MSMT17-v2-train"
    _test_prefix  = "MSMT17-v2-test"

    def __init__(
        self,
        root      ='',
        verbose   =True,
        pid_begin =0,
        i_modality=None,     # e.g. ["RGB","IR","TI"]
        c_modality=None,     # [] | ["RGB"] | ["RGB","IR","TI"]
        **kwargs
    ):
        super().__init__()
        # ------------------------------------------------------------------ #
        # config
        # ------------------------------------------------------------------ #
        self.i_modality = i_modality if i_modality is not None else ["RGB"]
        self.c_modality = c_modality if c_modality is not None else []
        self.pid_begin  = pid_begin

        # directories
        self.dataset_dir   = osp.join(root, self.dataset_dir)
        self.train_dir     = osp.join(self.dataset_dir, 'mask_train_v2')
        self.test_dir      = osp.join(self.dataset_dir, 'mask_test_v2')
        self.caption_dir   = osp.join(root, 'cap_predictions', 'MSMT17-v2 Captions') # fixed path for captions

        self.list_train    = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val      = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query    = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery  = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()

        # caption dicts (train / test share the same structure)
        self.train_caps = (
            self._load_caption_dicts(self._train_prefix) if self.c_modality else None
        )
        self.test_caps  = (
            self._load_caption_dicts(self._test_prefix ) if self.c_modality else None
        )

        # ------------------------------------------------------------------ #
        # build splits
        # ------------------------------------------------------------------ #
        train   = self._process_list(self.train_dir , self.list_train , self.train_caps)
        val     = self._process_list(self.train_dir , self.list_val   , self.train_caps)
        train  += val                                                   # official practice
        query   = self._process_list(self.test_dir  , self.list_query , self.test_caps )
        gallery = self._process_list(self.test_dir  , self.list_gallery, self.test_caps )

        if verbose:
            print("=> MSMT17 V2 loaded with multi-modality support")
            self.print_dataset_statistics(train, query, gallery, self.c_modality)

        self.train, self.query, self.gallery = train, query, gallery
        self.num_train_pids, *_   = self.get_imagedata_info(self.train)
        self.num_query_pids, *_   = self.get_imagedata_info(self.query)
        self.num_gallery_pids,*_  = self.get_imagedata_info(self.gallery)

    # --------------------------------------------------------------------- #
    # file checks
    # --------------------------------------------------------------------- #
    def _check_before_run(self):
        must = [self.dataset_dir, self.train_dir, self.test_dir,
                self.list_train, self.list_val, self.list_query, self.list_gallery]
        for p in must:
            if not osp.exists(p):
                raise RuntimeError(f"Required file / dir not found: {p}")
        if self.c_modality and not osp.exists(self.caption_dir):
            raise RuntimeError(f"Caption directory missing: {self.caption_dir}")

    # --------------------------------------------------------------------- #
    # caption helpers
    # --------------------------------------------------------------------- #
    def _load_caption_dict(self, txt_file):
        dic = {}
        with open(txt_file, 'r') as f:
            for line in f:
                if '\t' not in line: continue
                fname, cap = line.rstrip('\n').split('\t', 1)
                dic[fname] = cap.strip()
        return dic

    def _load_caption_dicts(self, prefix):
        """Return list[dict] following self.c_modality order."""
        dicts = []
        for mod in self.c_modality:
            mod_suffix = 'visible' if mod == "RGB" else mod
            txt = osp.join(self.caption_dir, f"{prefix}-{mod_suffix}.txt")
            dicts.append(self._load_caption_dict(txt))
        return dicts

    # --------------------------------------------------------------------- #
    # main parser
    # --------------------------------------------------------------------- #
    def _process_list(self, img_root, list_file, caption_dicts):
        """
        list_file lines:  <rel_path> <pid>\n
        rel_path example: 0001/0001_00000_00.jpg
        camid is encoded in the filename: *_c??*  (official script → index 2 after split '_')
        """
        dataset, pid_container = [], set()

        with open(list_file, 'r') as f:
            lines = [l.strip() for l in f.readlines() if l.strip()]

        for line in lines:
            rel_path, pid_str = line.split(' ')
            pid   = int(pid_str)
            camid = int(rel_path.split('_')[2])  # same as original loader

            filename  = osp.basename(rel_path)
            pid_dir   = rel_path.split('/')[0]    # 0001

            # -------- collect image paths (selected i_modality) -------------
            img_paths, valid = [], True
            for mod in self.i_modality:
                folder = 'visible' if mod == "RGB" else mod
                full   = osp.join(img_root, pid_dir, folder, filename)
                if not osp.exists(full):
                    valid = False; break
                img_paths.append(full)
            if not valid: continue

            # -------- captions (optional) ------------------------------------
            captions = None
            if caption_dicts:
                captions = [dic.get(filename, "") for dic in caption_dicts]

            dataset.append((
                img_paths,
                captions,
                self.pid_begin + pid,
                camid - 1,   # make 0-based
                1            # view_id constant
            ))
            pid_container.add(pid)

        return dataset
