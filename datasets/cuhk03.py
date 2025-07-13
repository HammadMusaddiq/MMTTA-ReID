# cuhk03.py
import os
import os.path as osp
import json
from   .bases import BaseImageDataset

class CUHK03(BaseImageDataset):
    """
    CUHK03 *labeled* split with multi-modality and optional caption support.
    """
    dataset_dir   = 'cuhk03'
    _caption_base = 'cuhk03-im-labeled'          # ← matches your txt files

    def __init__(
        self,
        root      ='',
        split_id  =0,
        verbose   =True,
        i_modality=None,   # list like ["RGB","IR","TI"]
        c_modality=None,   # [] ⇒ disable captions  |  ["RGB"]  |  ["RGB","IR",...]
        **kwargs
    ):
        super().__init__()
        # ------------------------------------------------------------------ #
        # config
        # ------------------------------------------------------------------ #
        self.i_modality = i_modality if i_modality is not None else ["RGB"]
        self.c_modality = c_modality if c_modality is not None else []

        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.img_dir     = osp.join(self.dataset_dir, 'images_labeled')
        self.split_path  = osp.join(self.dataset_dir, 'splits_new_labeled.json')

        # caption txt files live next to the images, e.g.
        #   cuhk03/im_labeled/cap_predictions/cuhk03-im-labeled-visible.txt
        self.caption_dir = osp.join(self.dataset_dir, 'cap_predictions', 'cuhk03 Captions')

        self._check_before_run()

        # ------------------------------------------------------------------ #
        # caption dicts (if any) – only 1 set (train/query/gallery share them)
        # ------------------------------------------------------------------ #
        self.caption_dicts = (
            self._load_caption_dicts(self._caption_base) if self.c_modality else None
        )

        # ------------------------------------------------------------------ #
        # load split
        # ------------------------------------------------------------------ #
        with open(self.split_path, 'r') as f:
            splits = json.load(f)
        split = splits[split_id]

        train   = self._process_names(split['train'  ], relabel=True )
        query   = self._process_names(split['query'  ], relabel=False)
        gallery = self._process_names(split['gallery'], relabel=False)

        if verbose:
            print("=> CUHK03 (labeled) loaded with multi-modality support")
            self.print_dataset_statistics(train, query, gallery, self.c_modality)

        self.train, self.query, self.gallery = train, query, gallery

        # summary numbers
        self.num_train_pids, *_ = self.get_imagedata_info(self.train)
        self.num_query_pids, *_ = self.get_imagedata_info(self.query)
        self.num_gallery_pids,*_ = self.get_imagedata_info(self.gallery)

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    def _check_before_run(self):
        if not osp.exists(self.dataset_dir):  raise RuntimeError(f"{self.dataset_dir} missing")
        if not osp.exists(self.img_dir):      raise RuntimeError(f"{self.img_dir} missing")
        if not osp.exists(self.split_path):   raise RuntimeError(f"{self.split_path} missing")
        if self.c_modality and not osp.exists(self.caption_dir):
            raise RuntimeError(f"{self.caption_dir} (captions) missing")

    # ---- captions -------------------------------------------------------- #
    def _load_caption_dict(self, txt_file):
        dic = {}
        with open(txt_file, 'r') as f:
            for line in f:
                if '\t' not in line: continue
                fname, cap = line.rstrip('\n').split('\t', 1)
                dic[fname] = cap.strip()
        return dic

    def _load_caption_dicts(self, base_prefix):
        """return list[dict] in the order of self.c_modality"""
        dicts = []
        for mod in self.c_modality:
            mod_suffix = 'visible' if mod == "RGB" else mod
            path = osp.join(self.caption_dir, f"{base_prefix}-{mod_suffix}.txt")
            dicts.append(self._load_caption_dict(path))
        return dicts

    # ---- core processing ------------------------------------------------- #
    def _process_names(self, name_tuples, relabel=False):
        """
        `name_tuples` element looks like (relative_path, pid, camid)
        We convert it into:
            (
              [path_vis, path_IR, path_TI],  # or fewer depending on i_modality
              captions (list|None),
              pid, camid, view_id(=1)
            )
        """
        data, pid_container = [], set()

        for rel_path, pid, camid in name_tuples:
            filename = osp.basename(rel_path)          # e.g. 0001_c1s1_001051.jpg

            # ---------- collect image paths ----------
            img_paths, valid = [], True
            for mod in self.i_modality:
                folder = 'visible' if mod == "RGB" else mod
                full   = osp.join(self.img_dir, folder, filename)
                if not osp.exists(full):
                    valid = False; break
                img_paths.append(full)

            if not valid: continue
            pid_container.add(pid)

            # ---------- collect captions (optional) ----------
            captions = None
            if self.c_modality:
                captions = [
                    dic.get(filename, "") for dic in self.caption_dicts
                ]

            data.append((img_paths, captions, pid, camid, 1))  # view_id = 1

        # relabel if requested
        if relabel:
            id_map = {pid:i for i,pid in enumerate(sorted(pid_container))}
            data = [
                (paths, caps, id_map[pid], cam, view)
                for paths, caps, pid, cam, view in data
            ]

        return data
