# encoding: utf-8
"""
Celebrity-ReID loader (RGB / IR / TI images + optional captions)

Author: sherlock (adapted)
"""

import os, os.path as osp
from glob import glob
from .bases import BaseImageDataset


class Celebrity(BaseImageDataset):
    """
    Directory layout (same style as Market):

      Data/
        ├──Celeb-reID/
        │    ├── train/
        │    │   ├── visible/*.jpg
        │    │   ├── IR/*.jpg
        │    │   └── TI/*.jpg
        │    ├── query/   (same three sub-folders)
        │    ├── gallery/ (same three sub-folders)
        └── cap_predictions/                  <-- captions root
            └── Celeb-ReID Captions/           <-- keep txt files here
                    Celeb-bbox-train-visible.txt
                    Celeb-bbox-train-IR.txt
                    ...
    """
    dataset_dir = 'Celeb-reID'

    # --------------------------------------------------------------------- #
    # constructor
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        root: str = '',
        verbose: bool = True,
        pid_begin: int = 0,
        i_modality=None,
        c_modality=None,
        **kwargs,
    ):
        super().__init__()

        # image / caption modalities requested by the caller
        self.i_modality = i_modality if i_modality else ['RGB']
        self.c_modality = c_modality if c_modality else []

        # -----------------------------------------------------------------
        # directory bookkeeping
        # -----------------------------------------------------------------
        self.dataset_dir  = osp.join(root, self.dataset_dir)
        self.train_dir    = osp.join(self.dataset_dir, 'train')
        self.query_dir    = osp.join(self.dataset_dir, 'query')
        self.gallery_dir  = osp.join(self.dataset_dir, 'gallery')

        self.caption_dir            = osp.join(self.dataset_dir, 'cap_predictions')
        self.train_caption_prefix   = 'Celeb-reID-train'    
        self.query_caption_prefix   = 'Celeb-reID-query'
        self.gallery_caption_prefix = 'Celeb-reID-gallery'

        self.pid_begin = pid_begin
        self._check_before_run()

        # -----------------------------------------------------------------
        # create splits
        # -----------------------------------------------------------------
        train   = self._process_dir(self.train_dir,   self.train_caption_prefix,   relabel=True)
        query   = self._process_dir(self.query_dir,   self.query_caption_prefix,   relabel=False, is_query=True)
        gallery = self._process_dir(self.gallery_dir, self.gallery_caption_prefix, relabel=False)

        if verbose:
            print('=> Celebrity loaded with multi-modality + caption support')
            self.print_dataset_statistics(train, query, gallery, self.c_modality)

        # public handles ---------------------------------------------------
        self.train, self.query, self.gallery = train, query, gallery

        ( self.num_train_pids,
          self.num_train_items,
          self.num_train_cams,
          self.num_train_vids,
          self.num_train_imgs,
          self.num_train_caps ) = self.get_imagedata_info(self.train)

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    @staticmethod
    def _load_caption_dict(txt_path: str):
        d = {}
        if not osp.exists(txt_path):
            return d
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or '\t' not in line:
                    continue
                fname, cap = line.split('\t', 1)
                d[fname.strip()] = cap.strip()
        return d

    def _load_caption_dicts(self, prefix: str):
        """Return list[dict] – one caption-dict per *caption* modality."""
        dicts = []
        for mod in self.c_modality:                       # ← note: caption list
            suff = 'visible' if mod == 'RGB' else mod
            path = osp.join(self.caption_dir, f'{prefix}-{suff}.txt')
            dicts.append(self._load_caption_dict(path))
        return dicts

    # --------------------------------------------------------------------- #
    # directory walker
    # --------------------------------------------------------------------- #
    def _process_dir(self, dir_path, prefix, *, relabel=False, is_query=False):
        """
        Build list items:
           ( [img_path_rgb,img_path_ir,img_path_ti], [capR,capI,capT], pid, cam, 1 )
        If captions disabled ⇒ second element omitted.
        """
        vis_files = glob(osp.join(dir_path, 'visible', '*.jpg'))
        pid_container = { int(osp.basename(p).split('_')[0]) for p in vis_files }
        pid2label = { pid: idx for idx, pid in enumerate(sorted(pid_container)) }

        cap_dicts = self._load_caption_dicts(prefix) if self.c_modality else None
        out = []

        for vis_path in vis_files:
            fname   = osp.basename(vis_path)
            pid_raw = int(fname.split('_')[0])
            pid     = self.pid_begin + (pid2label[pid_raw] if relabel else pid_raw)

            # gather image paths ------------------------------------------------
            img_paths, ok = [], True
            for mod in self.i_modality:
                folder = 'visible' if mod == 'RGB' else mod
                pth    = osp.join(dir_path, folder, fname)
                if not osp.exists(pth):
                    ok = False; break
                img_paths.append(pth)
            if not ok:
                continue  # some modality missing – skip sample

            # gather captions ---------------------------------------------------
            cap_list = []
            if self.c_modality:
                for d in cap_dicts:
                    cap_list.append(d.get(fname, ''))

            camid = 0 if is_query else 1
            if self.c_modality:
                out.append((img_paths, cap_list, pid, camid, 1))
            else:
                out.append((img_paths,          pid, camid, 1))

        return out

    # --------------------------------------------------------------------- #
    # misc
    # --------------------------------------------------------------------- #
    def _check_before_run(self):
        for p in (self.train_dir, self.query_dir, self.gallery_dir):
            if not osp.exists(p):
                raise RuntimeError(f"'{p}' is missing")


__all__ = ['Celebrity']
