# prcc.py
import os, os.path as osp
from   .bases import BaseImageDataset

class PRCC(BaseImageDataset):
    """
    Partial REID in Camouflage Clothing (PRCC) with RGB / IR / TI
    and optional caption support.

    Caption files shipped with the dataset:
        prcc-train-visible.txt   prcc-train-IR.txt   prcc-train-TI.txt
        prcc-val-visible.txt     ...
        prcc-test-A-visible.txt  ...
        prcc-test-B-*.txt        prcc-test-C-*.txt
    """
    dataset_dir     = "prcc/rgb"                     # root/prcc/{train,val,test}
    _train_prefix   = "prcc-train"
    _val_prefix     = "prcc-val"
    _query_prefix   = "prcc-test-A"              # test-A → query
    _gallery_prefix = "prcc-test-C"              # test-C → gallery

    cam_map = {'A': 0, 'B': 1, 'C': 2}

    # --------------------------------------------------------------------- #
    def __init__(
        self,
        root      ='',
        verbose   =True,
        pid_begin =0,
        i_modality=None,          # ["RGB","IR","TI"] subset
        c_modality=None,          # [] | ["RGB"] | same as i_modality
        **kwargs
    ):
        super().__init__()
        # configuration ---------------------------------------------------- #
        self.i_modality = i_modality if i_modality is not None else ["RGB"]
        self.c_modality = c_modality if c_modality is not None else []
        self.pid_begin  = pid_begin

        # dirs ------------------------------------------------------------- #
        self.dataset_dir  = osp.join(root, self.dataset_dir)
        self.train_dir    = osp.join(self.dataset_dir, "train")
        self.val_dir      = osp.join(self.dataset_dir, "val")
        self.query_dir    = osp.join(self.dataset_dir, "test", "A")
        self.gallery_dir  = osp.join(self.dataset_dir, "test", "C")
        self.caption_dir  = osp.join(self.dataset_dir, "cap_predictions", "PRCC Captions")

        self._check_before_run()

        # caption dict collections ---------------------------------------- #
        self.train_caps   = self._load_caption_dicts(self._train_prefix ) if self.c_modality else None
        self.val_caps     = self._load_caption_dicts(self._val_prefix   ) if self.c_modality else None
        self.query_caps   = self._load_caption_dicts(self._query_prefix ) if self.c_modality else None
        self.gallery_caps = self._load_caption_dicts(self._gallery_prefix) if self.c_modality else None

        # build splits ----------------------------------------------------- #
        train   = self._process_folder(self.train_dir  , self.train_caps)
        val     = self._process_folder(self.val_dir    , self.val_caps  )
        train  += val                                                      # official practice
        query   = self._process_folder(self.query_dir , self.query_caps , is_query=True )
        gallery = self._process_folder(self.gallery_dir, self.gallery_caps, is_query=False)

        if verbose:
            print("=> PRCC loaded with multi-modality support")
            self.print_dataset_statistics(train, query, gallery, self.c_modality)

        self.train, self.query, self.gallery = train, query, gallery
        self.num_train_pids, *_   = self.get_imagedata_info(self.train)
        self.num_query_pids, *_   = self.get_imagedata_info(self.query)
        self.num_gallery_pids,*_  = self.get_imagedata_info(self.gallery)

    # --------------------------------------------------------------------- #
    # helpers
    # --------------------------------------------------------------------- #
    def _check_before_run(self):
        for p in [self.dataset_dir, self.train_dir, self.val_dir,
                  self.query_dir,  self.gallery_dir]:
            if not osp.exists(p):
                raise RuntimeError(f"Required dir missing: {p}")
        if self.c_modality and not osp.exists(self.caption_dir):
            raise RuntimeError(f"Caption directory missing: {self.caption_dir}")

    # -------- caption ---------------------------------------------------- #
    def _load_caption_dict(self, txt_path):
        dic = {}
        with open(txt_path, 'r') as f:
            for line in f:
                if '\t' not in line: continue
                fname, cap = line.rstrip('\n').split('\t', 1)
                dic[fname] = cap.strip()
        return dic

    def _load_caption_dicts(self, prefix):
        lst = []
        for mod in self.c_modality:
            suf   = 'visible' if mod == "RGB" else mod
            txt   = osp.join(self.caption_dir, f"{prefix}-{suf}.txt")
            lst.append(self._load_caption_dict(txt))
        return lst

    # -------- main parser ------------------------------------------------ #
    def _process_folder(self, folder_root, caption_dicts, is_query=False):
        """
        Each identity folder contains sub-folders {visible, IR, TI}.  
        Filenames encode camera as <CamLetter>_XXXX.jpg.
        """
        dataset = []
        for pid_str in os.listdir(folder_root):
            pid_dir = osp.join(folder_root, pid_str)
            if not osp.isdir(pid_dir): continue
            pid = self.pid_begin + int(pid_str)

            # iterate over visible filenames (they are the superset)
            for fname in os.listdir(osp.join(pid_dir, "visible")):
                cam_letter = fname.split('_')[0]
                camid      = self.cam_map.get(cam_letter, 0)
                view_id    = 1                                          # constant

                # gather image paths for selected modalities
                img_paths, valid = [], True
                for mod in self.i_modality:
                    sub = 'visible' if mod == "RGB" else mod
                    full = osp.join(pid_dir, sub, fname)
                    if not osp.exists(full): valid=False; break
                    img_paths.append(full)
                if not valid: continue

                # captions
                captions = None
                if caption_dicts:
                    captions = [dic.get(fname, "") for dic in caption_dicts]

                dataset.append((img_paths, captions, pid, camid, view_id) if captions
                               else (img_paths, pid, camid, view_id))
        return dataset
