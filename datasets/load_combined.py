from .market1501    import Market1501
from .msmt17        import MSMT17
from .cuhk03        import CUHK03
from .rgbnt201      import RGBNT201
from .celebrity     import Celebrity
from .market1501_MM import Market1501_MM
from .prcc          import PRCC
from .bases import BaseImageDataset
import itertools

class CombinedSet(BaseImageDataset):
    """
    `sub_datasets` is a list of instantiated BaseImageDataset objects.
    The class concatenates their {train, query, gallery} lists *as is*
    and then computes global summary statistics so the rest of the code
    (sampler, logger table, etc.) can stay unchanged.
    """

    def __init__(self, sub_datasets):
        super().__init__()
        self.subs = sub_datasets

        # ---------------------------------------------------------------
        # 1) concatenate samples
        # ---------------------------------------------------------------
        self.train   = list(itertools.chain.from_iterable(d.train   for d in self.subs))
        self.query   = list(itertools.chain.from_iterable(d.query   for d in self.subs))
        self.gallery = list(itertools.chain.from_iterable(d.gallery for d in self.subs))

        # ---------------------------------------------------------------
        # 2) global statistics expected by `make_dataloader`
        # ---------------------------------------------------------------
        # PIDs (after the bias-shift you already apply in `load_multi_dataset`)
        self.num_train_pids  = sum(d.num_train_pids  for d in self.subs)
        self.num_query_pids  = sum(getattr(d, "num_query_pids", 0)  for d in self.subs)
        self.num_gallery_pids = sum(getattr(d, "num_gallery_pids", 0) for d in self.subs)

        # “items” (samples before transforms) – keep original meaning
        self.num_train_items = len(self.train)
        self.num_train_images = self.num_train_items   # alias used by some print-helpers

        # cams: gather unique camera IDs across all train samples
        train_cam_ids = {
            sample[2] if isinstance(sample[2], int) else tuple(sample[2])
            for d in self.subs for sample in d.train
        }
        self.num_train_cams = len(train_cam_ids) if train_cam_ids else 1

        # views: take the **max** among sub-datasets (safe fallback)
        self.num_train_vids = max(getattr(d, "num_train_vids", 1) for d in self.subs)

        # ---------------------------------------------------------------
        # 3) optional – let the nice summary table work:
        # ---------------------------------------------------------------
        self.num_query_imgs   = len(self.query)
        self.num_gallery_imgs = len(self.gallery)


def load_multi_dataset(cfg, root, c_modality = None):
    """
    Decide which datasets to pack together based on cfg.DATASETS.NAMES.
    Example yaml:
        DATASETS:
          NAMES: ["market1501", "msmt17", "celebrity"]
    """
    name_map = {
        "market1501":    Market1501,
        "msmt17":        MSMT17,
        "cuhk03":        CUHK03,
        "rgbnt201":      RGBNT201,
        "celebrity":     Celebrity,
        "market1501_mm": Market1501_MM,
        "prcc":          PRCC,
    }

    ds_objs = []
    pid_bias = 0
    for name in cfg.DATASETS.NAMES:
        cls   = name_map[name.lower()]
        ds    = cls(root=root,
                    i_modality=cfg.IMAGE_MODALITY,
                    # c_modality=(cfg.IMAGE_MODALITY if (
                    #                 cfg.CAPTION.ENABLE and cfg.CAPTION.STRATEGY=="matched")
                    #             else (["RGB"] if cfg.CAPTION.ENABLE else [])),
                    c_modality = c_modality,
                    pid_begin=pid_bias,
                    verbose= not cfg.DATASETS.MULTI)
        pid_bias += ds.num_train_pids     # avoid id collision *inside* each loader
        ds_objs.append(ds)

    combined = CombinedSet(ds_objs)

    combined.print_dataset_statistics(combined.train,
                                      combined.query,
                                      combined.gallery, c_modality)
    
    return combined
