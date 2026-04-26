import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .bases import ImageDataset
from timm.data.random_erasing import RandomErasing
from .sampler import RandomIdentitySampler
from .dukemtmcreid import DukeMTMCreID
from .market1501 import Market1501
from .msmt17 import MSMT17
from .sampler_ddp import RandomIdentitySampler_DDP
import torch.distributed as dist
from .occ_duke import OCC_DukeMTMCreID
from .vehicleid import VehicleID
from .veri import VeRi
from .rgbnt201 import RGBNT201
from .rgbnt100 import RGBNT100
from .rgbn300 import RGBN300
from .cuhk03 import CUHK03
from .prcc import PRCC
from .celebrity import Celebrity
from .market1501_MM import Market1501_MM

from datasets.load_combined import load_multi_dataset

__factory = {
    'market1501': Market1501,
    'dukemtmc': DukeMTMCreID,
    'msmt17': MSMT17,
    'occ_duke': OCC_DukeMTMCreID,
    'veri': VeRi,
    'VehicleID': VehicleID,
    'rgbnt201': RGBNT201,
    'rgbnt100': RGBNT100,
    'rgbn300': RGBN300,
    'cuhk03': CUHK03,
    'prcc': PRCC,
    'celeb_reid': Celebrity,
    'market1501_MM': Market1501_MM,
}


# def train_collate_fn(batch):
#     """
#     # collate_fn这个函数的输入就是一个list，list的长度是一个batch size，list中的每个元素都是__getitem__得到的结果
#     """
#     imgs_1, imgs_2, imgs_3, pids, camids, viewids , _ = zip(*batch)
#     pids = torch.tensor(pids, dtype=torch.int64)
#     viewids = torch.tensor(viewids, dtype=torch.int64)
#     camids = torch.tensor(camids, dtype=torch.int64)
#     return torch.stack(imgs_1, dim=0), torch.stack(imgs_2, dim=0), torch.stack(imgs_3, dim=0), pids, camids, viewids,

def train_collate_fn(batch):
    """
    Dynamic collate_fn that handles image-only or image+caption data.
    """
    # Check if captions are included in the batch
    if isinstance(batch[0][3], list):  # If 4th element is a list, it's caption list
        imgs_1, imgs_2, imgs_3, captions, pids, camids, viewids, _ = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        return torch.stack(imgs_1, dim=0), torch.stack(imgs_2, dim=0), torch.stack(imgs_3, dim=0), list(captions), pids, camids, viewids
    else:
        imgs_1, imgs_2, imgs_3, pids, camids, viewids, _ = zip(*batch)
        pids = torch.tensor(pids, dtype=torch.int64)
        camids = torch.tensor(camids, dtype=torch.int64)
        viewids = torch.tensor(viewids, dtype=torch.int64)
        return torch.stack(imgs_1, dim=0), torch.stack(imgs_2, dim=0), torch.stack(imgs_3, dim=0), pids, camids, viewids

def train_collate_fn_dual(batch):
    """
    Collate function for training data, expecting two modalities.
    Input: list of tuples from __getitem__, each with (img_1, img_2, pid, camid, trackid, filename)
    Output: stacked tensors for two modalities, PIDs, camera IDs, and track IDs.
    """
    imgs_1, imgs_2, pids, camids, trackids, _ = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    trackids = torch.tensor(trackids, dtype=torch.int64)
    camids = torch.tensor(camids, dtype=torch.int64)
    return torch.stack(imgs_1, dim=0), torch.stack(imgs_2, dim=0), pids, camids, trackids

# def val_collate_fn(batch):
#     imgs_1, imgs_2, imgs_3, pids, camids, viewids, img_paths = zip(*batch)
#     pids = torch.tensor(pids, dtype=torch.int64)
#     viewids = torch.tensor(viewids, dtype=torch.int64)
#     camids_batch = torch.tensor(camids, dtype=torch.int64)
#     camids = torch.tensor(camids, dtype=torch.int64)
#     return torch.stack(imgs_1, dim=0), torch.stack(imgs_2, dim=0), torch.stack(imgs_3, dim=0), pids, camids, viewids

# def val_collate_fn(batch):
#     if isinstance(batch[0][3], list):  # Captions included
#         imgs_1, imgs_2, imgs_3, captions, pids, camids, viewids, img_paths = zip(*batch)
#         return (
#             torch.stack(imgs_1, dim=0),
#             torch.stack(imgs_2, dim=0),
#             torch.stack(imgs_3, dim=0),
#             list(captions),
#             torch.tensor(pids, dtype=torch.int64),
#             torch.tensor(camids, dtype=torch.int64),
#             torch.tensor(viewids, dtype=torch.int64),
#             # list(img_paths)
#         )
#     else:  # No captions
#         imgs_1, imgs_2, imgs_3, pids, camids, viewids, img_paths = zip(*batch)
#         pids = torch.tensor(pids, dtype=torch.int64)
#         viewids = torch.tensor(viewids, dtype=torch.int64)
#         camids_batch = torch.tensor(camids, dtype=torch.int64)
#         camids = torch.tensor(camids, dtype=torch.int64)
#         return torch.stack(imgs_1, dim=0), torch.stack(imgs_2, dim=0), torch.stack(imgs_3, dim=0), pids, camids, viewids

def val_collate_fn(batch):
    if isinstance(batch[0][3], list):  # Captions included
        imgs_1, imgs_2, imgs_3, captions, pids, camids, viewids, img_paths = zip(*batch)

        # Flatten and copy captions properly
        if isinstance(captions[0], list):
            # captions is tuple of list → make list of lists
            captions = [cap for cap in captions]
        else:
            # fallback (shouldn't hit)
            captions = list(captions)

        return (
            torch.stack(imgs_1, dim=0),
            torch.stack(imgs_2, dim=0),
            torch.stack(imgs_3, dim=0),
            captions,  # keep as list, do NOT touch with torch.tensor()
            torch.tensor(pids, dtype=torch.int64),
            torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64),
        )
    else:
        imgs_1, imgs_2, imgs_3, pids, camids, viewids, img_paths = zip(*batch)
        return (
            torch.stack(imgs_1, dim=0),
            torch.stack(imgs_2, dim=0),
            torch.stack(imgs_3, dim=0),
            torch.tensor(pids, dtype=torch.int64),
            torch.tensor(camids, dtype=torch.int64),
            torch.tensor(viewids, dtype=torch.int64),
        )


def make_dataloader(cfg):
    train_transforms = T.Compose([
            T.Resize(cfg.INPUT.SIZE_TRAIN, interpolation=3),
            # T.RandomHorizontalFlip(p=cfg.INPUT.PROB),
            T.Pad(cfg.INPUT.PADDING),
            T.RandomCrop(cfg.INPUT.SIZE_TRAIN),
            T.ToTensor(),
            T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD),
            RandomErasing(probability=cfg.INPUT.RE_PROB, mode='pixel', max_count=1, device='cpu'),
            # RandomErasing(probability=cfg.INPUT.RE_PROB, mean=cfg.INPUT.PIXEL_MEAN)
        ])

    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_workers = cfg.DATALOADER.NUM_WORKERS

    image_modality = cfg.IMAGE_MODALITY  # e.g., ["RGB", "IR", "TI"]
    caption_modality = []

    if cfg.CAPTION.ENABLE:
        if cfg.CAPTION.STRATEGY == "matched":
            caption_modality = image_modality
        else:
            caption_modality = ["RGB"]

    if cfg.DATASETS.MULTI:
        # use every name in the tuple
        dataset = load_multi_dataset(cfg, root=cfg.DATASETS.ROOT_DIR, c_modality = caption_modality)
        # dataset = merge_datasets(cfg)
    
    else:
        # only the first name in the tuple
        first_name = cfg.DATASETS.NAMES[0]
        dataset = __factory[first_name](
            root       = cfg.DATASETS.ROOT_DIR,
            i_modality = image_modality,
            c_modality = caption_modality,
            verbose    = not cfg.DATASETS.MULTI,
        )

    # dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR)
    # dataset = __factory[cfg.DATASETS.NAMES](root=cfg.DATASETS.ROOT_DIR, i_modality=image_modality, c_modality=caption_modality)  # Pass modality here   

    train_set = ImageDataset(dataset.train, train_transforms)
    # train_set_normal = ImageDataset(dataset.train, val_transforms)
    num_classes = dataset.num_train_pids
    cam_num = dataset.num_train_cams
    view_num = dataset.num_train_vids

    if 'triplet' in cfg.DATALOADER.SAMPLER:

        if cfg.MODEL.DIST_TRAIN:
            # ─── multi-GPU triplet ────────────────────────────────────────────────
            world_size      = dist.get_world_size()
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // world_size

            data_sampler = RandomIdentitySampler_DDP(
                dataset.train,    # inherits DistributedSampler → has set_epoch()
                mini_batch_size,
                cfg.DATALOADER.NUM_INSTANCE
            )

            train_loader = DataLoader(
                train_set,
                batch_size   = mini_batch_size,
                sampler      = data_sampler,          # ← NO BatchSampler wrapper
                num_workers  = num_workers,
                pin_memory   = True,
                collate_fn   = train_collate_fn,
                drop_last    = True,
            )

        else:
            # ─── single-GPU triplet ───────────────────────────────────────────────
            train_loader = DataLoader(
                train_set,
                batch_size   = cfg.SOLVER.IMS_PER_BATCH,
                sampler      = RandomIdentitySampler(
                                   dataset.train,
                                   cfg.SOLVER.IMS_PER_BATCH,
                                   cfg.DATALOADER.NUM_INSTANCE),
                num_workers  = num_workers,
                collate_fn   = train_collate_fn,
                drop_last    = True,
            )

    elif 'softmax' in cfg.DATALOADER.SAMPLER:

        if cfg.MODEL.DIST_TRAIN:
            # ─── multi-GPU softmax (plain DistributedSampler) ────────────────────
            data_sampler = DistributedSampler(dataset.train, shuffle=True)
            mini_batch_size = cfg.SOLVER.IMS_PER_BATCH // dist.get_world_size()

            train_loader = DataLoader(
                train_set,
                batch_size   = mini_batch_size,
                sampler      = data_sampler,
                num_workers  = num_workers,
                pin_memory   = True,
                collate_fn   = train_collate_fn,
                drop_last    = True,
            )
        else:
            # ─── single-GPU softmax ──────────────────────────────────────────────
            train_loader = DataLoader(
                train_set,
                batch_size   = cfg.SOLVER.IMS_PER_BATCH,
                shuffle      = True,
                num_workers  = num_workers,
                pin_memory   = True,
                collate_fn   = train_collate_fn,
                drop_last    = True,
            )

    else:
        raise ValueError(f"Unsupported sampler {cfg.DATALOADER.SAMPLER}")

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    val_set_ttt = ImageDataset(dataset.query + dataset.gallery, val_transforms)

    val_loader_ttt = DataLoader(
        val_set_ttt, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader_ttt, val_loader, len(dataset.query), num_classes, cam_num, view_num
