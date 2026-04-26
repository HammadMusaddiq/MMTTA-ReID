from utils.logger import setup_logger
from datasets import make_dataloader
from model import make_model
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler
from loss import make_loss
from processor import do_train
import random
import torch
import numpy as np
import os
import argparse
from config import cfg
import torch.distributed as dist

torch.cuda.empty_cache()

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("--dist", action="store_true")
    parser.add_argument(
        "--local_rank", "--local-rank",        # torchrun passes --local-rank
        type=int,
        default=int(os.environ.get("LOCAL_RANK", 0))
    )
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # ---------- cfg ----------
    cfg.merge_from_file(args.config_file)
    cfg.defrost()
    cfg.MODEL.DIST_TRAIN = args.dist
    cfg.freeze()

    set_seed(cfg.SOLVER.SEED)

    # ---------- distributed init ----------
    if cfg.MODEL.DIST_TRAIN:                 # multi-GPU branch
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = args.local_rank
    else:
        local_rank = 0                           # single-GPU

    # rank = dist.get_rank() if dist.is_initialized() else 0

    rank = dist.get_rank() if dist.is_initialized() else 0

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=True, rank=rank)
    # logger = setup_logger("transreid", output_dir, if_train=True)
    logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    device = torch.device("cuda", local_rank)

    train_mode = cfg.MODEL.TRAIN_MODE
    fuse_strategy = cfg.MODEL.FUSE_STRATEGY

    image_modality = cfg.IMAGE_MODALITY
    caption_enabled = cfg.CAPTION.ENABLE
    caption_strategy = cfg.CAPTION.STRATEGY if caption_enabled else "N/A"
    caption_modalities = image_modality if caption_enabled and caption_strategy == "matched" else ["RGB"] if caption_enabled else []

    logger.info("Image Modalities: {}".format(image_modality))
    logger.info("Caption Enabled: {}".format(caption_enabled))
    logger.info("Caption Strategy: {}".format(caption_strategy))
    logger.info("Caption Modalities: {}".format(caption_modalities))


    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    
    device = torch.device("cuda") if args.local_rank < 0 else torch.device("cuda", args.local_rank)
    
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num, train_mode=train_mode, fuse_strategy=fuse_strategy)
    model.to(device)

    if cfg.MODEL.DIST_TRAIN:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],     # index inside CUDA_VISIBLE_DEVICES
            output_device=local_rank,
            find_unused_parameters=True)
    
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    scheduler = create_scheduler(cfg, optimizer)

    do_train(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_func,
        num_query, local_rank
    )


# CUDA_VISIBLE_DEVICES=2 python train.py --config_file configs/RGBNT201/vit_base.yml
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file configs/MSMT/vit_base.yml
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file configs/Market1501/vit_base.yml
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file configs/Cuhk03/vit_base.yml
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file configs/PRCC/vit_base.yml
# CUDA_VISIBLE_DEVICES=1 python train.py --config_file configs/CelebReID/vit_base.yml


# CUDA_VISIBLE_DEVICES=1 python train.py --config_file configs/Market1501/vit_base_2M.yml
# CUDA_VISIBLE_DEVICES=2 python train.py --config_file configs/Market1501/vit_base_3M.yml

# CUDA_VISIBLE_DEVICES=0,2 torchrun --standalone --nproc_per_node=2 train.py --config_file configs/Market1501/vit_base_3M.yml --dist


