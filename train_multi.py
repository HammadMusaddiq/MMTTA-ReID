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
import torch.distributed as dist
import argparse
# from timm.scheduler import create_scheduler
from config import cfg

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

if __name__ == '__main__':
    
    # ---------------------------------------------
    # 0.  Argument parsing & config
    # ---------------------------------------------
    parser = argparse.ArgumentParser("ReID Baseline Training")
    parser.add_argument("--config_file", default="", type=str)
    parser.add_argument("opts", nargs=argparse.REMAINDER)
    # --- REMOVE the manual --local_rank flag ---
    args = parser.parse_args()

    # Merge cfg ---------------------------------------------------------------
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    set_seed(cfg.SOLVER.SEED)

    # ---------------------------------------------
    # 1.  Distributed set-up  (torchrun does the env vars)
    # ---------------------------------------------
    # torchrun exports LOCAL_RANK, RANK, WORLD_SIZE
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if cfg.MODEL.DIST_TRAIN and world_size > 1:
        torch.cuda.set_device(local_rank)              # ← crucial
        dist.init_process_group(backend="nccl", init_method="env://")

    device = torch.device("cuda", local_rank)          # single source-of-truth

    # ---------------------------------------------
    # 2.  Logging / output dir  (only rank-0 writes)
    # ---------------------------------------------
    if dist.is_initialized():
        is_master = dist.get_rank() == 0
    else:
        is_master = True

    if is_master:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    logger = setup_logger("transreid", cfg.OUTPUT_DIR, if_train=True)#, rank=local_rank
    if is_master:
        logger.info(f"Running with config:\n{cfg}")

    # ---------------------------------------------
    # 3.  Data
    # ---------------------------------------------
    train_loader, train_loader_normal, val_loader, num_query, \
        num_classes, camera_num, view_num = make_dataloader(cfg)

    # ---------------------------------------------
    # 4.  Model / loss / optimiser
    # ---------------------------------------------
    train_mode   = cfg.MODEL.TRAIN_MODE
    fuse_strategy = cfg.MODEL.FUSE_STRATEGY

    model = make_model(cfg, num_classes, camera_num, view_num,
                       train_mode=train_mode, fuse_strategy=fuse_strategy)
    model.to(device)

    if cfg.MODEL.DIST_TRAIN and world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            find_unused_parameters=True)

    loss_func, center_criterion = make_loss(cfg, num_classes)
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)
    scheduler = create_scheduler(cfg, optimizer)

    # ---------------------------------------------
    # 5.  Train
    # ---------------------------------------------
    do_train(cfg, model, center_criterion,
             train_loader, val_loader,
             optimizer, optimizer_center, scheduler,
             loss_func, num_query, local_rank)

    # # (optional) barrier for clean shutdown
    # if dist.is_initialized():
    #     dist.barrier()
    #     dist.destroy_process_group()


    # parser = argparse.ArgumentParser(description="ReID Baseline Training")
    # parser.add_argument(
    #     "--config_file", default="", help="path to config file", type=str
    # )

    # parser.add_argument("opts", help="Modify config options using the command-line", default=None,
    #                     nargs=argparse.REMAINDER)
    # #parser.add_argument("--local_rank", default=0, type=int)
    # parser.add_argument("--local_rank", "--local-rank", default=0, type=int)
    # args = parser.parse_args()

    # if args.config_file != "":
    #     cfg.merge_from_file(args.config_file)
    # cfg.merge_from_list(args.opts)
    # cfg.freeze()

    # set_seed(cfg.SOLVER.SEED)

    # if cfg.MODEL.DIST_TRAIN:
    #     torch.cuda.set_device(args.local_rank)

    # output_dir = cfg.OUTPUT_DIR
    # if output_dir and not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    # logger = setup_logger("transreid", output_dir, if_train=True)
    # logger.info("Saving model in the path :{}".format(cfg.OUTPUT_DIR))
    # logger.info(args)

    # if args.config_file != "":
    #     logger.info("Loaded configuration file {}".format(args.config_file))
    #     with open(args.config_file, 'r') as cf:
    #         config_str = "\n" + cf.read()
    #         logger.info(config_str)
    # logger.info("Running with config:\n{}".format(cfg))

    # if cfg.MODEL.DIST_TRAIN:
    #     torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    # train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # train_mode = cfg.MODEL.TRAIN_MODE

    # fuse_strategy = cfg.MODEL.FUSE_STRATEGY

    # model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num, train_mode=train_mode, fuse_strategy=fuse_strategy)

    # loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)

    # optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)

    # scheduler = create_scheduler(cfg, optimizer)



    # do_train(
    #     cfg,
    #     model,
    #     center_criterion,
    #     train_loader,
    #     val_loader,
    #     optimizer,
    #     optimizer_center,
    #     scheduler,
    #     loss_func,
    #     num_query, args.local_rank
    # )
