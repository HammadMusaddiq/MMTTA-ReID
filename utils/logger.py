import logging
import os
import sys
import os.path as osp
# def setup_logger(name, save_dir, if_train):
#     logger = logging.getLogger(name)
#     logger.setLevel(logging.DEBUG)

#     ch = logging.StreamHandler(stream=sys.stdout)
#     ch.setLevel(logging.DEBUG)
#     formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
#     ch.setFormatter(formatter)
#     logger.addHandler(ch)

#     if save_dir:
#         if not osp.exists(save_dir):
#             os.makedirs(save_dir)
#         if if_train:
#             fh = logging.FileHandler(os.path.join(save_dir, "train_log.txt"), mode='w')
#         else:
#             fh = logging.FileHandler(os.path.join(save_dir, "test_log.txt"), mode='w')
#         fh.setLevel(logging.DEBUG)
#         fh.setFormatter(formatter)
#         logger.addHandler(fh)

#     return logger

def setup_logger(name, save_dir, if_train=True, *, rank=0):
    """
    rank == 0 → console  + common log-file
    rank >  0 → (silent console) +- no common file
                (optionally write your own rank-specific file)
    """
    logger = logging.getLogger(name)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")

    # ── console ──────────────────────────────────────────────
    if rank == 0:                                # only once
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(logging.DEBUG)
        sh.setFormatter(fmt)
        logger.addHandler(sh)

    # ── file ─────────────────────────────────────────────────
    if save_dir and rank == 0:                   # <── change here
        os.makedirs(save_dir, exist_ok=True)
        fname = "train_log.txt" if if_train else "test_log.txt"
        fh = logging.FileHandler(osp.join(save_dir, fname), mode="w")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # # optional: per-rank file
    # elif save_dir:                              # ranks > 0
    #     os.makedirs(save_dir, exist_ok=True)
    #     fh = logging.FileHandler(
    #         osp.join(save_dir, f"train_log_rank{rank}.txt"), mode="w")
    #     fh.setLevel(logging.DEBUG)
    #     fh.setFormatter(fmt)
    #     logger.addHandler(fh)

    return logger