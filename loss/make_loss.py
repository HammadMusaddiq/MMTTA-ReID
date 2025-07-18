import torch
import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth, LabelSmoothingCrossEntropy
from .triplet_loss import TripletLoss
from .center_loss import CenterLoss
from .multi_modal_margin_loss_new import multiModalMarginLossNew
from .multi_modal_id_margin_loss import IDMarginLossNew
from .contrastive_loss import ContrastiveLoss
from .vision_language_infonce import VisionLanguageInfoNCELoss
from .caption_adaptive_triplet import CaptionAdaptiveTripletLoss
from .distill_l2 import DistillL2Loss



def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048

    # Caption-related setup
    use_caption = cfg.CAPTION.ENABLE
    caption_strategy = cfg.CAPTION.STRATEGY
    image_modalities = cfg.IMAGE_MODALITY
    caption_modalities = image_modalities if caption_strategy == "matched" else ["RGB"]

    cap_triplet = TripletLoss(cfg.SOLVER.MARGIN)
    cap_contrastive = ContrastiveLoss(margin=0.3)
    cap_infonce = VisionLanguageInfoNCELoss(tau=0.08).to(cfg.MODEL.DEVICE)
    cap_adatri = CaptionAdaptiveTripletLoss(alpha=1.0).to(cfg.MODEL.DEVICE)
    if cfg.MODEL.DISTILL.ENABLE:
        distill = DistillL2Loss(weight=1.0).to(cfg.MODEL.DEVICE)
    else:
        distill = None
    cap_weight = cfg.CAPTION.LOSS.WEIGHT if use_caption else 0.0

    # Loss components
    criterion_m = multiModalMarginLossNew(margin=1)
    criterion_i = IDMarginLossNew(margin=0.5, dist_type='l1')
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)

    if 'triplet' in cfg.MODEL.METRIC_LOSS_TYPE:
        triplet = TripletLoss(cfg.SOLVER.MARGIN if not cfg.MODEL.NO_MARGIN else None)
        print(f"Using {'soft ' if cfg.MODEL.NO_MARGIN else ''}triplet loss with margin: {cfg.SOLVER.MARGIN}")
    else:
        raise ValueError(f"Expected triplet loss but got {cfg.MODEL.METRIC_LOSS_TYPE}")

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print(f"Label smoothing ON, num_classes: {num_classes}")

    if sampler == 'softmax':
        def loss_func(score, feat, target):
            return F.cross_entropy(score, target)

    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam, captions=None):
            # -------- ID LOSS --------
            if isinstance(score, list):
                ID_LOSS = sum(F.cross_entropy(s, target) for s in score) / len(score)
            else:
                ID_LOSS = F.cross_entropy(score, target)

            # -------- IMAGE MODALITY LOSSES --------
            if isinstance(feat, list):
                norm_feats = [F.normalize(f, p=2, dim=1) for f in feat[:4]]
                MMM_LOSS = criterion_m(norm_feats[1], norm_feats[2], norm_feats[3], target)
                IDM_LOSS = criterion_i(norm_feats[1], norm_feats[2], norm_feats[3], target)
                TRI_LOSS = 0.5 * MMM_LOSS + 0.5 * IDM_LOSS + triplet(feat[0], target)[0]
            else:
                TRI_LOSS = triplet(feat, target)[0]

            if distill is not None and isinstance(feat, list) and len(feat) > 5:
                DIST_LOSS = distill(feat[0], feat[5])
            else:
                DIST_LOSS = torch.tensor(0.0, device=target.device)

            
            # -------- CAPTION LOSS --------
            caption_loss = torch.tensor(0.0, device=target.device)
            if use_caption and captions is not None and isinstance(feat, list) and len(feat) > 4:
                cap_feats = feat[4]                             # list OR tensor
                mod_map = {"RGB": 0, "IR": 1, "TI": 2}
                valid_mods = 0

                for i, cap_mod in enumerate(caption_modalities):
                    img_idx = mod_map.get(cap_mod, -1)
                    if img_idx < 0 or img_idx >= len(feat) - 1:
                        continue

                    img_vec = feat[1 + img_idx]
                    cap_vec = cap_feats[i] if caption_strategy == "matched" and len(cap_feats) > 1 else cap_feats[0]

                    if img_vec is None or cap_vec is None:
                        continue
                    
                    cap_vec = cap_vec.to(img_vec.device)

                    # img_vec = F.normalize(img_vec, p=2, dim=1)
                    # cap_vec = F.normalize(cap_vec, p=2, dim=1)

                    if cfg.CAPTION.LOSS.TRIPLET:
                        caption_loss += cap_triplet(cap_vec, target)[0]
                    if cfg.CAPTION.LOSS.ADAPTIVE_TRIPLET:           # <-- new flag
                        caption_loss += cap_adatri(img_vec, cap_vec, target)
                    if cfg.CAPTION.LOSS.CONTRASTIVE:
                        caption_loss += cap_contrastive(img_vec, cap_vec, target)
                    if cfg.CAPTION.LOSS.INFO_NCE:
                        caption_loss += cap_infonce(img_vec, cap_vec)

                    valid_mods += 1

                if valid_mods > 0:
                    caption_loss /= valid_mods

            total_loss = (
                cfg.MODEL.ID_LOSS_WEIGHT * ID_LOSS +
                cfg.MODEL.TRIPLET_LOSS_WEIGHT * TRI_LOSS +
                cap_weight * caption_loss + 
                (cfg.MODEL.DISTILL.W * DIST_LOSS if distill is not None else 0.0))

            return total_loss

    else:
        raise ValueError(f"Unsupported sampler: {sampler}")

    return loss_func, center_criterion


def make_loss_ttt(cfg, num_classes):    # modified by gu
    sampler = cfg.DATALOADER.SAMPLER
    feat_dim = 2048
    criterion_m = multiModalMarginLossNew(margin=1)
    criterion_i = IDMarginLossNew(margin=1, dist_type='l1')
    center_criterion = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=True)  # center loss

    if sampler == 'softmax':
        def loss_func(score, feat, target, target_cam):
            # import pdb
            # pdb.set_trace()
            pseudo_label = score.max(1)[1]
            # --- Entropy minimisation ---------------------------------
            if cfg.TTA.ENTROPY_ENABLE:
                ENT_LOSS = entropy_loss(score)
            else:
                ENT_LOSS = torch.tensor(0.0, device=score.device)


            MMM_LOSS = criterion_m(F.normalize(feat[1], p=2, dim=1), F.normalize(feat[2], p=2, dim=1), F.normalize(feat[3], p=2, dim=1), pseudo_label)
            if pseudo_label[0] != pseudo_label[1]:
                IDM_LOSS = criterion_i(F.normalize(feat[1], p=2, dim=1), F.normalize(feat[2], p=2, dim=1), F.normalize(feat[3], p=2, dim=1), pseudo_label)
                TTT_LOSS = MMM_LOSS + IDM_LOSS
            else:
                TTT_LOSS = MMM_LOSS
            # return cfg.MODEL.TRIPLET_LOSS_WEIGHT * TTT_LOSS
            return (cfg.MODEL.TRIPLET_LOSS_WEIGHT * TTT_LOSS + cfg.TTA.ENTROPY_W * ENT_LOSS)



    else:
        print('expected sampler should be softmax, triplet, softmax_triplet or softmax_triplet_center'
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func, center_criterion
