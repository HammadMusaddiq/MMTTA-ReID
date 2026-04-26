import torch
import torch.nn as nn
from .backbones.resnet import ResNet, Bottleneck
import copy
from .backbones.vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID, AdapterBlock
from loss.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from model.nomic_backbones import FrozenNomicVision, FrozenNomicText
#from sentence_transformers import SentenceTransformer
# Small utility so we don’t crash when a modality is missing
def pick_feats(feat_list, idx):
    """
    Returns feat_list[idx] if it exists **and** is not None,
    otherwise returns None.
    """
    return (
        feat_list[idx]
        if (len(feat_list) > idx and feat_list[idx] is not None)
        else None
    )
def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        #nn.init.constant_(m.bias, 0.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param('/home/zi.wang/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth')
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes*3, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes*3)
        
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x1, x2, x3, label=None):  # label is unused if self.cos_layer == 'no'
        x1 = self.base(x1)
        x2 = self.base(x2)
        x3 = self.base(x3)
        global_feat1 = nn.functional.avg_pool2d(x1, x1.shape[2:4])
        global_feat1 = global_feat1.view(global_feat1.shape[0], -1)  # flatten to (bs, 2048)
        global_feat2 = nn.functional.avg_pool2d(x2, x2.shape[2:4])
        global_feat2 = global_feat2.view(global_feat2.shape[0], -1)  # flatten to (bs, 2048)
        global_feat3 = nn.functional.avg_pool2d(x3, x3.shape[2:4])
        global_feat3 = global_feat3.view(global_feat3.shape[0], -1)  # flatten to (bs, 2048)


        global_feat = torch.cat([global_feat1, global_feat2, global_feat3], dim=1)
        feat = self.bottleneck(global_feat)

        # -------------------------------------------------------------
        # DEBUG: inspect features once per step on main GPU
     
        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, [global_feat, global_feat1, global_feat2, global_feat3]
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


# model class for 3 modalities
class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, device):
        super(build_transformer, self).__init__()
        # --- Static dims ---
        self.vit_dim   = 768            # ViT backbone width
        #self.txt_dim   = 1024           # Nomic text encoder output dim
        self.in_planes = self.vit_dim   # Default, may be overridden for DeiT-small

        # --- Configs ---
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE

        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.use_caption = cfg.CAPTION.ENABLE
        self.distill_on  = cfg.MODEL.DISTILL.ENABLE
        caption_model_path = cfg.MODEL.CAPTION_MODEL_PATH
        # device = torch.device(cfg.MODEL.DEVICE)

        # --- Caption encoder and tokenizer ---
        if self.use_caption:
            self.caption_strategy = cfg.CAPTION.STRATEGY
            self.caption_modalities = cfg.IMAGE_MODALITY if self.caption_strategy == "matched" else ["RGB"]
            self.text_encoder = AutoModel.from_pretrained(f"{caption_model_path}/nomic_text/", local_files_only=True, trust_remote_code=True)
            self.text_encoder.to(device)
            self.text_encoder.eval()
            # device = next(self.text_encoder.parameters()).device   # ← new
            # self.text_encoder = self.text_encoder.to(device).eval()
            # self.text_encoder.to(torch.device(cfg.MODEL.DEVICE))  #new add 
            # self.text_encoder.eval()  #new add
            self.tokenizer = AutoTokenizer.from_pretrained(f"{caption_model_path}/nomic_text/", local_files_only=True, trust_remote_code=True)

            with torch.no_grad():
    
                text_device = next(self.text_encoder.parameters()).device
                dummy_inputs = self.tokenizer("search_document: dummy", return_tensors="pt").to(text_device)
                #dummy_tok = {k: v.to(next(self.text_encoder.parameters()).device) for k, v in dummy_tok.items()}
                cls_vec      = self.text_encoder(**dummy_inputs).last_hidden_state[:, 0]  # [1, dim]
                self.txt_dim = cls_vec.shape[-1]          # e.g. 1024

            
            # ➌  Projection into ViT width ----------------------------------------
            self.cap_proj = nn.Linear(self.txt_dim, self.in_planes, bias=False)
            self.cap_proj.apply(weights_init_kaiming)
        
        else:
            self.text_encoder = self.tokenizer = None
            self.txt_dim      = 0
            self.cap_proj     = None
            # self.caption_strategy = None
            # self.caption_modalities = None
            # self.text_encoder = None
            # self.tokenizer = None

        # --- Teacher vision encoder ---
        self.teacher_vis = (
            FrozenNomicVision(
                ckpt=f"{caption_model_path}/nomic_vision/", device=device,
            ).to(device).eval() if self.distill_on else None
        )

        if self.teacher_vis is not None:
            self.teacher_vis.requires_grad_(False)
        
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        # --- Camera/view logic ---
        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        # --- Backbone ---
        num_mod_types = len(cfg.IMAGE_MODALITY)
        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](
            img_size=cfg.INPUT.SIZE_TRAIN,
            sie_xishu=cfg.MODEL.SIE_COE,
            camera=camera_num,
            view=view_num,
            stride_size=cfg.MODEL.STRIDE_SIZE,
            num_mod_types=num_mod_types,  
            drop_path_rate=cfg.MODEL.DROP_PATH,
            drop_rate=cfg.MODEL.DROP_OUT,
            attn_drop_rate=cfg.MODEL.ATT_DROP_RATE
        )

        # --- Override for DeiT-small ---
        # if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
        #     self.in_planes = 384
        self.in_planes = self.base.embed_dim_out

         # -------- number of modalities finally concatenated ------------
        self.mod_concat  = 3 + (1 if self.use_caption else 0)   # RGB,IR,TI,(txt)
        self.concat_dim  = self.in_planes * self.mod_concat

        # ---------- projections that depend on final in_planes ----------
        if self.use_caption:
            self.cap_proj = nn.Linear(self.txt_dim, self.in_planes, bias=False)
            self.cap_proj.apply(weights_init_kaiming)
        else:
            self.cap_proj = None

        # if self.distill_on:
        #     self.teacher_proj = nn.Linear(self.txt_dim, self.in_planes, bias=False)
        #     self.teacher_proj.apply(weights_init_kaiming)
        # else:
        #     self.teacher_proj = None
        if self.distill_on:
            # Dynamically determine teacher output dim
            # with torch.no_grad():
            #     dummy = torch.randn(1, 3, 224, 224)
            #     vis_dim = self.teacher_vis(dummy).shape[-1]
            # self.teacher_proj = nn.Linear(vis_dim, self.in_planes, bias=False)

            self.teacher_proj = nn.Linear(self.teacher_vis.out_dim,       # =1024
+                self.in_planes, bias=False)
            self.teacher_proj.apply(weights_init_kaiming)
        else:
            self.teacher_proj = None
        # ----------------------------------------------------------------

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path, test_time_training=True)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        # self.bottleneck = nn.BatchNorm1d(self.in_planes * 3)
        self.bottleneck = nn.BatchNorm1d(self.concat_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        # --- Metric-learning head (classifier) ---
        # head_in = self.in_planes * 3
        head_in = self.concat_dim
        if self.ID_LOSS_TYPE == 'arcface':
            self.classifier = Arcface(head_in, self.num_classes,
                                    s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            self.classifier = Cosface(head_in, self.num_classes,
                                    s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            self.classifier = AMSoftmax(head_in, self.num_classes,
                                    s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            self.classifier = CircleLoss(head_in, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(head_in, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

      

    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """Constructs fully connected layer
        Args:
            fc_dims (list or tuple): dimensions of fc layers, if None, no fc layers are constructed
            input_dim (int): input dimension
            dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None

        assert isinstance(
            fc_dims, (list, tuple)
        ), 'fc_dims must be either list or tuple, but got {}'.format(
            type(fc_dims)
        )

        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim

        self.feature_dim = fc_dims[-1]

        return nn.Sequential(*layers)


    def forward(self, x1, x2, x3, label=None, cam_label= None, view_label=None, captions=None):
        # ----------------------- student visual ------------------------
        global_feat1 = self.base(x1, 0, cam_label=cam_label, view_label=view_label)  # RGB
        global_feat2 = self.base(x2, 1, cam_label=cam_label, view_label=view_label)  # IR
        global_feat3 = self.base(x3, 2, cam_label=cam_label, view_label=view_label)  # TI
        global_feat1 = F.normalize(global_feat1, p=2, dim=1)
        global_feat2 = F.normalize(global_feat2, p=2, dim=1)
        global_feat3 = F.normalize(global_feat3, p=2, dim=1)
        text_feats = None

        # if text_feats is None:
        #     global_feat = torch.cat((global_feat1, global_feat2, global_feat3,text_feats),dim=1) 
        # else:
        #     global_feat = torch.cat((global_feat1, global_feat2, global_feat3),dim=1)
                                 

        # ---------- CAPTION STREAM ----------
        text_feats = None
        if self.use_caption and captions is not None:
            # ①  prefix for Nomic Text
            prefixed = [f"search_document: {c}" for c in captions] \
                    if isinstance(captions, (list, tuple)) else [f"search_document: {captions}"]

            # ②  tokenise & encode (no grad – frozen)
            with torch.no_grad():
                inputs = self.tokenizer(prefixed, padding=True, truncation=True,
                                        return_tensors="pt").to(x1.device)
                cls_vec = self.text_encoder(**inputs).last_hidden_state[:, 0]

            # ③  project to ViT width if required
            text_feats = self.cap_proj(cls_vec) if self.cap_proj else cls_vec          # (B,D)

        # ---------- PACK RETURN LIST ----------
        # feat_list = [global_feat, global_feat1, global_feat2, global_feat3, text_feats]

        # # teacher CLS (optional, may be None)
        # if self.teacher_vis:
        #     with torch.no_grad():
        #         t_raw = self.teacher_vis(x1)                    # (B,1024 or 768 ...)
        #     feat_list.append(self.teacher_proj(t_raw) if self.teacher_proj else t_raw)

        # training ─ return score + list /  eval ─ return feats

        # ---------- CAPTION STREAM ----------
        

     
#         # --- CAPTION STREAM -------------------------------------------------
#         text_feats = None  # Ensure always defined
#         if self.use_caption and captions is not None:
#             # 1) task prefix – choose one; here we embed captions as *documents*
#             prefixed = [f"search_document: {c}" for c in captions] if isinstance(captions, (list, tuple)) else [f"search_document: {captions}"]

#             # 2) tokenise
#             inputs = self.tokenizer(prefixed, padding=True, truncation=True,
#                             return_tensors="pt").to(x1.device)

#                     # 3) CLS pooling (official recipe)
#             with torch.no_grad():
#                 out = self.text_encoder(**inputs).last_hidden_state[:, 0]    # [B, 768]

#             if self.cap_proj is not None:           # project into ViT dim (768 / 384)
#                 out = self.cap_proj(out)

#             # 4) replicate across caption-modalities (L = len(self.caption_modalities))
#             text_feats = out.unsqueeze(0).repeat(len(self.caption_modalities), 1, 1)  # [L,B,D]
  

     
#         else:
#             text_feats = None
# # --------------------------------------------------------------------
#         if text_feats is not None and text_feats.size(0) == 1:    # shape (1,B,D) → (B,D)
#             text_feats = text_feats.squeeze(0)

#         feat_list = [global_feat, global_feat1, global_feat2, global_feat3, text_feats]

#         if self.teacher_vis is not None:
#             with torch.no_grad():
#                 t_raw = self.teacher_vis(x1)
#             feat_list.append(self.teacher_proj(t_raw))            # (B, in_planes)

   

       

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

   
            feat_list = [
            global_feat1,      # RGB stream        – shape [B, in_planes]
            global_feat2,      # IR  stream
            global_feat3,      # TI  stream
            text_feats  ]       # caption stream    – None or [L,B,in_planes]]

            if self.teacher_vis is not None:
                with torch.no_grad():
                    t_raw = self.teacher_vis(x1).detach()          # teacher on RGB only
                t_cls = self.teacher_proj(t_raw) if self.teacher_proj else t_raw
                feat_list.append(t_cls)                   # teacher RGB
            return cls_score, feat_list

        # ---------- PACK RETURN LIST ----------
        #feat_list = [global_feat, global_feat1, global_feat2, global_feat3, text_feats]

        # # teacher CLS (optional, may be None)
        # if self.teacher_vis:
        #     with torch.no_grad():
        #         t_raw = self.teacher_vis(x1)                    # (B,1024 or 768 ...)
        #     feat_list.append(self.teacher_proj(t_raw) if self.teacher_proj else t_raw)


        # if self.training:
        #     if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
        #         cls_score = self.classifier(feat, label)
        #     else:
        #         cls_score = self.classifier(feat)

   
        #     feat_list = [
        #     global_feat1,      # RGB stream        – shape [B, in_planes]
        #     global_feat2,      # IR  stream
        #     global_feat3,      # TI  stream
        #     text_feats  ]       # caption stream    – None or [L,B,in_planes]]

        #     if self.teacher_vis is not None:
        #         with torch.no_grad():
        #             t_raw = self.teacher_vis(x1)          # teacher on RGB only
        #         t_cls = self.teacher_proj(t_raw) if self.teacher_proj else t_raw
        #         feat_list.append(t_cls)                   # teacher RGB
        #     return cls_score, feat_list

        

        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID
}

def make_model(cfg, num_class, camera_num, view_num, train_mode, fuse_strategy, device):


    if cfg.MODEL.NAME == 'transformer':
        if cfg.MODEL.JPM:
            # model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=cfg.MODEL.RE_ARRANGE)
            print('===========building transformer with JPM module ===========')
        else:
            model = build_transformer(num_class, camera_num, view_num, cfg, __factory_T_type, device)
            print('===========building transformer===========')
    else:
        model = Backbone(num_class, cfg)
        print('===========building ResNet===========')
    return model