"""
nomic_backbones.py
Utility wrappers that keep Nomic-Embed models fully **offline** and **frozen**.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoImageProcessor,
    AutoTokenizer,
    AutoConfig,
)
from typing import Union

# ----------------------------------------------------------------------------- #
#  Local mirrors – edit these two paths once, keep the rest of the code clean   #
# ----------------------------------------------------------------------------- #
_DEF_LOCAL = {
    "vision": "model/nomic/nomic_vision",  # <-- put real paths here
    "text": "model/nomic/nomic_text",
}


# ----------------------------------------------------------------------------- #
#  Robust loader – never hits the Internet unless you explicitly allow it       #
# ----------------------------------------------------------------------------- #
def _safe_load(ckpt: str, local_only: bool = True):
    """
    Try to load a HF model from `ckpt`           (a directory or repo-id).
    If that fails (and `local_only` is True), fall back to the local mirror defined above.
    """
    if os.path.isdir(ckpt):
        return AutoModel.from_pretrained(ckpt, local_files_only=True, trust_remote_code=True)

    # if local_only:
    #     tag = "vision" if "vision" in ckpt.lower() else "text"
    #     return AutoModel.from_pretrained(_DEF_LOCAL[tag], local_files_only=True, trust_remote_code=True)

    # final fall-back: allow remote download
    return AutoModel.from_pretrained(ckpt, trust_remote_code=True)


# ----------------------------------------------------------------------------- #
#  Frozen Nomic Vision – returns ℓ2-normalised CLS embeddings                   #
# ----------------------------------------------------------------------------- #
class FrozenNomicVision(nn.Module):
    """
    Wrapper around `nomic-ai/nomic-embed-vision-v1.5`.
    Accepts **either** pre-processed dicts (`pixel_values`, …) **or** raw
    tensors / PIL images.  All params are frozen.
    """

    # def __init__(self, ckpt: str = _DEF_LOCAL["vision"], local_only: bool = True,
    #     device: torch.device | str | None = None,):
    #     super().__init__()
    
    def __init__(self, ckpt: str, local_only: bool = True,
                device: Union[torch.device, str, None] = None):
        super().__init__()

        
        #self.backbone   = _safe_load(ckpt, local_only).eval().to(device)
        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.backbone = _safe_load(ckpt, local_only).to(dev).eval()
        self.processor  = AutoImageProcessor.from_pretrained(ckpt, local_files_only=True)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, images):
        # preprocessing --------------------------------------------------------
        if isinstance(images, dict):                      # already tokenised
            batch = {k: v.to(self.backbone.device) for k, v in images.items()}
        else: 
            if images.max() > 1.0 or images.min() < 0.0:
                images = (images - images.min()) / (images.max() - images.min())
            batch = self.processor(images, return_tensors="pt").to(self.backbone.device)                                            # raw tensor / PIL
            #batch = self.processor(images, return_tensors="pt").to(self.backbone.device)

        # model forward --------------------------------------------------------
        out = self.backbone(**batch).last_hidden_state[:, 0]   # CLS token
        return F.normalize(out, p=2, dim=-1)                  # [B, 1024]


# ----------------------------------------------------------------------------- #
#  Frozen Nomic Text – returns pooled CLS embeddings                            #
# ----------------------------------------------------------------------------- #
class FrozenNomicText(nn.Module):
    """
    Wrapper around `nomic-ai/nomic-embed-text-v1.5`.
    Pass a tokenised batch (input_ids, attention_mask, …).  All params frozen.
    """

    # def __init__(self, ckpt: str = _DEF_LOCAL["text"], local_only: bool = True,
    #     device: torch.device | str | None = None,):
    #     super().__init__()


    def __init__(self, ckpt: str, local_only: bool = True,
                device: Union[torch.device, str, None] = None):
        super().__init__()


        #self.backbone = _safe_load(ckpt, local_only).eval()
        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.backbone = _safe_load(ckpt, local_only).to(dev).eval()
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, **inputs):
        """
        Example call:
            tok = tokenizer(["search_query: …"], return_tensors='pt')
            emb = text_encoder(**tok)
        """
        #outputs = self.backbone(**inputs)
        outputs = self.backbone(**{k: v.to(self.backbone.device, non_blocking=True)
                                   for k, v in inputs.items()})
        cls_vec = outputs.last_hidden_state[:, 0]          # [B, 1024]
        return F.normalize(cls_vec, p=2, dim=-1)
