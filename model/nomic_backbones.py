import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer

# ------------------------------------------------------------------------- #
#  Frozen Nomic‑Embed encoders                                              #
# ------------------------------------------------------------------------- #

class FrozenNomicVision(nn.Module):
    """
    CLS feature from the vision branch of nomic‑embed‑vision‑v1.5
    Returned dim = 1024, matching TransReID after the adapters.
    """
    def __init__(self,
                 model_name: str = "nomic-ai/nomic-embed-vision-v1.5"):
        super().__init__()
        self.processor = AutoProcessor.from_pretrained(
            '/data_sata/ReID_Group/ReID_Group/TestTimeTraining/MMTTA-ReID-v1-Umair-2/MMTTA-ReID-4M-v1-Umair/model/nomic/nomic_vision', trust_remote_code=True)
        self.model = AutoModel.from_pretrained('/data_sata/ReID_Group/ReID_Group/TestTimeTraining/MMTTA-ReID-v1-Umair-2/MMTTA-ReID-4M-v1-Umair/model/nomic/nomic_vision', trust_remote_code=True)
        self.out_dim = self.model.config.hidden_size   # 1024
        
        self.model.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, images):
        # pixel_values expects float in [0,1] resized to 224×224. Caller handles
        # any resizing/normalisation – here we extract the CLS token.
        #out = self.model(pixel_values=images, output_hidden_states=True)
        # batch = self.processor(images=images, return_tensors="pt")\
        #             .to(images.device, images.dtype)
        if images.min() < 0:
            images = images * 0.5 + 0.5
        images = images.clamp_(0, 1)
        imgs_np = [img.permute(1, 2, 0).cpu().numpy() for img in images]
        # batch = self.processor(images=imgs_np, return_tensors="pt",do_rescale=False).to(images.device)

        # # batch = self.processor(images=images, return_tensors="pt")\
        # #             .to(images.device, images.dtype)
        # out = self.model(**batch, output_hidden_states=True)
        #  #return out.hidden_states[-1][:, 0]       # (B,1024)
        # return out.hidden_states[-1][:, 0]          # (B,1024)
        batch = self.processor(
            images=imgs_np,
            return_tensors="pt",
            do_rescale=False
        ).to(images.device)

        outputs = self.model(**batch)               # no extra kwargs

        # nomic-embed-vision returns a dict with the key `"embeds"`
        # (for safety we fall back to common field names)
        if isinstance(outputs, dict):
            if "embeds" in outputs:
                cls = outputs["embeds"]             # (B,1024)
            elif "image_embeds" in outputs:
                cls = outputs["image_embeds"]
            elif "last_hidden_state" in outputs:
                cls = outputs["last_hidden_state"][:, 0]
            else:
                raise RuntimeError("Cannot find vision embedding in model output")
        else:                                       # tuple or tensor
            cls = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

        return cls                                  # (B,1024)


class FrozenNomicText(nn.Module):
    """
    CLS feature from the text branch of nomic‑embed‑text‑v1.5
    Returned dim = 768.
    """
    def __init__(self,
                 model_name: str = "nomic-ai/nomic-embed-text-v1.5"):
        super().__init__()
        self.processor = AutoTokenizer.from_pretrained(model_name)
        self.model     = AutoModel.from_pretrained(model_name)
        self.out_dim   = self.model.config.hidden_size               # 768
        self.model.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, captions):
        """
        `captions` is a python list [str] (batch) or list [list[str]]
        Matching make_dataloader, we keep the outer list length = B.
        """
        if isinstance(captions[0], list):
            captions = [' '.join(c) for c in captions]   # flatten per‑sample list
        enc = self.processor(captions,
                             padding=True,
                             truncation=True,
                             return_tensors="pt").to(
                             next(self.model.parameters()).device)
        out = self.model(**enc, output_hidden_states=True)
        return out.last_hidden_state[:, 0]          # CLS → (B,768)
