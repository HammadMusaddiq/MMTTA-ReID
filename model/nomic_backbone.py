import torch.nn as nn, torch
from transformers import AutoModel

class FrozenNomicVision(nn.Module):
    def __init__(self, ckpt="/data_sata/ReID_Group/ReID_Group/TestTimeTraining/MMTTA-ReID-4M-v1-Umair/model/nomic/nomic_vision"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(ckpt)
        for p in self.backbone.parameters():
            p.requires_grad_(False)

    def forward(self, x):                 # x = (B,3,H,W)
        return self.backbone(x).last_hidden_state[:, 0]  # CLS

class FrozenNomicText(nn.Module):
    def __init__(self, ckpt="/data_sata/ReID_Group/ReID_Group/TestTimeTraining/MMTTA-ReID-4M-v1-Umair/model/nomic/nomic_text"):
        super().__init__()
        self.text = AutoModel.from_pretrained(ckpt)
        for p in self.text.parameters():
            p.requires_grad_(False)

    def forward(self, token_ids):         # tokenised captions
        return self.text(**token_ids).last_hidden_state[:, 0]

