import torch
import torch.nn.functional as F
import torch.nn as nn

# class ContrastiveLoss(nn.Module):
#     def __init__(self, margin=0.3):
#         super(ContrastiveLoss, self).__init__()
#         self.margin = margin
#         self.cos_sim = nn.CosineSimilarity(dim=1)

#     def forward(self, img_feat, cap_feat, label):
#         sim = self.cos_sim(img_feat, cap_feat)  # [B]
#         pos = sim[label == label]  # same class
#         neg = sim[label.unsqueeze(1) != label.unsqueeze(0)]  # different class
#         loss = torch.clamp(self.margin - pos.mean() + neg.mean(), min=0.0)
#         return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, img_feat, cap_feat, labels):
        # img_feat, cap_feat: [B, D]
        # labels: [B]
        B = img_feat.size(0)

        # Normalize features
        img_feat = F.normalize(img_feat, p=2, dim=1)
        cap_feat = F.normalize(cap_feat, p=2, dim=1)

        # Compute similarity matrix between all image-caption pairs
        sim_matrix = torch.matmul(img_feat, cap_feat.T)  # [B, B]

        # Create label match matrix
        labels = labels.view(-1, 1)
        matches = (labels == labels.T).float()  # [B, B]
        diffs = 1 - matches

        # Positive loss: encourage similarity for same identity pairs
        pos_loss = (1 - sim_matrix) * matches  # lower sim → higher loss

        # Negative loss: penalize high similarity for different identity pairs
        neg_loss = F.relu(sim_matrix - self.margin) * diffs  # sim > margin → penalized

        # Combine
        total_loss = (pos_loss.sum() + neg_loss.sum()) / (B * B)
        return total_loss