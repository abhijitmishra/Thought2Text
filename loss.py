import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, E1, E2, labels):
        # Compute the Euclidean distance between the embeddings
        euclidean_distance = F.pairwise_distance(E1, E2)

        # Create a binary label matrix indicating if pairs are from the same class
        # Since we only have one set of labels, compare each pair of embeddings with itself
        label = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()

        # We want similar labels to be labeled as 0 and dissimilar labels labelled as 1
        label = 1 - label

        # Compute the contrastive loss
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + label
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, E1, E2, labels=None):
        # Ensure that both embeddings have the same shape
        assert E1.shape == E2.shape, f"Embeddings must have the same shape "

        # Compute the mean squared error loss
        loss = nn.functional.mse_loss(E1, E2)
        return loss
