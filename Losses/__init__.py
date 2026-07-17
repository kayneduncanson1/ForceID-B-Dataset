import torch
import torch.nn as nn


def get_dists(embeddings, labels, phase):

    n = embeddings.size(0)

    # Get pair-wise L2 distance matrix:
    dist = torch.norm(embeddings[:, None] - embeddings, dim=2, p=2)

    # Define masks for valid positives and negatives:
    mask_pos = labels.expand(n, n).eq(labels.expand(n, n).t())
    mask_neg = ~mask_pos
    mask_pos[torch.eye(n).bool()] = 0

    # Initialise new matrix with same shape as dist containing -infinity:
    ninf = torch.ones_like(dist) * float('-inf')

    # Apply positive mask:
    dist_ap = torch.where(mask_pos, dist, ninf)

    # Apply negative mask:
    if phase == 'train':

        # Use -dist so that the max operation can be used later (line 68) to return the minimum anchor-negative
        # distances (representing 'hardest' anchor-negative pairs) for batch hard triplet margin loss:
        dist_an = torch.where(mask_neg, -dist, ninf)

    else:

        dist_an = torch.where(mask_neg, dist, ninf)

    return dist_ap, dist_an


# Get loss and accuracy on a balanced val/test set based on predictions generated using the difficult random subset
# search method (hence the 'Hard' in the class name 'EvalHard'):
class EvalHard(nn.Module):

    def __init__(self, margin):

        super(EvalHard, self).__init__()
        self.margin = margin

    def forward(self, dist_ap, dist_an, count_samples_min):

        # Given that the number of positives for each anchor is one fewer than the number of negatives per negative ID,
        # need to exclude the last (random) negative from each negative ID:
        dist_an_clipped = torch.stack(torch.split(dist_an, int(count_samples_min), dim=1))[:, :, :-1]\
            .transpose(1, 0).reshape(dist_an.size(0), -1)

        # Get indices of valid anchor-positive and anchor-negative distances:
        indices_dist_ap_valid = torch.nonzero(dist_ap != float('-inf'), as_tuple=False)
        indices_dist_an_valid = torch.nonzero(dist_an_clipped != float('-inf'), as_tuple=False)

        # Get valid anchor-positive and anchor-negative distances:
        dist_ap_valid = torch.stack(torch.split(dist_ap[indices_dist_ap_valid[:, 0],
                                                        indices_dist_ap_valid[:, 1]], int(count_samples_min - 1)))
        dist_an_valid = torch.stack(torch.split(dist_an_clipped[indices_dist_an_valid[:, 0],
                                                                indices_dist_an_valid[:, 1]],
                                                int(dist_an_clipped.size(1) - (count_samples_min - 1))))

        # Get minimum anchor-negative distances:
        dist_an_min = torch.min(torch.stack(torch.split(dist_an_valid, int(count_samples_min - 1), dim=1)),
                                dim=0, keepdim=False)[0]

        # Clamp loss values at zero and then get the mean loss over the random subset:
        loss = torch.clamp(dist_ap_valid - dist_an_min + self.margin, min=0.).mean()

        with torch.no_grad():

            # Accuracy is based on whether the closest (i.e., most similar) sample from the random subset is a
            # positive:
            acc = (dist_an_min > dist_ap_valid).float().mean().item()

        return loss, acc
