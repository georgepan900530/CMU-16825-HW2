import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_points, knn_gather
from pytorch3d.loss import mesh_laplacian_smoothing


# define losses
def voxel_loss(voxel_src, voxel_tgt):
    # voxel_src: b x h x w x d
    # voxel_tgt: b x h x w x d
    # loss = Binary Cross Entropy
    # implement some loss for binary voxel grids
    assert voxel_src.shape == voxel_tgt.shape
    loss = F.binary_cross_entropy_with_logits(voxel_src, voxel_tgt)
    return loss


def chamfer_loss(point_cloud_src, point_cloud_tgt):
    # point_cloud_src, point_cloud_src: b x n_points x 3
    # loss_chamfer = mean(min(d(x_i, y_j)) + mean(min(d(y_j, x_i)))
    # implement chamfer loss from scratch
    assert point_cloud_src.shape == point_cloud_tgt.shape

    # For each point in src, find the closest points in tgt and compute the distance
    # knn_points returns the distances and indices of the top k nearest neighbors for each point in src
    knn_src = knn_points(point_cloud_src, point_cloud_tgt, K=1)
    knn_src_dist = knn_src.dists[..., 0]  # (b, n_points)

    # Do the same for the target
    knn_tgt = knn_points(point_cloud_tgt, point_cloud_src, K=1)
    knn_tgt_dist = knn_tgt.dists[..., 0]  # (b, n_points)

    # Compute the loss
    loss_chamfer = knn_src_dist.mean(dim=1) + knn_tgt_dist.mean(dim=1)
    loss_chamfer = loss_chamfer.mean()  # mean over batch
    return loss_chamfer


def smoothness_loss(mesh_src):
    # loss_laplacian =
    # implement laplacian smoothening loss
    loss_laplacian = mesh_laplacian_smoothing(mesh_src)
    return loss_laplacian
