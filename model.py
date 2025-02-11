from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d


class VoxelDecoder(nn.Module):
    """
    Decoder for voxel prediction.

    Parameters:
    -----
        in_dim: int, input dimension of the encoder (512 in our case)

    Returns
    -----
        out: 3D voxel grid with shape (B, D, H, W)
        shape: b x 32 x 32 x 32
    """

    def __init__(self, in_dim=512):
        super(VoxelDecoder, self).__init__()
        self.in_dim = in_dim

        # Project the input to higher dimension
        self.fc1 = nn.Linear(in_dim, 2048)

        # Transpose concolution for spatial decoding
        # Note that the output shape of a single convTranspose3d is (D_in - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc1(x)
        # resize into a 5D tensor with shape (B, C, D, H, W)
        x = x.view(x.shape[0], 256, 2, 2, 2)
        x = self.upsample(x)  # shape: b x 1 x 32 x 32 x 32
        return x


class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            self.decoder = VoxelDecoder(in_dim=512).to(self.device)
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3
            self.n_point = args.n_points
            # TODO:
            # self.decoder =
        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3
            # try different mesh initializations
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(
                mesh_pred.verts_list() * args.batch_size,
                mesh_pred.faces_list() * args.batch_size,
            )
            # TODO:
            # self.decoder =

    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0, 3, 1, 2))
            encoded_feat = (
                self.encoder(images_normalize).squeeze(-1).squeeze(-1)
            )  # b x 512
        else:
            encoded_feat = images  # in case of args.load_feat input images are pretrained resnet18 features of b x 512 size

        # call decoder
        if args.type == "vox":
            # TODO:
            # voxels_pred =
            return voxels_pred

        elif args.type == "point":
            # TODO:
            # pointclouds_pred =
            return pointclouds_pred

        elif args.type == "mesh":
            # TODO:
            # deform_vertices_pred =
            mesh_pred = self.mesh_pred.offset_verts(
                deform_vertices_pred.reshape([-1, 3])
            )
            return mesh_pred
