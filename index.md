# CMU 16825 Assignment 2 - George Pan (fuchengp)

## Q1 Exploring Loss Functions

### Q1.1 Fitting a voxel grid

In this problem, we are asked to implement binary cross-entropy loss for voxel fitting. Below are the visualizations of the optimized voxel grid and the ground truth voxel grid.

**Optimized voxel grid**
![q1-1-src](results/q1/q1-1_src.gif)
**Ground truth voxel grid**
![q1-1-gt](results/q1/q1-1_tgt.gif)

As we can see, fitting a random voxel grid into the target voxel grid with binary cross-entropy loss results in quite impressive result.

### Q1.2 Fitting a point cloud

In this problem, we aimed to fit a point loud to the target using chamfer loss which measure the distance between two point clouds by computing the average closest point distance between them in both directions. Below are the visualizations of the optimized point cloud and the ground truth point cloud.

**Optimized point cloud**
![q1-2-src](results/q1/q1-2_src.gif)
**Ground truth point cloud**
![q1-2-gt](results/q1/q1-2_tgt.gif)

Similar to the voxel case, the source point cloud can fit to the target point cloud well with the use of chamfer loss function.

### Q1.3 Fitting a mesh

In this part, we defined an additional smoothness loss by minimizing the Laplacian of the mesh. Below are the visualizations of the resulting mesh and the ground truth mesh.

**Optimized mesh**
![q1-3-src](results/q1/q1-3_src.gif)
**Ground truth mesh**
![q1-3-gt](results/q1/q1-3_tgt.gif)

Similarly, the smootheness loss allowed the source mesh to fit the ground truth mesh much smoother and better.

## Q2 Reconstructing 3D from single view

### Q2.1 Image to voxel grid
In this part, we need to construct a deep learning model to predict the occupancy of voxel grid. Specifically, I built my model based on [Pix2Vox](https://github.com/hzxie/Pix2Vox/blob/master/models/decoder.py). Below are the visualizations of the input image, predicted voxel grid, ground truth voxel grid, and ground truth mesh.

#### Samples

| **Description** | **Sample 0** | **Sample20** | **Sample 40** |
| -------------- | ------------------------ | ------------------------ | ------------------------ |
| **Input RGB** | ![q2-1-img](results/q2/vox_small/q2_vox_rgb_0.png) | ![q2-1-img2](results/q2/vox_small/q2_vox_rgb_20.png) | ![q2-1-img3](results/q2/vox_small/q2_vox_rgb_40.png) |
| **Voxel grid prediction** | ![q2-1-pred](results/q2/vox_small/q2_vox_pred_0.gif) | ![q2-1-pred2](results/q2/vox_small/q2_vox_pred_20.gif) | ![q2-1-pred3](results/q2/vox_small/q2_vox_pred_40.gif) |
| **Voxel grid ground truth** | ![q2-1-gt](results/q2/vox_small/q2_vox_gt_0.gif)         | ![q2-1-gt2](results/q2/vox_small/q2_vox_gt_20.gif) | ![q2-1-gt3](results/q2/vox_small/q2_vox_gt_40.gif) |
| **Mesh ground truth** | ![q2-1-gt-mesh](results/q2/vox_small/q2_mesh_gt_0.gif)       | ![q2-1-gt-mesh2](results/q2/vox_small/q2_mesh_gt_20.gif) | ![q2-1-gt-mesh3](results/q2/vox_small/q2_mesh_gt_40.gif) |

As we can see from the above table, the predictions of voxel grids are not as well as expected. This can also be indentify when observing the training loss (Binar Cross-Entropy Loss) which stucked at around **0.1**. I have tried smaller and larger model. However, the performance of each model are similar.