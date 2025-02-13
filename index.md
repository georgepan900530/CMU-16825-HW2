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

In this part, we need to construct a deep learning model to predict the occupancy of voxel grid. Specifically, I built my model based on [Pix2Vox](https://github.com/hzxie/Pix2Vox/blob/master/models/decoder.py). Below are the visualizations of the input image, predicted voxel grid, ground truth voxel grid, and the ground truth mesh.

#### Visualizations

| **Description**             | **Sample 0**                                           | **Sample249**                                             | **Sample 638**                                            |
| --------------------------- | ------------------------------------------------------ | --------------------------------------------------------- | --------------------------------------------------------- |
| **Input RGB**               | ![q2-1-img](results/q2/vox_final/q2_vox_rgb_0.png)     | ![q2-1-img2](results/q2/vox_final/q2_vox_rgb_249.png)     | ![q2-1-img3](results/q2/vox_final/q2_vox_rgb_638.png)     |
| **Voxel grid prediction**   | ![q2-1-pred](results/q2/vox_final/q2_vox_pred_0.gif)   | ![q2-1-pred2](results/q2/vox_final/q2_vox_pred_249.gif)   | ![q2-1-pred3](results/q2/vox_final/q2_vox_pred_638.gif)   |
| **Voxel grid ground truth** | ![q2-1-gt](results/q2/vox_final/q2_vox_gt_0.gif)       | ![q2-1-gt2](results/q2/vox_final/q2_vox_gt_249.gif)       | ![q2-1-gt3](results/q2/vox_final/q2_vox_gt_638.gif)       |
| **Mesh ground truth**       | ![q2-1-gt-mesh](results/q2/vox_final/q2_mesh_gt_0.gif) | ![q2-1-gt-mesh2](results/q2/vox_final/q2_mesh_gt_249.gif) | ![q2-1-gt-mesh3](results/q2/vox_final/q2_mesh_gt_638.gif) |

As we can see from the above table, the predictions of voxel grids are not as well as expected. This can also be indentify when observing the training loss (Binar Cross-Entropy Loss) which stucked at around **0.1**. I have tried smaller and larger model. However, the performance of each model are similar.

### Q2.2 Image to point cloud

In this section, we aim to train a model to predict the coordinates of a point cloud. Note that I have tested with different number of points since the point cloud will be extremely sparse if using insuffiecient points. Detail comparison will be shown in the following section. The following shows the visualizations (10000 points) of the input image, predicted point cloud, ground truth point cloud, and the ground truth mesh.

#### Visualizations

| **Description**              | **Sample 0**                                             | **Sample 150**                                              | **Sample 450**                                              |
| ---------------------------- | -------------------------------------------------------- | ----------------------------------------------------------- | ----------------------------------------------------------- |
| **Input RGB**                | ![q2-2-img](results/q2/point_10000/q2_point_rgb_0.png)   | ![q2-2-img2](results/q2/point_10000/q2_point_rgb_150.png)   | ![q2-2-img3](results/q2/point_10000/q2_point_rgb_450.png)   |
| **Point cloud prediction**   | ![q2-2-pred](results/q2/point_10000/q2_point_pred_0.gif) | ![q2-2-pred2](results/q2/point_10000/q2_point_pred_150.gif) | ![q2-2-pred3](results/q2/point_10000/q2_point_pred_450.gif) |
| **Point cloud ground truth** | ![q2-2-gt](results/q2/point_10000/q2_point_gt_0.gif)     | ![q2-2-gt2](results/q2/point_10000/q2_point_gt_150.gif)     | ![q2-2-gt3](results/q2/point_10000/q2_point_gt_450.gif)     |
| **Mesh ground truth**        | ![q2-2-gt-mesh](results/q2/point_10000/q2_mesh_gt_0.gif) | ![q2-2-gt-mesh2](results/q2/point_10000/q2_mesh_gt_150.gif) | ![q2-1-gt-mesh3](results/q2/point_10000/q2_mesh_gt_450.gif) |

In my opinion, the point cloud reconstruction performed better than that of voxel as we can see a more aligned shapes between the prediction and the ground truth. However, the point clouds are still fairly sparse compared to the ground truth. Therefore, it is reasonable to test different number of points when training point clouds.

### Q2.3 Image to Mesh

In this section, we aimed to reconstruct 3D meshes from a single image. The following shows the visualizations of the input image, predicted meshe, and the ground truth mesh.

#### Visualizations

| **Description**       | **Sample 0**                                      | **Sample 150**                                       | **Sample 450**                                       |
| --------------------- | ------------------------------------------------- | ---------------------------------------------------- | ---------------------------------------------------- |
| **Input RGB**         | ![q2-3-img](results/q2/mesh/q2_mesh_rgb_0.png)    | ![q2-3-img2](results/q2/mesh/q2_mesh_rgb_150.png)    | ![q2-3-img3](results/q2/mesh/q2_mesh_rgb_450.png)    |
| **Mesh prediction**   | ![q2-3-pred](results/q2/mesh/q2_mesh_pred_0.gif)  | ![q2-3-pred2](results/q2/mesh/q2_mesh_pred_150.gif)  | ![q2-3-pred3](results/q2/mesh/q2_mesh_pred_450.gif)  |
| **Mesh ground truth** | ![q2-3-gt-mesh](results/q2/mesh/q2_mesh_gt_0.gif) | ![q2-3-gt-mesh2](results/q2/mesh/q2_mesh_gt_150.gif) | ![q2-1-gt-mesh3](results/q2/mesh/q2_mesh_gt_450.gif) |

As we can see, the performance of the mesh reconstruction is quite poor where the reconstruction shows different meshes collapsing together. This indicates that reconstruction from a single view might be challenging to the model.

### Q2.4 Quantitative comparisons

For quantitative comaprison, we can the following three curves of F1 score of voxel, point cloud and mesh. Note that I used the 10000 points curve for the point cloud.

![f1-vox](results/q2/vox_final/eval_vox.png) ![f1-point](results/q2/point_10000/eval_point.png) ![f1-mesh](results/q2/mesh/eval_mesh.png)

The F1-score curves indicate that the point cloud method performs best, capturing the essential structure of 3D objects with high accuracy probably due to its flexibility to represent complex shapes. Meshes perform second well, offering a continuous surface representation that captures smooth details but may struggle with complex topologies. Voxels, with their grid-like structure, have the lowest F1-scores, as they often miss finer details and complex geometries, limiting their effectiveness in detailed reconstructions.

### Q2.5 Analyse effects of hyperparams variations

In this section, I will provide the ablation study on the number of points when fitting to point clouds since it is more intuitive. I trained three different models with 1000, 5000, and 10000 points. Below are the visualization comparisons of the predictions of these three models.

| **Description**              | **1000 points (sample 0)**                                | **5000 points (sample 0)**                                | **10000 points (sample 0)**                                |
| ---------------------------- | --------------------------------------------------------- | --------------------------------------------------------- | ---------------------------------------------------------- |
| **Input RGB**                | ![q2-5-imgp1](results/q2/point_1000/q2_point_rgb_0.png)   | ![q2-5-imgp2](results/q2/point_5000/q2_point_rgb_0.png)   | ![q2-5-imgp3](results/q2/point_10000/q2_point_rgb_0.png)   |
| **Point cloud prediction**   | ![q2-5-predp1](results/q2/point_1000/q2_point_pred_0.gif) | ![q2-5-predp2](results/q2/point_5000/q2_point_pred_0.gif) | ![q2-5-predp3](results/q2/point_10000/q2_point_pred_0.gif) |
| **Point cloud ground truth** | ![q2-5-gtp1](results/q2/point_1000/q2_point_gt_0.gif)     | ![q2-5-gtp2](results/q2/point_5000/q2_point_gt_0.gif)     | ![q2-5-gtp3](results/q2/point_10000/q2_point_gt_0.gif)     |
| **Mesh ground truth**        | ![q2-5-gtp-mesh1](results/q2/point_1000/q2_mesh_gt_0.gif) | ![q2-5-gtp-mesh2](results/q2/point_5000/q2_mesh_gt_0.gif) | ![q2-5-gtp-mesh3](results/q2/point_10000/q2_mesh_gt_0.gif) |
| **F1 curves**                | ![q2-5-f1p1](results/q2/point_1000/eval_point.png)        | ![q2-5-f1p2](results/q2/point_5000/eval_point.png)        | ![q2-5-f1p3](results/q2/point_10000/eval_point.png)        |

As we can see, the ablation study on point cloud shows that increasing the number of points significantly improves point cloud reconstruction. With 1000 points, the predictions are sparse and lack detail. As the number increases to 5000 and 10000, the reconstructions become more detailed and aligned with the ground truth, as reflected in higher F1 scores. This indicates that a higher point density enhances the model's ability to capture complex geometries. However, I also noticed that the training and evalation time increased as more number of points were used. Therefore, it's crucial to balance accuracy with computational cost, as more points demand greater resources.

### Q2.6 Intepret the model

For this section, I am interested in what the voxel model learned after training in each intermediate layer. Since I am using
