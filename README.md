# OccDepth: A Depth-aware Method for 3D Semantic Occupancy Network 


# News

- Code will release soon!!
- **2023/02/28** Paper released on [Arxiv](https://arxiv.org/abs/2302.13540).
- **2023/02/17** Demo release.

# Abstract
In this paper, we propose the first stereo SSC method named OccDepth, which fully exploits implicit depth information from stereo images to help the recovery of 3D geometric structures. The Stereo Soft Feature Assignment (Stereo-SFA) module is proposed to better fuse 3D depth-aware features by implicitly learning the correlation between stereo images. Besides, the Occupancy Aware Depth (OAD) module is used to obtain geometry-aware 3D features by knowledge distillation using pre-trained depth models. In addition, a reformed TartanAir benchmark, named SemanticTartanAir, is provided in this paper for further testing our OccDepth method on SSC task.

# Video Demo

Mesh results compared with ground truth on KITTI-08:
<p align="center">
<img src="./assets/demo.gif" alt="video loading..." />
</p>
Voxel results compared with ground truth on KITTI-08:
<p align="center">
<img src="./assets/demo_voxel.gif" alt="video loading..." />
</p>
Full demo videos can be downloaded via `git lfs pull`, the demo videos are saved as "assets/demo.mp4" and "assets/demo_voxel.mp4". 

# Results
## Qualitative Results
<div align="center">
<img width=374 src="./assets/result1-1.png"/><img width=400 src="./assets/result1-2.png"/>


Fig. 1: RGB based Semantic Scene Completion with/without depth-aware. (a) Our proposed OccDepth method can detect smaller and farther objects. (b) Our proposed OccDepth method complete road better.
</div>

## Quantitative results on SemanticKITTI

<div align="center">
Table 1. Performance on SemanticKITTI (hidden test set). 

|Method            |Input        | SC  IoU       | SSC mIoU       |
|:----------------:|:----------:|:--------------:|:--------------:|
| **2.5D/3D**      |            |                |                |
| LMSCNet(st)   | OCC        | 33.00          | 5.80           |
| AICNet(st)    | RGB, DEPTH | 32.8           | 6.80           |
| JS3CNet(st)   | PTS        | 39.30          | 9.10           |
| **2D**           |            |                |                |
| MonoScene        | RGB        | 34.16          | 11.08          |
| MonoScene(st) | Stereo RGB | 40.84          | 13.57          |
| OccDepth (ours)  | Stereo RGB | **45.10**      | **15.90**      |
</div>
The scene completion (SC IoU) and semantic scene completion (SSC mIoU) are reported for modified baselines (marked with "st") and our OccDepth.

## Detailed results on SemanticKITTI.
<div align="center">
<img src="./assets/result2.png"/>
</div>


# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.

# Related Repo's
* https://github.com/wzzheng/TPVFormer
* https://github.com/FANG-MING/occupancy-for-nuscenes
