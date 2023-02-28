from pytorch_lightning import Trainer
from occdepth.models.OccDepth import OccDepth
from occdepth.data.NYU.nyu_dm import NYUDataModule
from occdepth.data.semantic_kitti.kitti_dm import KittiDataModule
from occdepth.data.tartanair.tartanair_dm import TartanAirDataModule

import hydra
from omegaconf import DictConfig
import torch
import os
from hydra.utils import get_original_cwd

config_path= os.getenv('DATA_CONFIG')

@hydra.main(config_name=config_path)
def main(config: DictConfig):
    torch.set_grad_enabled(False)
    load_strict = not (config.conv3d_triplane_super or config.use_dino_distill)
    config.conv3d_triplane_super = False
    config.use_dino_distill = False
    if config.dataset == "kitti":
        config.batch_size = 1
        full_scene_size = tuple(config.full_scene_size)
        data_module = KittiDataModule(
            root=config.data_root,
            preprocess_root=config.data_preprocess_root,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size_per_gpu),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            pattern_id=config.pattern_id,
            multi_view_mode=config.multi_view_mode,
            use_stereo_depth_gt=config.use_stereo_depth_gt,
            use_lidar_depth_gt=config.use_lidar_depth_gt,
            data_stereo_depth_root=config.data_stereo_depth_root,
            data_lidar_depth_root=config.data_lidar_depth_root,
        )

    elif config.dataset == "NYU":
        config.batch_size = 1
        full_scene_size = tuple(config.full_scene_size)
        data_module = NYUDataModule(
            root=config.data_root,
            preprocess_root=config.data_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size_per_gpu),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            pattern_id=config.pattern_id,
        )
    elif config.dataset == "tartanair":
        data_module = TartanAirDataModule(
            config=config,
        )

    trainer = Trainer(
        sync_batchnorm=True, deterministic=True, gpus=config.n_gpus, accelerator="ddp"
    )

    model_path = os.path.join(get_original_cwd(), "trained_models", "occdepth.ckpt")

    print(
        "##### Max CUDA memory before load model: {} G".format(
            torch.cuda.max_memory_allocated() / (1024**3)
        )
    )
    model = OccDepth.load_from_checkpoint(
        model_path,
        full_scene_size=full_scene_size,
        config=config,
        strict=load_strict,
    )
    model.cuda()
    model.eval()
    print(
        "##### Max CUDA memory after load model: {} G".format(
            torch.cuda.max_memory_allocated() / (1024**3)
        )
    )
    data_module.setup()
    val_dataloader = data_module.val_dataloader()
    trainer.test(model, test_dataloaders=val_dataloader)
    print(
        "##### Max CUDA memory during all evaluation process: {} G".format(
            torch.cuda.max_memory_allocated() / (1024**3)
        )
    )


if __name__ == "__main__":
    main()
