from occdepth.data.semantic_kitti.kitti_dm import KittiDataModule
from occdepth.data.semantic_kitti.params import (
    semantic_kitti_class_frequencies,
    kitti_class_names,
)
from occdepth.data.NYU.params import (
    class_weights as NYU_class_weights,
    NYU_class_names,
)
from occdepth.data.NYU.nyu_dm import NYUDataModule

from occdepth.data.tartanair.tartanair_dm import TartanAirDataModule
from occdepth.data.tartanair.params import (
    class_weights as tartanair_class_weights,
    tartanair_class_names,
)
from torch.utils.data.dataloader import DataLoader
from occdepth.models.OccDepth import OccDepth
from occdepth.scripts.utils import load_pretrain_model
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import os
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything

hydra.output_subdir = None
config_path= os.getenv('DATA_CONFIG')

@hydra.main(config_name=config_path)
def main(config: DictConfig):
    print(f"load config: dataset={config.dataset}")
    exp_name = config.exp_prefix
    exp_name += "_{}_{}".format(config.dataset, config.run)
    exp_name += "_FrusSize_{}".format(config.frustum_size)
    exp_name += "_nRelations{}".format(config.n_relations)
    exp_name += "_WD{}_lr{}".format(config.weight_decay, config.lr)

    if config.CE_ssc_loss:
        exp_name += "_CEssc"
    if config.geo_scal_loss:
        exp_name += "_geoScalLoss"
    if config.sem_scal_loss:
        exp_name += "_semScalLoss"
    if config.fp_loss:
        exp_name += "_fpLoss"

    if config.relation_loss:
        exp_name += "_CERel"
    if config.context_prior:
        exp_name += "_3DCRP"
    if config.multi_view_mode:
        exp_name += "_MultiView"
    if config.cascade_cls:
        exp_name += "_cascadecls"
    if config.use_depth_gt or config.use_stereo_depth_gt or config.use_lidar_depth_gt:
        exp_name += "_withdepth"

    max_epochs = config.max_epochs
    logdir = config.logdir
    full_scene_size = tuple(config.full_scene_size)
    # Setup dataloaders
    if config.dataset == "kitti":
        class_names = kitti_class_names
        class_weights = torch.from_numpy(
            1 / np.log(semantic_kitti_class_frequencies + 0.001)
        )
        semantic_kitti_class_frequencies_occ = np.array(
            [
                semantic_kitti_class_frequencies[0],
                semantic_kitti_class_frequencies[1:].sum(),
            ]
        )
        class_weights_occ = torch.from_numpy(
            1 / np.log(semantic_kitti_class_frequencies_occ + 0.001)
        )

        data_module = KittiDataModule(
            root=config.data_root,
            preprocess_root=config.data_preprocess_root,
            frustum_size=config.frustum_size,
            project_scale=config.project_scale,
            batch_size=int(config.batch_size_per_gpu),
            num_workers=int(config.num_workers_per_gpu),
            pattern_id=config.pattern_id,
            multi_view_mode=config.multi_view_mode,
            use_stereo_depth_gt=config.use_stereo_depth_gt,
            use_lidar_depth_gt=config.use_lidar_depth_gt,
            data_stereo_depth_root=config.data_stereo_depth_root,
            data_lidar_depth_root=config.data_lidar_depth_root,
            occluded_cls=config.occluded_cls,
            use_strong_img_aug=config.use_strong_img_aug,
        )

    elif config.dataset == "NYU":
        class_names = NYU_class_names
        class_weights = NYU_class_weights  # torch.FloatTensor([0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        class_weights_occ = torch.FloatTensor([0.05, 2])
        data_module = NYUDataModule(
            root=config.data_root,
            preprocess_root=config.data_preprocess_root,
            n_relations=config.n_relations,
            frustum_size=config.frustum_size,
            batch_size=int(config.batch_size_per_gpu),
            num_workers=int(config.num_workers_per_gpu * config.n_gpus),
            pattern_id=config.pattern_id,
            use_depth_gt=config.use_depth_gt,
            use_strong_img_aug=config.use_strong_img_aug,
        )
    elif config.dataset == "tartanair":
        class_names = tartanair_class_names
        class_weights = tartanair_class_weights
        class_weights_occ = torch.FloatTensor([0.05, 2])
        data_module = TartanAirDataModule(
            config=config,
        )
    else:
        print(f"{config.dataset} not invalid")
        raise NotImplementedError
    project_res = ["1"]
    if config.project_1_2:
        exp_name += "_Proj_2"
        project_res.append("2")
    if config.project_1_4:
        exp_name += "_4"
        project_res.append("4")
    if config.project_1_8:
        exp_name += "_8"
        project_res.append("8")

    print("exp=", exp_name)
    print("config=", config)

    # Initialize OccDepth model
    model = OccDepth(
        full_scene_size=full_scene_size,
        project_res=project_res,
        class_names=class_names,
        class_weights=class_weights,
        class_weights_occ=class_weights_occ,
        config=config,
    )

    if config.enable_log:
        logger = TensorBoardLogger(save_dir=logdir, name=exp_name, version="")
        lr_monitor = LearningRateMonitor(logging_interval="step")
        checkpoint_callbacks = [
            ModelCheckpoint(
                save_last=True,
                monitor="val/mIoU",
                save_top_k=1,
                mode="max",
                filename="{epoch:03d}-{val/mIoU:.5f}",
            ),
            ModelCheckpoint(
                save_last=True,
                monitor="val/IoU",
                save_top_k=1,
                mode="max",
                filename="{epoch:03d}-{val/IoU:.5f}",
            ),
            lr_monitor,
        ]
    else:
        logger = False
        checkpoint_callbacks = False

    model_path = os.path.join(logdir, exp_name, "checkpoints/last.ckpt")
    if os.path.isfile(model_path):
        # Continue training from last.ckpt
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            resume_from_checkpoint=model_path,
            sync_batchnorm=True,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator="ddp",
            amp_backend="native",
            gradient_clip_val=config.gradient_clip_val,
            deterministic=config.deterministic,  # will increase gpu memory
        )
    else:
        # Train from scratch
        trainer = Trainer(
            callbacks=checkpoint_callbacks,
            sync_batchnorm=True,
            max_epochs=max_epochs,
            gpus=config.n_gpus,
            logger=logger,
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            flush_logs_every_n_steps=100,
            accelerator="ddp",
            amp_backend="native",
            gradient_clip_val=config.gradient_clip_val,
            deterministic=config.deterministic,  # will increase gpu memory
        )

    trainer.fit(model, data_module)
    print("Training done.")


if __name__ == "__main__":
    seed_everything(42, workers=True)
    main()
