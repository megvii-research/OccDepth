from torch.utils.data.dataloader import DataLoader
from occdepth.data.semantic_kitti.kitti_dataset import KittiDataset
import pytorch_lightning as pl
from occdepth.data.semantic_kitti.collate import collate_fn
from occdepth.data.utils.torch_util import worker_init_fn


class KittiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        preprocess_root,
        project_scale=2,
        frustum_size=4,
        batch_size=4,
        num_workers=6,
        pattern_id=1,
        multi_view_mode=False,
        use_stereo_depth_gt=False,
        use_lidar_depth_gt=False,
        data_stereo_depth_root=None,
        data_lidar_depth_root=None,
        occluded_cls=False,
        use_strong_img_aug=False,
    ):
        super().__init__()
        self.root = root
        self.preprocess_root = preprocess_root
        self.project_scale = project_scale
        self.batch_size = batch_size  # batch size per GPU
        self.num_workers = num_workers
        self.frustum_size = frustum_size
        self.pattern_id = pattern_id
        self.multi_view_mode = multi_view_mode

        # depth gt
        self.use_stereo_depth_gt = use_stereo_depth_gt
        self.use_lidar_depth_gt = use_lidar_depth_gt
        self.with_depth_gt = self.use_stereo_depth_gt or self.use_lidar_depth_gt
        self.data_stereo_depth_root = data_stereo_depth_root
        self.data_lidar_depth_root = data_lidar_depth_root

        # occluded gt
        self.occluded_cls = occluded_cls
        self.with_occluded = self.occluded_cls

        # strong img aug
        self.use_strong_img_aug = use_strong_img_aug

    def setup(self, stage=None):
        self.train_ds = KittiDataset(
            split="train",
            root=self.root,
            preprocess_root=self.preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0.5,
            color_jitter=(0.4, 0.4, 0.4),
            pattern_id=self.pattern_id,
            multi_view_mode=self.multi_view_mode,
            use_stereo_depth_gt=self.use_stereo_depth_gt,
            use_lidar_depth_gt=self.use_lidar_depth_gt,
            data_stereo_depth_root=self.data_stereo_depth_root,
            data_lidar_depth_root=self.data_lidar_depth_root,
            with_occluded=self.with_occluded,
            use_strong_img_aug=self.use_strong_img_aug,
        )

        self.val_ds = KittiDataset(
            split="val",
            root=self.root,
            preprocess_root=self.preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
            pattern_id=self.pattern_id,
            multi_view_mode=self.multi_view_mode,
            use_stereo_depth_gt=False,
            with_occluded=False,
            use_strong_img_aug=False,
        )

        self.test_ds = KittiDataset(
            split="test",
            root=self.root,
            preprocess_root=self.preprocess_root,
            project_scale=self.project_scale,
            frustum_size=self.frustum_size,
            fliplr=0,
            color_jitter=None,
            pattern_id=self.pattern_id,
            multi_view_mode=self.multi_view_mode,
            use_stereo_depth_gt=False,
            with_occluded=False,
            use_strong_img_aug=False,
        )

    def train_dataloader(self):
        from functools import partial

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=partial(
                collate_fn,
                with_depth_gt=self.with_depth_gt,
                with_occluded=self.with_occluded,
            ),
        )

    def val_dataloader(self):
        from functools import partial

        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=partial(collate_fn, with_depth_gt=False, with_occluded=False),
        )

    def test_dataloader(self):
        from functools import partial

        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=partial(collate_fn, with_depth_gt=False, with_occluded=False),
        )


if __name__ == "__main__":
    import pickle, hydra, os
    from tqdm import tqdm
    from pathlib import Path

    mode = "dump_data"
    config_path= os.getenv('DATA_CONFIG')
    pwd_dir = os.path.abspath(os.path.join(config_path, "../../../.."))
    print(pwd_dir)
    @hydra.main(config_name=config_path)
    def test(config):
        ds = KittiDataModule(
            root=config.data_root,
            preprocess_root=config.data_preprocess_root,
            frustum_size=config.frustum_size,
            project_scale=config.project_scale,
            batch_size=2,
            num_workers=0,
            pattern_id=config.pattern_id,
            multi_view_mode=config.multi_view_mode,
            use_stereo_depth_gt=config.use_stereo_depth_gt,
            use_lidar_depth_gt=config.use_lidar_depth_gt,
            data_stereo_depth_root=config.data_stereo_depth_root,
            data_lidar_depth_root=config.data_lidar_depth_root,
        )
        ds.setup()
        train_ds = ds.train_dataloader()
        for i, d in tqdm(enumerate(train_ds)):
            if mode == "dump_data":
                print(i,d["sequence"],d["frame_id"])
                pickle_path = os.path.join(pwd_dir,"data.pkl")
                with open(pickle_path, "wb") as f:
                    pickle.dump(d, f)
                print("dump data to pickle:", pickle_path, "\n", "keys=", d.keys())
                break

    test()
