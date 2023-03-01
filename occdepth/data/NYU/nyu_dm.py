from torch.utils.data.dataloader import DataLoader
from occdepth.data.NYU.nyu_dataset import NYUDataset
from occdepth.data.NYU.collate import collate_fn
import pytorch_lightning as pl
from occdepth.data.utils.torch_util import worker_init_fn


class NYUDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root,
        preprocess_root,
        n_relations=4,
        batch_size=4,
        frustum_size=4,
        num_workers=6,
        pattern_id=0,
        use_depth_gt=False,
        use_strong_img_aug=False,
    ):
        super().__init__()
        self.n_relations = n_relations
        self.preprocess_root = preprocess_root
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.frustum_size = frustum_size
        self.pattern_id = pattern_id
        self.use_depth_gt = use_depth_gt
        self.with_depth_gt = use_depth_gt

        # strong img aug
        self.use_strong_img_aug = use_strong_img_aug

    def setup(self, stage=None):
        self.train_ds = NYUDataset(
            split="train",
            preprocess_root=self.preprocess_root,
            n_relations=self.n_relations,
            root=self.root,
            fliplr=0.5,
            frustum_size=self.frustum_size,
            color_jitter=(0.4, 0.4, 0.4),
            pattern_id=self.pattern_id,
            use_depth_gt=self.use_depth_gt,
            use_strong_img_aug=self.use_strong_img_aug,
        )
        self.test_ds = NYUDataset(
            split="test",
            preprocess_root=self.preprocess_root,
            n_relations=self.n_relations,
            root=self.root,
            frustum_size=self.frustum_size,
            fliplr=0.0,
            color_jitter=None,
            pattern_id=self.pattern_id,
            use_depth_gt=self.use_depth_gt,
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
            collate_fn=partial(collate_fn, with_depth_gt=self.with_depth_gt),
        )

    def val_dataloader(self):
        from functools import partial

        print("val dataloader batchsize: {}; set to 1".format(self.batch_size))
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=partial(collate_fn, with_depth_gt=self.with_depth_gt),
        )

    def test_dataloader(self):
        from functools import partial

        print("test dataloader batchsize: {}; set to 1".format(self.batch_size))
        return DataLoader(
            self.test_ds,
            batch_size=1,
            num_workers=self.num_workers,
            drop_last=False,
            shuffle=False,
            pin_memory=True,
            collate_fn=partial(collate_fn, with_depth_gt=self.with_depth_gt),
        )
