from torch.utils.data.dataloader import DataLoader
from occdepth.data.tartanair.tartanair_dataset import TartanAirDataset
import pytorch_lightning as pl
from occdepth.data.tartanair.collate import collate_fn
from occdepth.data.utils.torch_util import worker_init_fn


class TartanAirDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
    ):
        super().__init__()
        self.root = config.data_dir_root
        self.preprocess_root = config.data_preprocess_root
        self.batch_size = int(config.batch_size / config.n_gpus)  # batch size per GPU
        self.num_workers = config.num_workers_per_gpu * config.n_gpus
        self.config = config

    def setup(self, stage=None):
        self.train_ds = TartanAirDataset(
            split="train",
            config=self.config,
            fliplr=0.5,
            color_jitter=(0.4, 0.4, 0.4),
        )

        self.val_ds = TartanAirDataset(
            split="val",
            config=self.config,
            fliplr=0,
            color_jitter=None,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            worker_init_fn=worker_init_fn,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    import pickle, hydra, os

    config_path = os.getenv("DATA_CONFIG")
    log_dir = os.getenv("DATA_LOG")
    pwd_dir = os.path.abspath(os.path.join(log_dir, "../.."))

    @hydra.main(config_name=config_path)
    def test(config):
        config.batch_size_per_gpu = 1
        config.n_gpus = 1
        config.num_workers_per_gpu = 0
        print(config)
        ds = TartanAirDataModule(
            config=config,
        )
        ds.setup()
        train_ds = ds.train_dataloader()
        for i, d in enumerate(train_ds):
            print(i)
            with open(os.path.join(pwd_dir, "data.pkl"), "wb") as f:
                pickle.dump(d, f)
            break

    test()
