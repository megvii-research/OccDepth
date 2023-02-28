import pytorch_lightning as pl
import torch
import torch.nn as nn
from occdepth.models.unet3d_nyu import UNet3D as UNet3DNYU
from occdepth.models.unet3d_kitti import UNet3D as UNet3DKitti

from occdepth.loss.sscMetrics import SSCMetrics
from occdepth.loss.ssc_loss import sem_scal_loss, CE_ssc_loss, KL_sep, geo_scal_loss
from occdepth.models.SFA import SFA
from occdepth.loss.CRP_loss import compute_super_CP_multilabel_loss
import numpy as np
import torch.nn.functional as F
from occdepth.models.unet2d import UNet2D
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.checkpoint import checkpoint

# flosp_depth
from occdepth.models.flosp_depth.flosp_depth import FlospDepth
from occdepth.models.flosp_depth import flosp_depth_conf_map

# depth loss
from occdepth.loss.depth_loss import DepthClsLoss

# PCA
from sklearn.decomposition import PCA

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class OccDepth(pl.LightningModule):
    def __init__(
        self,
        class_names,
        class_weights,
        class_weights_occ=None,
        full_scene_size=None,
        project_res=[],
        config=None,
        infer_mode=False,
    ):
        super().__init__()

        self.project_res = project_res
        self.full_scene_size = full_scene_size
        self.class_names = class_names
        self.class_weights = class_weights
        self.class_weights_occ = class_weights_occ
        # parse config
        self.dataset = config.dataset
        self.frustum_size = config.frustum_size
        self.project_scale = config.project_scale
        self.n_relations = config.n_relations
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.fp_loss = config.fp_loss
        self.frustum_size = config.frustum_size
        self.context_prior = config.context_prior
        self.relation_loss = config.relation_loss
        self.CE_ssc_loss = config.CE_ssc_loss
        self.sem_scal_loss = config.sem_scal_loss
        self.geo_scal_loss = config.geo_scal_loss

        self.n_classes = config.n_classes
        self.feature = config.feature
        self.feature_2d_oc = config.feature_2d_oc
        self.trans_2d_to_3d = config.trans_2d_to_3d
        self.cascade_cls = config.cascade_cls
        self.occluded_cls = config.occluded_cls
        self.sem_step_decay_loss = config.sem_step_decay_loss

        # multi_view_mode
        self.multi_view_mode = config.multi_view_mode

        # cascade cls
        print("INFO: Use cascade cls: {}".format(self.cascade_cls))

        # occluded cls
        print("INFO: Use occluded cls: {}".format(self.occluded_cls))

        # onnx
        self.infer_mode = infer_mode
        if self.infer_mode:
            self.context_prior = False

        # depth gt
        self.use_stereo_depth_gt = config.use_stereo_depth_gt
        self.use_lidar_depth_gt = config.use_lidar_depth_gt
        self.use_depth_gt = config.use_depth_gt
        assert not (
            config.use_stereo_depth_gt and config.use_lidar_depth_gt
        ), "only with one depth data supported."
        self.with_depth_gt = (
            self.use_stereo_depth_gt or self.use_lidar_depth_gt or self.use_depth_gt
        )
        self.depth_loss_w = config.depth_loss_weight

        if self.dataset == "NYU":
            self.net_3d_decoder = UNet3DNYU(
                self.n_classes,
                nn.BatchNorm3d,
                n_relations=self.n_relations,
                feature=self.feature,
                full_scene_size=self.full_scene_size,
                context_prior=self.context_prior,
                cascade_cls=self.cascade_cls,
                infer_mode=self.infer_mode,
            )
        elif self.dataset == "kitti":
            self.net_3d_decoder = UNet3DKitti(
                self.n_classes,
                nn.BatchNorm3d,
                project_scale=self.project_scale,
                feature=self.feature,
                full_scene_size=self.full_scene_size,
                context_prior=self.context_prior,
                cascade_cls=self.cascade_cls,
                occluded_cls=self.occluded_cls,
                infer_mode=self.infer_mode,
            )
        self.net_rgb = UNet2D.build(
            out_feature=self.feature_2d_oc,
            use_decoder=True,
            backbone_2d_name=config.backbone_2d_name,
            return_up_feats=config.return_up_feats,
        )

        # log hyperparameters
        self.save_hyperparameters()

        self.train_metrics = SSCMetrics(self.n_classes)
        self.val_metrics = SSCMetrics(self.n_classes)
        self.test_metrics = SSCMetrics(self.n_classes)

        # init 2d-3d transformation
        self.init_2d_to_3d_trans(config)

        # step decay loss
        print("INFO: Use step decay loss: {}".format(self.sem_step_decay_loss))
        batch_size = config.batch_size_per_gpu * config.n_gpus
        if self.dataset == "kitti":
            self.total_batch = (3834 // batch_size) * 30
        elif self.dataset == "NYU":
            self.total_batch = (795 // batch_size) * 30
        else:
            raise NotImplementedError(self.dataset)
        self.cur_batch = 0


    def _init_reduce_weight(self):
        torch.nn.init.kaiming_normal_(self.reduce_conv.weight)

    def init_2d_to_3d_trans(self, config):
        # 2d->3d transformation
        print(
            "INFO: Selected 2d->3d transformation method: {}".format(
                self.trans_2d_to_3d
            )
        )

        if self.trans_2d_to_3d == "flosp":
            self.projects = {}
            self.scale_2ds = [1, 2, 4, 8]  # 2D scales
            for scale_2d in self.scale_2ds:
                self.projects[str(scale_2d)] = SFA(
                    config.full_scene_size,
                    project_scale=self.project_scale,
                    dataset=self.dataset,
                )

            self.projects = nn.ModuleDict(self.projects)
        elif self.trans_2d_to_3d == "flosp_depth":
            self.projects = {}
            self.scale_2ds = [1, 2, 4, 8]  # 2D scales
            for scale_2d in self.scale_2ds:
                self.projects[str(scale_2d)] = SFA(
                    config.full_scene_size,
                    project_scale=self.project_scale,
                    dataset=self.dataset,
                )

            self.projects = nn.ModuleDict(self.projects)
            self.flosp_depth_conf = flosp_depth_conf_map[self.dataset]
            self.flosp_depth_conf.update(
                {
                    "scene_size": config.full_scene_size,
                    "project_scale": config.project_scale,
                    "output_channels": config.feature,
                    "depth_net_conf": dict(
                        in_channels=config.feature,
                        mid_channels=self.flosp_depth_conf["depth_net_conf"][
                            "mid_channels"
                        ],
                    ),
                    "return_depth": self.with_depth_gt,
                    "infer_mode": self.infer_mode,
                }
            )
            self.flosp_depth = FlospDepth(**self.flosp_depth_conf)
            if self.with_depth_gt:
                self.depth_loss_fn = DepthClsLoss(
                    downsample_factor=self.flosp_depth_conf["downsample_factor"],
                    d_bound=self.flosp_depth_conf["d_bound"],
                )
        else:
            raise NotImplementedError(f"{self.trans_2d_to_3d} is not supported yet.")

    def process_rgbs(self, img, batch, n_views):
        depth_key = "gt_depth"
        x_rgb=[]
        x_rgb.append(self.net_rgb(img[:,0]))
        for i in range(1,n_views):
            with torch.no_grad():
                x_single_rgb=self.net_rgb(img[:,i])
                x_rgb.append(x_single_rgb)

        # ####use depth to generate right image######
        if(n_views==1 and depth_key in batch):
            if('virtual_bf' in batch):
                bf = batch["virtual_bf"][0].to(device)
            x_rgb_virtual={}
            for scale_2d in self.project_res:
                x_rgb_virtual["1_" + str(scale_2d)] = self.generate_virtual_img(batch,x_rgb[0]["1_" + str(scale_2d)],scale_2d,bf)
            x_rgb.append(x_rgb_virtual)
            n_views = 2

        return x_rgb,n_views

    def generate_virtual_img(self,batch,x_single_rgb,scale_2d,bf):
        depth_key = "gt_depth"
        depth_mat = batch[depth_key].to(device)

        x_scale = torch.clone(x_single_rgb)
        n_bs_scale, c_scale, h_scale, w_scale = x_scale.shape
        depth_mat_scale = nn.functional.interpolate(depth_mat,
                                        size=(h_scale,w_scale),
                                        mode="bilinear",
                                        align_corners=False,
        )

        bf_scale = bf/int(scale_2d)
        grid_dx = torch.div(bf_scale, depth_mat_scale).type_as(x_scale)
        grid_dx = torch.where(torch.isinf(grid_dx), torch.full_like(grid_dx, 0), grid_dx)
        h_d = torch.arange(-1,1,2/h_scale)
        w_d = torch.arange(-1,1,2/w_scale)
        meshx, meshy = torch.meshgrid((h_d, w_d))
        grid = []
        for i in range(n_bs_scale):
            grid.append(torch.stack((meshy, meshx), axis=2))
        grid = torch.stack(grid).to(device).type_as(grid_dx) # add batch dim
        grid_dx = grid_dx * 2/w_scale ## scale dx
 
        grid[:,:,:,0] = grid[:,:,:,0] + grid_dx[0,...]
        x_scale_new=nn.functional.grid_sample(x_scale, grid, mode='bilinear', padding_mode='border', align_corners=False)

        return x_scale_new
    
    def _forward_2d_to_3d(self, batch, x_rgb, img, bs, vox_origin):
        depth_pred = None
        if self.trans_2d_to_3d in ["flosp", "flosp_depth"]:
            x3ds_ori = []
            for i in range(bs):
                x3d = None
                for scale_2d in self.project_res:
                    # project features at each 2D scale to target 3D scale
                    scale_2d = int(scale_2d)
                    projected_pix = batch[
                        "projected_pix_{}".format(self.project_scale)
                    ][i].to(device)
                    fov_mask = batch["fov_mask_{}".format(self.project_scale)][i].to(
                        device
                    )
                    n_views=len(x_rgb)
                    x_rgb_reshape = []
                    for j in range(n_views):
                        x_rgb_reshape.append(x_rgb[j]["1_" + str(scale_2d)])
                    x_rgb_reshape = torch.stack(x_rgb_reshape,1 ).to(device)

                    # Sum all the 3D features
                    if x3d is None:
                        x3d = self.projects[str(scale_2d)](
                            x_rgb_reshape[i],
                            projected_pix // scale_2d,
                            fov_mask,
                        )
                    else:
                        x3d += self.projects[str(scale_2d)](
                            x_rgb_reshape[i],
                            projected_pix // scale_2d,
                            fov_mask,
                        )

                x3ds_ori.append(x3d)
            x3ds = torch.stack(x3ds_ori)
            if self.trans_2d_to_3d == "flosp_depth":
                rgb_feat_layer = "1_{}".format(
                    self.flosp_depth_conf["downsample_factor"]
                )
                x_rgb_reshape = []
                if self.dataset == "NYU":
                    n_views = 1
                for j in range(n_views):
                    x_rgb_reshape.append(x_rgb[j][rgb_feat_layer])
                img_feat = torch.stack(x_rgb_reshape,1 ).to(device)

                if self.infer_mode:
                    grids = batch["grids"]
                    scaled_pixel_size = batch["scaled_pixel_size"]
                    input_kwargs = {
                        "img_feat": img_feat,
                        "grids": grids,
                        "scaled_pixel_size": scaled_pixel_size,
                    }
                else:
                    input_kwargs = {
                        "img_feat": img_feat,
                        "cam_k": batch["cam_k"],
                        "T_velo_2_cam": batch["T_velo_2_cam"],
                        "ida_mats": batch["ida_mats"],
                        "vox_origin": vox_origin,
                    }
                if self.with_depth_gt:
                    x3ds_depth, depth_pred = self.flosp_depth(
                        **input_kwargs,
                    )
                else:
                    x3ds_depth = self.flosp_depth(
                        **input_kwargs,
                    )
                # TODO, tartanair dataset may need permute operation?
                if self.dataset == "NYU":
                    # TODO, more general way to avoid this operation
                    x3ds_depth = x3ds_depth.permute(0, 1, 2, 4, 3).contiguous()

                x3ds = x3ds * x3ds_depth * 100
        else:
            raise NotImplementedError(f"{self.trans_2d_to_3d} is not supported yet.")
        return x3ds, depth_pred

    def forward(self, batch):
        img = batch["img"].to(device)
        bs, n_views, c, h, w = img.shape
        out = {}
        """
        get feature map dict
            '1_1':  16x370x1220
            '1_2':  16x185x610
            '1_4':  16x93x305
            '1_8':  16x47x153
            '1_16': 16x24x77
        """
        x_rgb, n_views = self.process_rgbs(img,batch,n_views)

        if self.dataset == "NYU":
            vox_origin = batch["vox_origin"]
        elif self.dataset == "tartanair":
            vox_origin = batch["vox_origin"]
        elif self.dataset == "kitti":
            vox_origin = None
        else:
            raise NotImplementedError(
                "dataset is not supported: {}".format(self.dataset)
            )

        x3ds, depth_pred = self._forward_2d_to_3d(batch, x_rgb, img, bs, vox_origin)

        input_dict = {"x3d": x3ds}
        net_out = self.net_3d_decoder(input_dict)
        out.update(net_out)
        if self.with_depth_gt:
            out["depth_pred"] = depth_pred
        return out

    def step(self, batch, step_type, metric):
        bs = len(batch["img"])
        loss = 0
        out_dict = self(batch)
        ssc_pred = out_dict["ssc_logit"]
        target = batch["target"]
        cur_epoch = int((self.cur_batch / self.total_batch) * 30)

        if self.context_prior:
            P_logits = out_dict["P_logits"]
            CP_mega_matrices = batch["CP_mega_matrices"]

            if self.relation_loss:
                loss_rel_ce = compute_super_CP_multilabel_loss(
                    P_logits, CP_mega_matrices
                )
                loss += loss_rel_ce
                self.log(
                    step_type + "/loss_relation_ce_super",
                    loss_rel_ce.detach(),
                    on_epoch=True,
                    sync_dist=True,
                )

        class_weight = self.class_weights.type_as(batch["img"])
        if self.CE_ssc_loss:
            loss_ssc = CE_ssc_loss(ssc_pred, target, class_weight)
            loss += loss_ssc
            self.log(
                step_type + "/loss_ssc",
                loss_ssc.detach(),
                on_epoch=True,
                sync_dist=True,
            )
            if self.cascade_cls:
                occ_pred = out_dict["occ_logit"]
                target_occ = target.clone()
                target_occ[(target_occ != 0) & (target_occ != 255)] = 1
                class_weight_occ = self.class_weights_occ.type_as(batch["img"])
                loss_occ = CE_ssc_loss(occ_pred, target_occ, class_weight_occ)
                loss += loss_occ
                self.log(
                    step_type + "/loss_occ",
                    loss_occ.detach(),
                    on_epoch=True,
                    sync_dist=True,
                )
            if self.occluded_cls and "occluded" in batch:
                occluded_pred = out_dict["occluded_logit"]
                target_occluded = batch["occluded"]
                class_weight_occluded = torch.FloatTensor([1, 1])
                class_weight_occluded = class_weight_occluded.type_as(batch["img"])
                loss_occluded = CE_ssc_loss(
                    occluded_pred, target_occluded, class_weight_occluded
                )
                loss += loss_occluded
                self.log(
                    step_type + "/loss_occluded",
                    loss_occluded.detach(),
                    on_epoch=True,
                    sync_dist=True,
                )

        if self.with_depth_gt and self.trans_2d_to_3d=="flosp_depth" and "gt_depth" in batch:
            if self.use_stereo_depth_gt:
                depth_pred = out_dict["depth_pred"][:, 0].unsqueeze(
                    1
                )  # only left cam depth
            elif self.use_lidar_depth_gt:
                depth_pred = out_dict["depth_pred"]
            elif self.use_depth_gt:
                depth_pred = out_dict["depth_pred"]
            else:
                raise NotImplementedError("Only stereo depth gt supported.")
            depth_gt = batch["gt_depth"]
            loss_depth = (
                self.depth_loss_fn.get_depth_loss(depth_gt, depth_pred)
                * self.depth_loss_w
            )
            loss += loss_depth

            self.log(
                step_type + "/loss_depth",
                loss_depth.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.sem_scal_loss:
            if self.sem_step_decay_loss:
                sem_decay_scale = max(0.1, (1 - self.cur_batch / self.total_batch))
            else:
                sem_decay_scale = 1.0
            loss_sem_scal = sem_scal_loss(ssc_pred, target) * sem_decay_scale
            loss += loss_sem_scal
            self.log(
                step_type + "/loss_sem_scal",
                loss_sem_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.geo_scal_loss:
            loss_geo_scal = geo_scal_loss(ssc_pred, target)
            loss += loss_geo_scal
            self.log(
                step_type + "/loss_geo_scal",
                loss_geo_scal.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        if self.fp_loss and step_type != "test":
            frustums_masks = torch.stack(batch["frustums_masks"])
            frustums_class_dists = torch.stack(
                batch["frustums_class_dists"]
            ).float()  # (bs, n_frustums, n_classes)
            n_frustums = frustums_class_dists.shape[1]

            pred_prob = F.softmax(ssc_pred, dim=1)
            batch_cnt = frustums_class_dists.sum(0)  # (n_frustums, n_classes)

            frustum_loss = 0
            frustum_nonempty = 0
            for frus in range(n_frustums):
                frustum_mask = frustums_masks[:, frus, :, :, :].unsqueeze(1).float()
                prob = frustum_mask * pred_prob  # bs, n_classes, H, W, D
                prob = prob.reshape(bs, self.n_classes, -1).permute(1, 0, 2)
                prob = prob.reshape(self.n_classes, -1)
                cum_prob = prob.sum(dim=1)  # n_classes

                total_cnt = torch.sum(batch_cnt[frus])
                total_prob = prob.sum()
                if total_prob > 0 and total_cnt > 0:
                    frustum_target_proportion = batch_cnt[frus] / total_cnt
                    cum_prob = cum_prob / total_prob  # n_classes
                    frustum_loss_i = KL_sep(cum_prob, frustum_target_proportion)
                    frustum_loss += frustum_loss_i
                    frustum_nonempty += 1
            frustum_loss = frustum_loss / frustum_nonempty
            loss += frustum_loss
            self.log(
                step_type + "/loss_frustums",
                frustum_loss.detach(),
                on_epoch=True,
                sync_dist=True,
            )

        y_true = target.cpu().numpy()
        y_pred = ssc_pred.detach().cpu().numpy()
        y_pred = np.argmax(y_pred, axis=1)
        metric.add_batch(y_pred, y_true)

        self.log(step_type + "/loss", loss.detach(), on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        self.cur_batch += 1
        return self.step(batch, "train", self.train_metrics)

    def validation_step(self, batch, batch_idx):
        self.step(batch, "val", self.val_metrics)

    def validation_epoch_end(self, outputs):
        metric_list = [("train", self.train_metrics), ("val", self.val_metrics)]

        for prefix, metric in metric_list:
            stats = metric.get_stats()
            for i, class_name in enumerate(self.class_names):
                self.log(
                    "{}_SemIoU/{}".format(prefix, class_name),
                    stats["iou_ssc"][i],
                    sync_dist=True,
                )
            self.log("{}/mIoU".format(prefix), stats["iou_ssc_mean"], sync_dist=True)
            self.log("{}/IoU".format(prefix), stats["iou"], sync_dist=True)
            self.log("{}/Precision".format(prefix), stats["precision"], sync_dist=True)
            self.log("{}/Recall".format(prefix), stats["recall"], sync_dist=True)
            metric.reset()

    def test_step(self, batch, batch_idx):
        self.step(batch, "test", self.test_metrics)

    def test_epoch_end(self, outputs):
        classes = self.class_names
        metric_list = [("test", self.test_metrics)]
        for prefix, metric in metric_list:
            print("{}======".format(prefix))
            stats = metric.get_stats()
            print(
                "Precision={:.4f}, Recall={:.4f}, IoU={:.4f}".format(
                    stats["precision"] * 100, stats["recall"] * 100, stats["iou"] * 100
                )
            )
            print("class IoU: {}, ".format(classes))
            print(
                " ".join(["{:.4f}, "] * len(classes)).format(
                    *(stats["iou_ssc"] * 100).tolist()
                )
            )
            print("mIoU={:.4f}".format(stats["iou_ssc_mean"] * 100))
            metric.reset()

    def configure_optimizers(self):
        if self.dataset == "NYU":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[18, 24], gamma=0.4)
            return [optimizer], [scheduler]
        elif self.dataset == "kitti":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[18, 24], gamma=0.4)
            return [optimizer], [scheduler]
        elif self.dataset == "tartanair":
            optimizer = torch.optim.AdamW(
                self.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
            scheduler = MultiStepLR(optimizer, milestones=[20], gamma=0.1)
            return [optimizer], [scheduler]


if __name__ == "__main__":
    import hydra, pickle,os

    from occdepth.data.semantic_kitti.params import (
        semantic_kitti_class_frequencies,
        kitti_class_names,
    )
    from occdepth.data.NYU.params import (
        class_weights as NYU_class_weights,
        NYU_class_names,
    )
    from occdepth.data.tartanair.params import (
        class_weights as tartanair_class_weights,
        tartanair_class_names,
    )
    config_path= os.getenv('DATA_CONFIG')
    pwd_dir = os.path.abspath(os.path.join(config_path, "../../../.."))
    with open(os.path.join(pwd_dir,"data.pkl"), "rb") as f:
        fake_data = pickle.load(f)
    @hydra.main(config_name=config_path)
    def test_func(config):

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
        elif config.dataset == "NYU":
            class_names = NYU_class_names
            class_weights = NYU_class_weights  # torch.FloatTensor([0.05, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            class_weights_occ = torch.FloatTensor([0.05, 2])
        elif config.dataset == "tartanair":
            class_weights = tartanair_class_weights
            class_weights_occ = torch.FloatTensor([0.05, 2])
            class_names = tartanair_class_names

        # class_names = kitti_class_names
        project_res = ["1"]
        if config.project_1_2:
            project_res.append("2")
        if config.project_1_4:
            project_res.append("4")
        if config.project_1_8:
            project_res.append("8")

        do_export_onnx = "export_onnx" in config and config.export_onnx == True
        do_model_thop = "model_thop" in config and config.model_thop == True
        infer_mode = do_export_onnx or do_model_thop

        model = OccDepth(
            full_scene_size=tuple(config.full_scene_size),
            project_res=project_res,
            class_names=class_names,
            class_weights=class_weights,
            class_weights_occ=class_weights_occ,
            infer_mode=infer_mode,
            config=config,
        )
        # 传入需要两个数据 一个是 正常传入网络的结构，一个是网络
        # 如果网络加载到cuda中，传入的数据也需要.cuda()
        model.to(device)
        res = model(fake_data)
        if do_model_thop:
            from thop import profile
            from thop import clever_format

            flops, params = profile(model, (fake_data,))
            # flops, params = profile(model, fake_data)
            flops, params = clever_format([flops, params], "%.3f")
            print("flops: ", flops)
            print("params: ", params)
        print("res", [r.shape for r in res.values()])
        if do_export_onnx:
            print("Export onnx:")
            torch.onnx.export(
                model,
                fake_data,
                "total_model.onnx",
                opset_version=13,
                do_constant_folding=True,
            )

    test_func()
