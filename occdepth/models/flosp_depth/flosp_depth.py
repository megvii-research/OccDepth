import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.backbones.resnet import BasicBlock

import math
from occdepth.models.f2v.frustum_grid_generator import FrustumGridGenerator
from occdepth.models.f2v.frustum_to_voxel import FrustumToVoxel
from occdepth.models.f2v.sampler import Sampler


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = ranks[1:] != ranks[:-1]

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = ranks[1:] != ranks[:-1]

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        (kept,) = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            BatchNorm=BatchNorm,
        )
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            BatchNorm=BatchNorm,
        )
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            BatchNorm=BatchNorm,
        )
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            BatchNorm=BatchNorm,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), mid_channels, 1, bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.ReLU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        context_channels,
        depth_channels,
        infer_mode=False,
    ):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.mlp = Mlp(1, mid_channels, mid_channels)
        self.se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
        )
        # self.aspp = ASPP(mid_channels, mid_channels, BatchNorm=nn.InstanceNorm2d)

        self.depth_pred = nn.Conv2d(
            mid_channels, depth_channels, kernel_size=1, stride=1, padding=0
        )
        self.infer_mode = infer_mode

    def forward(
        self,
        x=None,
        sweep_intrins=None,
        scaled_pixel_size=None,
        scale_depth_factor=1000.0,
    ):
        if self.infer_mode:
            scaled_pixel_size = scaled_pixel_size
        else:
            inv_intrinsics = torch.inverse(sweep_intrins)
            pixel_size = torch.norm(
                torch.stack(
                    [inv_intrinsics[..., 0, 0], inv_intrinsics[..., 1, 1]], dim=-1
                ),
                dim=-1,
            ).reshape(-1, 1)
            scaled_pixel_size = pixel_size * scale_depth_factor

        x = self.reduce_conv(x)
        # aug_scale = torch.sqrt(sweep_post_rots_ida[..., 0, 0] ** 2 + sweep_post_rots_ida[..., 0, 1] ** 2).reshape(-1, 1)
        x_se = self.mlp(scaled_pixel_size)[..., None, None]

        x = self.se(x, x_se)
        x = self.depth_conv(x)
        # x = self.aspp(x)
        depth = self.depth_pred(x)
        return depth


class PCFE(nn.Module):
    """
    pixel cloud feature extraction
    """

    def __init__(self, in_channels, mid_channels, out_channels):
        super(PCFE, self).__init__()

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(
                mid_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            # nn.BatchNorm3d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x) + x
        x = self.out_conv(x)
        return x


class FlospDepth(nn.Module):
    def __init__(
        self,
        x_bound,
        y_bound,
        z_bound,
        d_bound,
        final_dim,
        downsample_factor,
        output_channels,
        depth_net_conf,
        scene_size,
        project_scale,
        return_depth,
        agg_voxel_mode="mean",  # "mean" or "sum"
        infer_mode=False,
        **kwargs
    ):
        super().__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels
        self.depth_channels = int((self.d_bound[1] - self.d_bound[0]) / self.d_bound[2])

        self.scene_size = scene_size
        self.project_scale = project_scale
        self.infer_mode = infer_mode

        # self.pcfe = self._configure_pcfe()
        self.depth_net_conf = depth_net_conf
        self.depth_net = self._configure_depth_net(depth_net_conf)

        self.register_buffer(
            "voxel_size",
            torch.Tensor(
                [row[2] * self.project_scale for row in [x_bound, y_bound, z_bound]]
            ),
        )
        self.register_buffer(
            "voxel_coord",
            torch.Tensor(
                [
                    row[0] + row[2] / 2.0 * self.project_scale
                    for row in [x_bound, y_bound, z_bound]
                ]
            ),
        )
        self.register_buffer(
            "voxel_num",
            torch.LongTensor(
                [
                    (row[1] - row[0]) / row[2] / self.project_scale
                    for row in [x_bound, y_bound, z_bound]
                ]
            ),
        )

        self.use_quickcumsum = True
        self.return_depth = return_depth

        grid_size = self.voxel_num
        pc_range = [
            x_bound[0],
            y_bound[0],
            z_bound[0],
            x_bound[1],
            y_bound[1],
            z_bound[1],
        ]

        disc_cfg = {
            "mode": "LID",
            "num_bins": self.depth_channels,
            "depth_min": self.d_bound[0],
            "depth_max": self.d_bound[1],
        }
        self.disc_cfg = disc_cfg
        if not self.infer_mode:
            self.grid_generator = FrustumGridGenerator(
                grid_size=grid_size, pc_range=pc_range, disc_cfg=disc_cfg
            )
        self.sampler = Sampler(mode="bilinear", padding_mode="zeros")
        self.agg_voxel_mode = agg_voxel_mode

    def _configure_depth_net(self, depth_net_conf):
        return nn.Sequential(
            DepthNet(
                depth_net_conf["in_channels"],
                depth_net_conf["mid_channels"],
                self.output_channels,
                self.depth_channels,
                infer_mode=self.infer_mode,
            )
        )

    def _forward_depth_net(
        self, feat, intrins_mat=None, scaled_pixel_size=None, *args, **kwargs
    ):
        if self.infer_mode:
            return self.depth_net[0](
                x=feat,
                sweep_intrins=None,
                scaled_pixel_size=scaled_pixel_size,
            )
        else:
            return self.depth_net[0](
                x=feat,
                sweep_intrins=intrins_mat,
                scaled_pixel_size=None,
            )

    def _forward_voxel_net(self, img_feat_with_depth):
        # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]
        img_feat_with_depth = img_feat_with_depth.permute(
            0, 3, 1, 4, 2
        ).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]
        n, h, c, w, d = img_feat_with_depth.shape
        img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)
        img_feat_with_depth = (
            self.pcfe(img_feat_with_depth)
            .view(n, h, c, w, d)
            .permute(0, 2, 4, 1, 3)
            .contiguous()
            .float()
        )
        return img_feat_with_depth

    def _configure_pcfe(self):
        """build pixel cloud feature extractor"""
        return PCFE(self.output_channels, self.output_channels, self.output_channels)

    def forward(
        self,
        img_feat,
        cam_k=None,
        T_velo_2_cam=None,
        ida_mats=None,
        vox_origin=None,
        grids=None,
        scaled_pixel_size=None,
    ):
        if vox_origin is not None:
            # NYU dataset
            if not self.infer_mode:

                x_max, y_max, z_max = 4.8, 4.8, 2.88
                x_bound = [vox_origin[0][0], vox_origin[0][0] + x_max, 0.08]
                y_bound = [vox_origin[0][1], vox_origin[0][1] + y_max, 0.08]
                z_bound = [vox_origin[0][2], vox_origin[0][2] + z_max, 0.08]
                # from IPython import embed; embed()
                self.register_buffer(
                    "voxel_size",
                    torch.Tensor(
                        [
                            row[2] * self.project_scale
                            for row in [x_bound, y_bound, z_bound]
                        ]
                    ),
                )
                self.register_buffer(
                    "voxel_coord",
                    torch.Tensor(
                        [
                            row[0] + row[2] / 2.0 * self.project_scale
                            for row in [x_bound, y_bound, z_bound]
                        ]
                    ),
                )
                self.register_buffer(
                    "voxel_num",
                    torch.LongTensor(
                        [
                            (row[1] - row[0]) / row[2] / self.project_scale
                            for row in [x_bound, y_bound, z_bound]
                        ]
                    ),
                )
                pc_range = [
                    x_bound[0],
                    y_bound[0],
                    z_bound[0],
                    x_bound[1],
                    y_bound[1],
                    z_bound[1],
                ]

                grid_size = self.voxel_num
                self.grid_generator = FrustumGridGenerator(
                    grid_size=grid_size, pc_range=pc_range, disc_cfg=self.disc_cfg
                )
                self.grid_generator = self.grid_generator.to(img_feat.device)
                self.voxel_coord = self.voxel_coord.to(img_feat.device)
                self.voxel_num = self.voxel_num.to(img_feat.device)
                self.voxel_size = self.voxel_size.to(img_feat.device)

        bs, n_cams, c, h, w = img_feat.shape
        img_feat = img_feat.reshape(bs * n_cams, c, h, w)

        if not self.infer_mode:
            ida_mats = torch.stack(ida_mats)
            T_velo_2_cam = torch.stack(T_velo_2_cam).to(torch.float32)
            cam_k = torch.stack(cam_k).to(torch.float32)

            intrins_mat = cam_k.new_zeros(bs, n_cams, 4, 4)
            intrins_mat[:, :, :3, :3] = cam_k
            intrins_mat[:, :, 3, 3] = 1

            # Generate sampling grid for frustum volume
            image_shape = cam_k.new_zeros(bs, 2)
            image_shape[:, 0:2] = torch.as_tensor(self.final_dim)

        if self.infer_mode:
            depth_feature = self._forward_depth_net(
                feat=img_feat,
                intrins_mat=None,
                scaled_pixel_size=scaled_pixel_size,
            )
        else:
            depth_feature = self._forward_depth_net(
                feat=img_feat,
                intrins_mat=intrins_mat,
                scaled_pixel_size=None,
            )
        depth = depth_feature.softmax(1)

        depth = depth.unsqueeze(1)

        depth = depth.reshape(
            bs,
            n_cams,
            depth.shape[1],
            depth.shape[2],
            depth.shape[3],
            depth.shape[4],
        )  # add N-cam dim

        all_voxel_features = []

        for i in range(n_cams):
            if self.infer_mode:
                grid = grids[i]
            else:
                grid = self.grid_generator(
                    lidar_to_cam=T_velo_2_cam[:, i],
                    cam_to_img=intrins_mat[:, i, :3, :],
                    ida_mats=ida_mats[:, i, ...],
                    image_shape=image_shape,
                )  # (B, X, Y, Z, 3)

            # Sample frustum volume to generate voxel volume
            voxel_features = self.sampler(
                input_features=depth[:, i, ...], grid=grid
            )  # (B, C, X, Y, Z)
            if self.agg_voxel_mode == "mean" and n_cams > 1:
                ones_feat = depth.new_ones(*depth.shape)
                voxel_mask = self.sampler(
                    input_features=ones_feat[:, i, ...], grid=grid
                )  # (B, C, X, Y, Z)
                if i == 0:
                    voxel_masks = [voxel_mask]
                else:
                    voxel_masks.append(voxel_mask)
            all_voxel_features.append(voxel_features)
        if n_cams == 1:
            agg_voxel_features = all_voxel_features[0]
        else:
            if self.agg_voxel_mode == "sum":
                agg_voxel_features = sum(all_voxel_features)
            elif self.agg_voxel_mode == "mean":
                agg_voxel_features = sum(all_voxel_features)
                masks = sum(voxel_masks)
                agg_voxel_features[masks > 0] = (
                    agg_voxel_features[masks > 0] / masks[masks > 0]
                )
            else:
                raise NotImplementedError(
                    "agg_voxel_mode: {}".format(self.agg_voxel_mode)
                )

        depth = depth.squeeze(2)
        if self.return_depth:
            return agg_voxel_features, depth
        else:
            return agg_voxel_features
