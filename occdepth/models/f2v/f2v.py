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
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(
            mid_channels, context_channels, kernel_size=1, stride=1, padding=0
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

    def forward(self, x, sweep_intrins, scale_depth_factor=1000.0):
        x = self.reduce_conv(x)
        context = self.context_conv(x)
        inv_intrinsics = torch.inverse(sweep_intrins)
        pixel_size = torch.norm(
            torch.stack([inv_intrinsics[..., 0, 0], inv_intrinsics[..., 1, 1]], dim=-1),
            dim=-1,
        ).reshape(-1, 1)
        # aug_scale = torch.sqrt(sweep_post_rots_ida[..., 0, 0] ** 2 + sweep_post_rots_ida[..., 0, 1] ** 2).reshape(-1, 1)
        scaled_pixel_size = pixel_size * scale_depth_factor
        x_se = self.mlp(scaled_pixel_size)[..., None, None]
        x = self.se(x, x_se)
        x = self.depth_conv(x)
        # x = self.aspp(x)
        depth = self.depth_pred(x)
        return torch.cat([depth, context], dim=1)


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


class LSS(nn.Module):
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

        self.pcfe = self._configure_pcfe()
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
        self.grid_generator = FrustumGridGenerator(
            grid_size=grid_size, pc_range=pc_range, disc_cfg=disc_cfg
        )
        self.sampler = Sampler(mode="bilinear", padding_mode="zeros")

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        # FIXME, general way to get fH and fW @liuweizhou
        # fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        fH, fW = math.ceil(ogfH / self.downsample_factor), math.ceil(
            ogfW / self.downsample_factor
        )
        d_coords = (
            torch.arange(*self.d_bound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = d_coords.shape
        x_coords = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        y_coords = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, sensor2lidar_mat, intrin_mat, ida_mat=None, bda_mat=None):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2lidar_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        # points = points
        D, H, W, X = points.shape
        points = points.view(1, 1, D, H, W, X, 1).expand(
            batch_size, num_cams, D, H, W, X, 1
        )

        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:],
            ),
            5,
        )

        combine = sensor2lidar_mat.matmul(torch.inverse(intrin_mat))
        points = (
            combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points).squeeze(-1)
        )
        return points[..., :3]

    def _configure_depth_net(self, depth_net_conf):
        return nn.Sequential(
            DepthNet(
                depth_net_conf["in_channels"],
                depth_net_conf["mid_channels"],
                self.output_channels,
                self.depth_channels,
            )
        )

    def _forward_depth_net(self, feat, intrins_mat=None, *args, **kwargs):
        return self.depth_net[0](
            feat,
            intrins_mat,
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

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat(
            [
                torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long)
                for ix in range(B)
            ]
        )
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (
            (geom_feats[:, 0] >= 0)
            & (geom_feats[:, 0] < self.voxel_num[0])
            & (geom_feats[:, 1] >= 0)
            & (geom_feats[:, 1] < self.voxel_num[1])
            & (geom_feats[:, 2] >= 0)
            & (geom_feats[:, 2] < self.voxel_num[2])
        )
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = (
            geom_feats[:, 0] * (self.voxel_num[1] * self.voxel_num[2] * B)
            + geom_feats[:, 1] * (self.voxel_num[2] * B)
            + geom_feats[:, 2] * B
            + geom_feats[:, 3]
        )

        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros(
            (B, C, self.voxel_num[2], self.voxel_num[0], self.voxel_num[1]),
            device=x.device,
        )
        final[
            geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]
        ] = x

        return final

    def forward(self, img_feat, cam_k=None, T_velo_2_cam=None):
        n, c, h, w = img_feat.shape

        cam_k = torch.stack(cam_k).to(torch.float32)
        intrins_mat = cam_k.new_zeros(n, 4, 4)
        intrins_mat[:, :3, :3] = cam_k
        intrins_mat[:, 3, 3] = 1
        intrins_mat = intrins_mat.unsqueeze(1)
        intrins_mat = intrins_mat

        depth_feature = self._forward_depth_net(
            img_feat,
            intrins_mat,
        )
        depth = depth_feature[:, : self.depth_channels].softmax(1)
        img_feat_with_depth = depth.unsqueeze(1) * depth_feature[
            :, self.depth_channels : (self.depth_channels + self.output_channels)
        ].unsqueeze(2)
        img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

        T_velo_2_cam = torch.stack(T_velo_2_cam).to(torch.float32)
        img_feat_with_depth = img_feat_with_depth  # NxCxDxHxW

        # Generate sampling grid for frustum volume
        image_shape = cam_k.new_zeros(n, 2)
        image_shape[:, 0:2] = torch.as_tensor(self.final_dim)

        grid = self.grid_generator(
            lidar_to_cam=T_velo_2_cam,
            cam_to_img=intrins_mat.squeeze(1)[:, :3, :],
            image_shape=image_shape,
        )  # (B, X, Y, Z, 3)

        # Sample frustum volume to generate voxel volume
        voxel_features = self.sampler(
            input_features=img_feat_with_depth, grid=grid
        )  # (B, C, X, Y, Z)

        return voxel_features
