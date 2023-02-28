# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from occdepth.models.modules import (
    SegmentationHead,
    SegmentationHeadCascadeCLS,
    SegmentationHeadOccludedCLS,
)
from occdepth.models.CRP3D import CPMegaVoxels
from occdepth.models.modules import Process, Upsample, Downsample, Convblock3d


class UNet3D(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        full_scene_size,
        feature,
        project_scale,
        context_prior=None,
        bn_momentum=0.1,
        cascade_cls=False,
        occluded_cls=False,
        infer_mode=False,
    ):
        super(UNet3D, self).__init__()
        self.business_layer = []
        self.project_scale = project_scale
        self.full_scene_size = full_scene_size
        self.feature = feature
        self.cascade_cls = cascade_cls
        self.occluded_cls = occluded_cls
        self.infer_mode = infer_mode

        size_l1 = (
            int(self.full_scene_size[0] / project_scale),
            int(self.full_scene_size[1] / project_scale),
            int(self.full_scene_size[2] / project_scale),
        )
        size_l2 = (size_l1[0] // 2, size_l1[1] // 2, size_l1[2] // 2)
        size_l3 = (size_l2[0] // 2, size_l2[1] // 2, size_l2[2] // 2)

        dilations = [1, 2, 3]
        self.process_l1 = nn.Sequential(
            Process(self.feature, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature, norm_layer, bn_momentum),
        )
        self.process_l2 = nn.Sequential(
            Process(self.feature * 2, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature * 2, norm_layer, bn_momentum),
        )

        self.up_13_l2 = Upsample(
            self.feature * 4, self.feature * 2, norm_layer, bn_momentum
        )
        self.up_12_l1 = Upsample(
            self.feature * 2, self.feature, norm_layer, bn_momentum
        )
        if self.project_scale == 1:
            self.up_l1_lfull = Convblock3d(
                self.feature, self.feature // 2, norm_layer, bn_momentum, stride=1
            )
        else:
            self.up_l1_lfull = Upsample(
                self.feature, self.feature // 2, norm_layer, bn_momentum
            )
        if self.cascade_cls:
            self.ssc_head = SegmentationHeadCascadeCLS(
                self.feature // 2, self.feature // 2, class_num, dilations
            )
        else:
            self.ssc_head = SegmentationHead(
                self.feature // 2, self.feature // 2, class_num, dilations
            )

        if self.occluded_cls:
            self.occluded_head = SegmentationHeadOccludedCLS(
                self.feature // 2, self.feature // 2, class_num, dilations
            )

        self.context_prior = context_prior
        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(
                self.feature * 4, size_l3, bn_momentum=bn_momentum
            )

    def forward(self, input_dict):
        res = {}

        x3d_l1 = input_dict["x3d"]

        x3d_l2 = self.process_l1(x3d_l1)

        x3d_l3 = self.process_l2(x3d_l2)

        if self.context_prior:
            ret = self.CP_mega_voxels(x3d_l3)
            x3d_l3 = ret["x"]
            for k in ret.keys():
                res[k] = ret[k]

        x3d_up_l2 = self.up_13_l2(x3d_l3) + x3d_l2
        x3d_up_l1 = self.up_12_l1(x3d_up_l2) + x3d_l1
        x3d_up_lfull = self.up_l1_lfull(x3d_up_l1)

        if not self.infer_mode:
            res["x3d_l1"] = x3d_up_l1
            res["x3d_l2"] = x3d_up_l2
            res["x3d_l3"] = x3d_l3

        if self.cascade_cls:
            ssc_logit_full, ssc_logit_full_occ = self.ssc_head(x3d_up_lfull)
            res["ssc_logit"] = ssc_logit_full
            if not self.infer_mode:
                res["occ_logit"] = ssc_logit_full_occ
        else:
            ssc_logit_full = self.ssc_head(x3d_up_lfull)
            res["ssc_logit"] = ssc_logit_full

        if self.occluded_cls:
            occluded_logit_full = self.occluded_head(x3d_up_lfull)
            if not self.infer_mode:
                res["occluded_logit"] = occluded_logit_full
        return res


if __name__ == "__main__":
    import hydra, pickle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def export_onnx():
        class_num = 20
        norm_layer = torch.nn.modules.batchnorm.BatchNorm3d
        feature = 64
        full_scene_size = (128, 128, 16)
        model = UNet3D(
            class_num,
            norm_layer,
            full_scene_size,
            feature,
            project_scale=2,
            context_prior=False,
            bn_momentum=0.1,
            cascade_cls=False,
            occluded_cls=False,
            infer_mode=False,
        )

        # 传入需要两个数据 一个是 正常传入网络的结构，一个是网络
        # 如果网络加载到cuda中，传入的数据也需要.cuda()
        fake_data = {
            "x3d": torch.rand(
                1, feature, full_scene_size[0], full_scene_size[1], full_scene_size[2]
            ).to(device)
        }

        model.to(device)
        res = model(fake_data)
        print("res", [r.shape for r in res.values()])
        torch.onnx.export(
            model,
            fake_data,
            "kitti_3d_decoder.onnx",
            opset_version=11,
            do_constant_folding=True,
        )

    export_onnx()
