# encoding: utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from occdepth.models.CRP3D import CPMegaVoxels
from occdepth.models.modules import (
    Process,
    Upsample,
    Downsample,
    ASPP,
)
from occdepth.models.modules import SegmentationHead, SegmentationHeadCascadeCLS


class UNet3D(nn.Module):
    def __init__(
        self,
        class_num,
        norm_layer,
        feature,
        full_scene_size,
        n_relations=4,
        project_res=[],
        context_prior=True,
        bn_momentum=0.1,
        cascade_cls=False,
        infer_mode=False,
    ):
        super(UNet3D, self).__init__()
        self.business_layer = []
        self.project_res = project_res
        self.cascade_cls = cascade_cls

        self.feature_1_4 = feature
        self.feature_1_8 = feature * 2
        self.feature_1_16 = feature * 4

        self.feature_1_16_dec = self.feature_1_16
        self.feature_1_8_dec = self.feature_1_8
        self.feature_1_4_dec = self.feature_1_4
        self.infer_mode = infer_mode

        self.process_1_4 = nn.Sequential(
            Process(self.feature_1_4, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature_1_4, norm_layer, bn_momentum),
        )
        self.process_1_8 = nn.Sequential(
            Process(self.feature_1_8, norm_layer, bn_momentum, dilations=[1, 2, 3]),
            Downsample(self.feature_1_8, norm_layer, bn_momentum),
        )
        self.up_1_16_1_8 = Upsample(
            self.feature_1_16_dec, self.feature_1_8_dec, norm_layer, bn_momentum
        )
        self.up_1_8_1_4 = Upsample(
            self.feature_1_8_dec, self.feature_1_4_dec, norm_layer, bn_momentum
        )
        if self.cascade_cls:
            self.ssc_head_1_4 = SegmentationHeadCascadeCLS(
                self.feature_1_4_dec, self.feature_1_4_dec, class_num, [1, 2, 3]
            )
        else:
            self.ssc_head_1_4 = SegmentationHead(
                self.feature_1_4_dec, self.feature_1_4_dec, class_num, [1, 2, 3]
            )

        self.context_prior = context_prior
        size_1_16 = tuple(np.ceil(i / 4).astype(int) for i in full_scene_size)

        if context_prior:
            self.CP_mega_voxels = CPMegaVoxels(
                self.feature_1_16,
                size_1_16,
                n_relations=n_relations,
                bn_momentum=bn_momentum,
            )

    #
    def forward(self, input_dict):
        res = {}

        x3d_1_4 = input_dict["x3d"]
        x3d_1_8 = self.process_1_4(x3d_1_4)
        x3d_1_16 = self.process_1_8(x3d_1_8)

        if self.context_prior:
            ret = self.CP_mega_voxels(x3d_1_16)
            x3d_1_16 = ret["x"]
            for k in ret.keys():
                res[k] = ret[k]

        x3d_up_1_8 = self.up_1_16_1_8(x3d_1_16) + x3d_1_8
        x3d_up_1_4 = self.up_1_8_1_4(x3d_up_1_8) + x3d_1_4

        if not self.infer_mode:
            res["x3d_l1"] = x3d_up_1_4
            res["x3d_l2"] = x3d_up_1_8
            res["x3d_l3"] = x3d_1_16

        if self.cascade_cls:
            ssc_logit_full, ssc_logit_full_occ = self.ssc_head_1_4(x3d_up_1_4)
            res["ssc_logit"] = ssc_logit_full
            if not self.infer_mode:
                res["occ_logit"] = ssc_logit_full_occ
        else:
            ssc_logit_1_4 = self.ssc_head_1_4(x3d_up_1_4)

            res["ssc_logit"] = ssc_logit_1_4

        return res


if __name__ == "__main__":
    import hydra, pickle

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def export_onnx():
        class_num = 12
        norm_layer = torch.nn.modules.batchnorm.BatchNorm3d
        feature = 200
        full_scene_size = (60, 36, 60)
        model = UNet3D(
            class_num,
            norm_layer,
            feature,
            full_scene_size,
            n_relations=4,
            project_res=[],
            context_prior=False,
            bn_momentum=0.1,
            cascade_cls=False,
            infer_mode=True,
        )

        # 传入需要两个数据 一个是 正常传入网络的结构，一个是网络
        # 如果网络加载到cuda中，传入的数据也需要.cuda()
        fake_data = {"x3d": torch.rand(1, 200, 60, 36, 60).to(device)}

        model.to(device)
        res = model(fake_data)
        print("res", [r.shape for r in res.values()])
        torch.onnx.export(
            model,
            fake_data,
            "nyu_3d_decoder.onnx",
            opset_version=11,
            do_constant_folding=True,
        )

    export_onnx()
