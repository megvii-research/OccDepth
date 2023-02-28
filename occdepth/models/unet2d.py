"""
Code adapted from https://github.com/shariqfarooq123/AdaBins/blob/main/models/unet_adaptive_bins.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

MODEL_NAME = "tf_efficientnet_b3_ns"
MODEL_CHANNELS = {
    "tf_efficientnet_b3_ns": [3, 24, 32, 48, 136],
    "tf_efficientnet_b4_ns": [3, 24, 32, 56, 160],
    "tf_efficientnet_b5_ns": [3, 32, 40, 64, 176],
    "tf_efficientnet_b7_ns": [3, 32, 48, 80, 224],
}
NUM_FEATURES = {
    "tf_efficientnet_b3_ns": 1536,
    "tf_efficientnet_b4_ns": 1792,
    "tf_efficientnet_b5_ns": 2048,
    "tf_efficientnet_b7_ns": 2560,
}


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()
        self._net = nn.Sequential(
            nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
            nn.Conv2d(
                output_features, output_features, kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(output_features),
            nn.LeakyReLU(),
        )

    def forward(self, x, concat_with):
        up_x = F.interpolate(
            x,
            size=(concat_with.shape[2], concat_with.shape[3]),
            mode="bilinear",
            align_corners=True,
        )
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)


class DecoderBN(nn.Module):
    def __init__(
        self,
        num_features,
        bottleneck_features,
        out_feature,
        use_decoder=True,
        backbone_2d_name=None,
        return_up_feats=None,
    ):
        super(DecoderBN, self).__init__()
        features = int(num_features)
        self.use_decoder = use_decoder
        self.backbone_2d_name = backbone_2d_name
        self.return_up_feats = return_up_feats

        self.conv2 = nn.Conv2d(
            bottleneck_features, features, kernel_size=1, stride=1, padding=1
        )

        self.out_feature_1_1 = out_feature
        self.out_feature_1_2 = out_feature
        self.out_feature_1_4 = out_feature
        self.out_feature_1_8 = out_feature
        self.out_feature_1_16 = out_feature
        self.feature_1_16 = features // 2
        self.feature_1_8 = features // 4
        self.feature_1_4 = features // 8
        self.feature_1_2 = features // 16
        self.feature_1_1 = features // 32

        if self.use_decoder:
            if self.return_up_feats <= 1:
                self.resize_output_1_1 = nn.Conv2d(
                    self.feature_1_1, self.out_feature_1_1, kernel_size=1
                )
            if self.return_up_feats <= 2:
                self.resize_output_1_2 = nn.Conv2d(
                    self.feature_1_2, self.out_feature_1_2, kernel_size=1
                )
            if self.return_up_feats <= 4:
                self.resize_output_1_4 = nn.Conv2d(
                    self.feature_1_4, self.out_feature_1_4, kernel_size=1
                )
            if self.return_up_feats <= 8:
                self.resize_output_1_8 = nn.Conv2d(
                    self.feature_1_8, self.out_feature_1_8, kernel_size=1
                )
            if self.return_up_feats <= 16:
                self.resize_output_1_16 = nn.Conv2d(
                    self.feature_1_16, self.out_feature_1_16, kernel_size=1
                )

            if self.return_up_feats <= 16:
                self.up16 = UpSampleBN(
                    skip_input=features + MODEL_CHANNELS[self.backbone_2d_name][4],
                    output_features=self.feature_1_16,
                )

            if self.return_up_feats <= 8:
                self.up8 = UpSampleBN(
                    skip_input=self.feature_1_16
                    + MODEL_CHANNELS[self.backbone_2d_name][3],
                    output_features=self.feature_1_8,
                )
            if self.return_up_feats <= 4:
                self.up4 = UpSampleBN(
                    skip_input=self.feature_1_8
                    + MODEL_CHANNELS[self.backbone_2d_name][2],
                    output_features=self.feature_1_4,
                )
            if self.return_up_feats <= 2:
                self.up2 = UpSampleBN(
                    skip_input=self.feature_1_4
                    + MODEL_CHANNELS[self.backbone_2d_name][1],
                    output_features=self.feature_1_2,
                )
            if self.return_up_feats <= 1:
                self.up1 = UpSampleBN(
                    skip_input=self.feature_1_2
                    + MODEL_CHANNELS[self.backbone_2d_name][0],
                    output_features=self.feature_1_1,
                )
        else:
            self.resize_output_1_1 = nn.Conv2d(3, out_feature, kernel_size=1)
            self.resize_output_1_2 = nn.Conv2d(32, out_feature * 2, kernel_size=1)
            self.resize_output_1_4 = nn.Conv2d(48, out_feature * 4, kernel_size=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = (
            features[4],
            features[5],
            features[6],
            features[8],
            features[11],
        )
        bs = x_block0.shape[0]
        x_d0 = self.conv2(x_block4)

        if self.use_decoder:
            res = {}
            if self.return_up_feats <= 16:
                x_1_16 = self.up16(x_d0, x_block3)
                res.update({"1_16": self.resize_output_1_16(x_1_16)})
            if self.return_up_feats <= 8:
                x_1_8 = self.up8(x_1_16, x_block2)
                res.update({"1_8": self.resize_output_1_8(x_1_8)})
            if self.return_up_feats <= 4:
                x_1_4 = self.up4(x_1_8, x_block1)
                res.update({"1_4": self.resize_output_1_4(x_1_4)})
            if self.return_up_feats <= 2:
                x_1_2 = self.up2(x_1_4, x_block0)
                res.update({"1_2": self.resize_output_1_2(x_1_2)})
            if self.return_up_feats <= 1:
                x_1_1 = self.up1(x_1_2, features[0])
                res.update({"1_1": self.resize_output_1_1(x_1_1)})
            return res
        else:
            x_1_1 = features[0]
            x_1_2, x_1_4, x_1_8, x_1_16 = (
                features[4],
                features[5],
                features[6],
                features[8],
            )
            x_global = features[-1].reshape(bs, 2560, -1).mean(2)
            return {
                "1_1": self.resize_output_1_1(x_1_1),
                "1_2": self.resize_output_1_2(x_1_2),
                "1_4": self.resize_output_1_4(x_1_4),
                "global": x_global,
            }


class Encoder(nn.Module):
    def __init__(self, backend):
        super(Encoder, self).__init__()
        self.original_model = backend

    def forward(self, x):
        features = [x]
        for k, v in self.original_model._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    features.append(vi(features[-1]))
            else:
                features.append(v(features[-1]))
        return features


class UNet2D(nn.Module):
    def __init__(
        self,
        backend,
        num_features,
        out_feature,
        use_decoder=True,
        backbone_2d_name=None,
        return_up_feats=1,
    ):
        super(UNet2D, self).__init__()
        self.use_decoder = use_decoder
        self.encoder = Encoder(backend)
        self.decoder = DecoderBN(
            out_feature=out_feature,
            use_decoder=use_decoder,
            bottleneck_features=num_features,
            num_features=num_features,
            backbone_2d_name=backbone_2d_name,
            return_up_feats=return_up_feats,
        )

    def forward(self, x, **kwargs):
        encoded_feats = self.encoder(x)
        unet_out = self.decoder(encoded_feats, **kwargs)
        return unet_out

    def get_encoder_params(self):  # lr/10 learning rate
        return self.encoder.parameters()

    def get_decoder_params(self):  # lr learning rate
        return self.decoder.parameters()

    @classmethod
    def build(cls, **kwargs):
        basemodel_name = kwargs["backbone_2d_name"]
        num_features = NUM_FEATURES[basemodel_name]

        print("Loading base model {}...".format(basemodel_name), end="")
        basemodel = torch.hub.load(
            "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
        )
        print("Done.")

        # Remove last layer
        print("Removing last two layers (global_pool & classifier).")
        basemodel.global_pool = nn.Identity()
        basemodel.classifier = nn.Identity()

        # Building Encoder-Decoder model
        print("Building Encoder-Decoder model..", end="")
        m = cls(basemodel, num_features=num_features, **kwargs)
        print("Done.")

        print("INFO: return_up_feats set to : {}.".format(kwargs["return_up_feats"]))

        return m


if __name__ == "__main__":
    model = UNet2D.build(out_feature=256, use_decoder=True)
