import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_help


def grid_sampler(g, input, grid, mode, padding_mode, align_corners):
    # mode
    #   'bilinear'      : onnx::Constant[value={0}]
    #   'nearest'       : onnx::Constant[value={1}]
    #   'bicubic'       : onnx::Constant[value={2}]
    # padding_mode
    #   'zeros'         : onnx::Constant[value={0}]
    #   'border'        : onnx::Constant[value={1}]
    #   'reflection'    : onnx::Constant[value={2}]
    mode = sym_help._maybe_get_const(mode, "i")
    padding_mode = sym_help._maybe_get_const(padding_mode, "i")
    mode_str = ["bilinear", "nearest", "bicubic"][mode]
    padding_mode_str = ["zeros", "border", "reflection"][padding_mode]
    align_corners = int(sym_help._maybe_get_const(align_corners, "b"))

    return g.op(
        "com.microsoft::GridSample",
        input,
        grid,
        mode_s=mode_str,
        padding_mode_s=padding_mode_str,
        align_corners_i=align_corners,
    )


register_custom_op_symbolic("::grid_sampler", grid_sampler, 1)


class Sampler(nn.Module):
    def __init__(self, mode="bilinear", padding_mode="zeros"):
        """
        Initializes module
        Args:
            mode [string]: Sampling mode [bilinear/nearest]
            padding_mode [string]: Padding mode for outside grid values [zeros/border/reflection]
        """
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(self, input_features, grid):
        """
        Samples input using sampling grid
        Args:
            input_features [torch.Tensor(N, C, H_in, W_in)]: Input feature maps
            grid [torch.Tensor(N, H_out, W,_out 2)]: Sampling grids for image features
        Returns
            output_features [torch.Tensor(N, C, H_out, W_out)]: Output feature maps
        """
        # Sample from grid
        output = F.grid_sample(
            input=input_features,
            grid=grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
        )
        return output
