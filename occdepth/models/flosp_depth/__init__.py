from occdepth.models.flosp_depth.flosp_depth_conf_kitti import (
    flosp_depth_conf as flosp_depth_conf_kitti,
)
from occdepth.models.flosp_depth.flosp_depth_conf_nyu import (
    flosp_depth_conf as flosp_depth_conf_nyu,
)

flosp_depth_conf_map = {
    "NYU": flosp_depth_conf_nyu,
    "kitti": flosp_depth_conf_kitti,
}
