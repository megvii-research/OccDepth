#!/bin/sh
workdir=$(cd $(dirname $0); pwd)
echo "workdir:" $workdir

export DATA_LOG=$workdir/logdir/b3_F32_FixMultiview
export DATA_CONFIG=$workdir/occdepth/config/semantic_kitti/multicam_flospdepth_crp_stereodepth_cascadecls_2080ti.yaml
export PYTHONPATH=$workdir:$PYTHONPATH
export ETS_TOOLKIT=qt4
export QT_API=pyqt5
