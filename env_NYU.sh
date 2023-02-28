#!/bin/sh
workdir=$(cd $(dirname $0); pwd)
echo "workdir:" $workdir

export DATA_LOG=$workdir/logdir/NYU_b3_F100_virtualStereo_sqrt
export DATA_CONFIG=$workdir/occdepth/config/NYU/multicam_flospdepth_crp_stereodepth_cascadecls_2080ti.yaml
export MONOSCENE_OUTPUT=$workdir/output
export PYTHONPATH=$workdir:$PYTHONPATH
export ETS_TOOLKIT=qt4
export QT_API=pyqt5
