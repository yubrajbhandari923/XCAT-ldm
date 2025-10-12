#!/bin/bash
cd /home/yb107/cvpr2025/DukeDiffSeg

LOGFILE="/home/yb107/logs/train_.log"
PIDFILE="/home/yb107/logs/train_.pid"

# Clean up any old PID file
if [ -f "$PIDFILE" ]; then
    rm "$PIDFILE"
fi

# python -m train.segdiff_1_0 --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/segdiff.yaml
# pipenv run bash -c "CUDA_VISIBLE_DEVICES=3,4,5,6 OMP_NUM_THREADS=8 nohup torchrun --nproc_per_node=4 -m train.medsegdiff_1_0 \
#   --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/medsegdiff.yaml > /home/yb107/logs/train_medsegdiff.log 2>&1 &"
# pipenv run bash -c "CUDA_VISIBLE_DEVICES=3,4,5,6 OMP_NUM_THREADS=8 nohup python -m train.medsegdiff_1_0 \
#   --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/medsegdiff.yaml > /home/yb107/logs/train_medsegdiff.log 2>&1 &"


# Launch in new process group with setsid
pipenv run bash -c "CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=8 setsid nohup python -m train.diffunet_1_0 \
  --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/diffunet.yaml > $LOGFILE 2>&1 & echo \$! > $PIDFILE"

echo "🚀 Training started — logs: $LOGFILE"
echo "📄 Main PID saved to: $PIDFILE"
echo "🔍 View logs: tail -f $LOGFILE"
echo "🛑 Kill training and all subprocesses: /home/yb107/cvpr2025/DukeDiffSeg/scripts/kill_training.sh"
echo "🎮 Check GPU usage: nvidia-smi"

