#!/bin/bash
cd /home/yb107/cvpr2025/DukeDiffSeg

LOGDIR="/home/yb107/logs"
LOGFILE="/home/yb107/logs/val_v2.log"
PIDFILE="/home/yb107/logs/val_v2.pid"

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
pipenv run bash -c "CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 setsid nohup python -m inference.diffunet_2_0 \
  training.num_gpus=1 \
  training.inference_mode=True \
  training.save_config_yaml=False \
  hydra.job.chdir=False \
  constraint=multi_class \
  task=colon_bowel \
  experiment.version=3.2 \
  evaluation.validation_max_num_samples=80 \
  model.params.use_spacing_info=True \
  training.start_epoch=1103 \
  training.resume=/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet_v2-multi_class-colon_bowel/3.2/checkpoints/DiffUNet_v2-multi_class-colon_bowel_latest_checkpoint_1103.pt \
  data.save_data=True \
  data.val_jsonl=/home/yb107/cvpr2025/DukeDiffSeg/data/c_grade_colons/3d_vlsmv2_c_grade_colon_dataset.jsonl \
  data.cache_dir=/data/usr/yb107/colon_data/cache_tmp \
  > $LOGFILE 2>&1 & echo \$! > $PIDFILE"

echo "🚀 Inference started — logs: $LOGFILE"
echo "📄 Main PID saved to: $PIDFILE"
echo "🔍 View logs: tail -f $LOGFILE"
echo "🛑 Kill inference and all subprocesses: /home/yb107/cvpr2025/DukeDiffSeg/scripts/kill_inference.sh"
echo "🎮 Check GPU usage: nvidia-smi"

