#!/bin/bash
cd /home/yb107/cvpr2025/DukeDiffSeg

LOGDIR="/home/yb107/logs"
LOGFILE="/home/yb107/logs/train_v3.log"
PIDFILE="/home/yb107/logs/train_v3.pid"

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
pipenv run bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 setsid nohup python -m train.diffunet_2_0 \
  training.num_gpus=8 \
  experiment.debug=False \
  data.save_data=False \
  data.batch_size_per_gpu=4 \
  model.params.use_spacing_info=True \
  evaluation.validation_max_num_samples=100 \
  hydra.job.chdir=false \
  hydra.run.dir=$LOGDIR \
  constraint=multi_class \
  task=colon_bowel \
  experiment.version=3.3 \
  training.accumulate_grad_steps=1 \
  evaluation.validation_interval=20 \
  diffusion.condition_drop_prob=0.0 \
  diffusion.guidance_scale=1.0 \
  > $LOGFILE 2>&1 & echo \$! > $PIDFILE"

echo "🚀 Training started — logs: $LOGFILE"
echo "📄 Main PID saved to: $PIDFILE"
echo "🔍 View logs: tail -f $LOGFILE"
echo "🛑 Kill training and all subprocesses: /home/yb107/cvpr2025/DukeDiffSeg/scripts/kill_training.sh"
echo "🎮 Check GPU usage: nvidia-smi"

