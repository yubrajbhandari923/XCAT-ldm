#!/bin/bash
cd /home/yb107/cvpr2025/DukeDiffSeg

SUFFIX="_diffunet5_0"

LOGDIR="/home/yb107/logs"
LOGFILE="/home/yb107/logs/train${SUFFIX}.log"
PIDFILE="/home/yb107/logs/train${SUFFIX}.pid"


# Clean up any old PID file
if [ -f "$PIDFILE" ]; then
    rm "$PIDFILE"
fi

# python -m train.segdiff_1_0 --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/segdiff.yaml
# pipenv run bash -c "CUDA_VISIBLE_DEVICES=3,4,5,6 OMP_NUM_THREADS=8 nohup torchrun --nproc_per_node=4 -m train.medsegdiff_1_0 \
#   --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/medsegdiff.yaml > /home/yb107/logs/train_medsegdiff.log 2>&1 &"
# pipenv run bash -c "CUDA_VISIBLE_DEVICES=3,4,5,6 OMP_NUM_THREADS=8 nohup python -m train.medsegdiff_1_0 \
#   --exp_config /home/yb107/cvpr2025/DukeDiffSeg/configs/experiments/medsegdiff.yaml > /home/yb107/logs/train_medsegdiff.log 2>&1 &"

  # training.resume.path=/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-multi_class_no_colon-colon/4.0/checkpoints/training/DiffUnet-multi_class_no_colon-colon_training_latest_checkpoint_375.pt \
  # training.resume.path=/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-binary-colon/4.16/checkpoints/training/DiffUnet-binary-colon_training_best_checkpoint_580_MeanDice0.4582.pt \
  # training.resume.path=/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-binary-colon/5.0/checkpoints/final_unet.pth \

# Launch in new process group with setsid
pipenv run bash -c "CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 OMP_NUM_THREADS=8 setsid nohup python -m train.diffunet_4_0 \
  training.num_gpus=7 \
  training.inference_mode=False \
  model.adverserial_train.enabled=False \
  evaluation.save_outputs.enabled=False \
  evaluation.validation_interval=5 \
  experiment.debug=False \
  data.save_data=False \
  training.resume.path=/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-binary-colon/5.0/checkpoints/training/DiffUnet-binary-colon_training_latest_checkpoint_1000.pt \
  data.batch_size_per_gpu=4 \
  data.val_batch_size=1 \
  hydra.job.chdir=false \
  hydra.run.dir=$LOGDIR \
  > $LOGFILE 2>&1 & echo \$! > $PIDFILE"

echo "🚀 Training started — logs: $LOGFILE"
echo "📄 Main PID saved to: $PIDFILE"
echo "🔍 View logs: tail -f $LOGFILE"
echo "🛑 Kill training and all subprocesses: /home/yb107/cvpr2025/DukeDiffSeg/scripts/kill.sh"
echo "🎮 Check GPU usage: nvidia-smi"

