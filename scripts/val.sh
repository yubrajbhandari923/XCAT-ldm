#!/bin/bash
cd /home/yb107/cvpr2025/DukeDiffSeg

SUFFIX="_diffunet"

LOGDIR="/home/yb107/logs"
LOGFILE="/home/yb107/logs/val${SUFFIX}.log"
PIDFILE="/home/yb107/logs/val${SUFFIX}.pid"

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
# pipenv run bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=8 setsid nohup python -m inference.ldm_1_0 \
#   training.num_gpus=8 \
#   training.inference_mode=True \
#   evaluation.stage1.validation_only=true \
#   evaluation.stage1.save_outputs.enabled=true \
#   training.save_config_yaml=False \
#   hydra.job.chdir=False \
#   constraint=multi_class \
#   task=colon_bowel \
#   experiment.version=1.2 \
#   evaluation.validation_max_num_samples=80 \
#   stage1.resume.path=/home/yb107/cvpr2025/DukeDiffSeg/outputs/ldm-multi_class-colon_bowel/1.2/checkpoints/stage1/LDM-multi_class-colon_bowel_stage1_best_checkpoint_80_Mean_Dice0.6824.pt \
#   data.cache_dir=/data/usr/yb107/colon_data/cache_tmp \
#   > $LOGFILE 2>&1 & echo \$! > $PIDFILE"

# create a symlink for /home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-multi_class_no_colon-colon/4.0/train_script.py to  /home/yb107/cvpr2025/DukeDiffSeg/inference dir

# ln -s /home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-multi_class_no_colon-colon/4.2/train_script.py /home/yb107/cvpr2025/DukeDiffSeg/inference/diffunet_4_2.py

# ln -s /home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-binary-colon/4.6/train_script.py /home/yb107/cvpr2025/DukeDiffSeg/inference/diffunet_4_6.py
  # data.cache_dir=/data/usr/yb107/colon_data/cache_mobina_mixed_colon_dataset_alll \

pipenv run bash -c "CUDA_VISIBLE_DEVICES=1,2,3,5 OMP_NUM_THREADS=8 setsid nohup python -m train.diffunet_4_0 \
  training.num_gpus=4 \
  training.inference_mode=True \
  evaluation.save_outputs.enabled=True \
  evaluation.save_outputs.save_inputs=False \
  evaluation.save_outputs.dir_postfix=c_grade_550_gs_2.0_final_small_with_skeletonization \
  experiment.debug=False \
  data.save_data=False \
  data.batch_size_per_gpu=1 \
  experiment.version=5.1 \
  diffusion.guidance_scale=2.0 \
  data.val_jsonl=/home/yb107/cvpr2025/DukeDiffSeg/data/c_grade_colons/3d_vlsmv2_c_grade_colon_dataset_with_body_filled_small.jsonl\
  training.resume.path=/home/yb107/cvpr2025/DukeDiffSeg/outputs/diffunet-binary-colon/5.1/checkpoints/training/DiffUnet-binary-colon_training_latest_checkpoint_550.pt \
  hydra.run.dir=$LOGDIR \
  > $LOGFILE 2>&1 & echo \$! > $PIDFILE"



echo "🚀 Inference started — logs: $LOGFILE"
echo "📄 Main PID saved to: $PIDFILE"
echo "🔍 View logs: tail -f $LOGFILE"
echo "🛑 Kill inference and all subprocesses: /home/yb107/cvpr2025/DukeDiffSeg/scripts/kill_inference.sh"
echo "🎮 Check GPU usage: nvidia-smi"

