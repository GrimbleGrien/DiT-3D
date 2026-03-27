python train.py --gpu 0 \
    --dataroot ../../PSF/data/ShapeNetCore.v2.PC15k/ \
    --category chair \
    --experiment_name 'mae500' \
    --model_type 'DiT-S/4' \
    --bs 16 \
    --voxel_size 32 \
    --lr 1e-4 \
    --use_tb \
    --niter 500 \
    --saveIter 1 \
    --use_mae \
    --mae_config_path 'configs/pretrainMAE.yaml' \
    --mae_points 1024 \
    --mae_mask_ratio 0.6 \
    --vizIter 100 \
    # --model 'checkpoints/mae1000/latest.pth'