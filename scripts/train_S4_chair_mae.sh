# python train.py --distribution_type 'multi' \
python train.py --dataroot ../../PSF/data/ShapeNetCore.v2.PC15k/ \
    --category chair \
    --experiment_name ../experiments/ \
    --model_type 'DiT-S/4' \
    --bs 16 \
    --voxel_size 32 \
    --lr 1e-4 \
    --use_tb \
    --niter 2 \
    --use_mae \
    --mae_config_path 'configs/pretrainMAE.yaml'
