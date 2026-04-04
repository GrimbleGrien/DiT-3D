python test.py \
    --model checkpoints/jmae500/best.pth \
    --dataroot ../../PSF/data/ShapeNetCore.v2.PC15k/ \
    --category chair \
    --model_type DiT-S/4 \
    --bs 16 \
    --eval_unconditional \
    --experiment_name dit3d_unconditional
