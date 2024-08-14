export OPENAI_LOGDIR="./result"

MODEL_FLAGS="--data_dir data_path --lr 1e-4 --weight_decay 0.05 --save_interval 10000 --batch_size 8 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --image_size 128 --learn_sigma False --noise_schedule cosine --num_channels 192 --num_head_channels 64 --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --predict_xstart True" 

python scripts/mrdiff_train.py $MODEL_FLAGS
