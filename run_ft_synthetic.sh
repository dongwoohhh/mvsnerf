
eval_dataset=nerf_synthetic

#for name in "scan8_train" "scan21_train" "scan30_train" "scan31_train" "scan34_train" "scan38_train" "scan40_train" "scan41_train"; do
#for name in "scan8" "scan21" "scan30" "scan31" "scan34" "scan38" "scan40" "scan41"; do
#for name in "fern" "flower" "fortress" "horns" "leaves" "orchids" "room" "trex"; do
for name in "lego" "chair" "drums" "ficus" "hotdog" "materials" "mic" "ship"; do
    eval_scenes=${name}
    echo $eval_scenes
    
    python train_mvs_nerf_finetuning_pl.py --dataset_name blender --datadir /media/hdd1/Datasets/MVSNeRF_data/mvs_training/${eval_dataset}/${name} \
        --expname ft_${name} --with_rgb_loss --batch_size 1024 --num_epochs 1 --imgScale_test 1.0 --white_bkgd --pad 0 --ckpt ./ckpts/mvsnerf-v0.tar --N_vis 1

done
