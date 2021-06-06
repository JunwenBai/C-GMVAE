python main.py --data_dir data/mirflickr/mirflickr_data.npy --train_idx data/mirflickr/mirflickr_train_idx.npy --valid_idx data/mirflickr/mirflickr_val_idx.npy --test_idx data/mirflickr/mirflickr_test_idx.npy --learning_rate 0.002 \
    --max_epoch 90 --label_dim 38 --z_dim 38 --feature_dim 1000 --nll_coeff 0.5 --l2_coeff 1.0 --c_coeff 0. --batch_size 128 --write_to_test_sh --dataset mirflickr \
    --lr_decay_ratio 0.5 --lr_decay_times 4. --check_freq 40 --keep_prob 0.5 --mode train --emb_size 2048 --reg gmvae --seed 1 \
    --train_ratio 1.0 --T0 50 --eta_min 2e-4 --T_mult 2
