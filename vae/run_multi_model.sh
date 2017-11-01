#!/bin/bash

echo "VAE type is: $1"
echo "Latent dim is: $2"

for NUM in `seq 0 1 9`
do
  python train_multi_vae.py --index $NUM --vae_type=$1 --latent_dim=$2
done
