#!/bin/bash

echo "Names: model_multi_$1"
echo "VAE type is: $1"
echo "Latent dim is: $2"
echo "# of epochs is: $3"

for NUM in `seq 0 1 9`
do
  python train_multi_vae.py --name model_multi_$1 --index $NUM --vae_type=$1 --latent_dim=$2 --n_epochs=$3
done
