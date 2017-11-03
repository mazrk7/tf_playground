#!/bin/bash

echo "Names: multi_$1_ldim$2"
echo "VAE type is: $1"
echo "Latent dim is: $2"
echo "# of epochs is: $3"

for NUM in `seq 0 1 9`
do
  python train_multi_vae.py --name multi_$1_ldim$2 --index $NUM --vae_type=$1 --latent_dim=$2 --n_epochs=$3
done
