#!/bin/bash
#source activate
MESHFILES=../../mesh_files
# DATAFILE=mnist_ico5.zip
DATAFILE=mnist_ico4_nn.zip

# assert mesh files exist
if [ ! -d $MESHFILES ]; then
    echo "[!] Mesh files do not exist..."
    exit
fi

# generate data
if [ ! -f $DATAFILE ]; then
    echo "[!] Data files do not exist. Preparing data..."
    python prepare_data.py --bandwidth 60 --no_rotate_train ---no_rotate_test --mnist_data_folder raw_data --direction NP --output_file $DATAFILE  --mesh_file $MESHFILES/icosphere_4.pkl
fi

# train
python train.py --mesh_folder $MESHFILES --datafile $DATAFILE --log_dir log_nn --optim adam --lr 1e-2 --epochs 50 --feat 16 --decay --batch-size 32
