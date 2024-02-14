#!/bin/bash

DATADIR="data/in2n/bear" #
DSF=2 # 4 

# Train a NeRFacto model
# Uncomment in stage 1
#CUDA_VISIBLE_DEVICES=2 ns-train nerfacto --data $DATADIR  nerfstudio-data --downscale-factor $DSF

# Run IN2N
MODEL_DIR="outputs/bear/nerfacto/2024-02-10_001905/nerfstudio_models/"
PROMPT="Turn the statue into a grizzly bear" #
DSF=2
IGS=1.5

# Run IN2N
# Uncomment in stage 2
CUDA_VISIBLE_DEVICES=2 ns-train in2n --data $DATADIR --mixed-precision False --load-dir $MODEL_DIR --pipeline.prompt {"$PROMPT"} --pipeline.guidance-scale 7.5 --pipeline.image-guidance-scale $IGS --pipeline.ip2p-use-full-precision True nerfstudio-data --downscale-factor $DSF



#
