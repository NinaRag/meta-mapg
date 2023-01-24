#!/bin/bash

# Load python virtualenv
#virtualenv venv
#source venv/bin/activate
#pip3 install -r requirements.txt

# Set GPU device ID
export CUDA_VISIBLE_DEVICES=-1

# For MuJoCo
# Please note that below MuJoCo and GLEW path may differ
# depends on a computer setting
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so

# Begin experiment
python3 main.py \
--seed 42 \
--config "ccf_pd.yaml" \
--opponent-shaping \
--prefix "" \
"$@"

python3 main.py \
--seed 42 \
--config "ccf_pd.yaml" \
--opponent-shaping \
--test-mode \
--prefix "test" \
"$@"

