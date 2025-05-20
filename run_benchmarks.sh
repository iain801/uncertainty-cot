#! /bin/bash

export CUDA_VISIBLE_DEVICES=5

export THRESHOLDS=(1e-1 15e-2 2e-1 25e-2 3e-1 35e-2 4e-1 45e-2 5e-1)
export WINDOW_SIZES=(2 5 10)

echo "Running benchmark with threshold 0.0 (control)"
python benchmark.py --threshold 0.0

for THRESHOLD in ${THRESHOLDS[@]}; do
    for WINDOW_SIZE in ${WINDOW_SIZES[@]}; do
        echo "Running benchmark with threshold $THRESHOLD and window size $WINDOW_SIZE"
        python benchmark.py --threshold $THRESHOLD --window $WINDOW_SIZE --rolling
    done
done