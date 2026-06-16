#!/bin/bash

set -e

py.exe -m pip install -r req.txt

echo "Generate flow..."
# py.exe generate_flow.py

MODELS=("IFRNet" "IFRNet_RGB2")

for MODEL in "${MODELS[@]}"; do

    echo "Starting experiment for: $MODEL"

    py.exe train.py --model_name "$MODEL" --epochs 100 --batch_size 2 --lr_start 1e-4 --lr_end 1e-5

    echo "  Benchmarking $MODEL..."
    py.exe benchmarks/speed_parameters.py --model_name "$MODEL"> "./checkpoint/${MODEL}/speed_parameters.txt"

    echo "  Evaluating $MODEL..."
    py.exe benchmarks/eval_vimeo90k.py --model_name "$MODEL" > "./checkpoint/${MODEL}/eval.txt"

    echo "Finished experiment for: $MODEL"
done

echo "All experiments completed successfully!"