#!/bin/bash
conda activate myenv  # Replace 'myenv' with your env name

# Create logs directory if it doesn't exist
mkdir -p logs

# Run for each environment, saving both print output and plots
for env in HalfCheetah-v5 Ant-v5 Humanoid-v5; do
    echo "Starting training for $env"
    python eval_trpo.py --env $env > logs/${env}_train.log 2>&1
    echo "Finished training for $env"
done