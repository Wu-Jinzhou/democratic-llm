This folder contains the PRISM 8B soft-panel weight-transform experiment.

Evaluated model set:
- `checkpoints/llama3.1-8b-soft-panel`
- `checkpoints/llama3.1-8b-soft-panel-sqrt`
- `checkpoints/llama3.1-8b-soft-panel-square`
- `checkpoints/llama3.1-8b-soft-panel-clipped`
- `checkpoints/llama3.1-8b-full-prism`
- `checkpoints/llama3.1-8b-hard-panel`

The default training budget matches the paper:
- `NUM_TRAIN_STEPS=3538`

Main entrypoint:
- `runs/experiments/soft_weighting/run_everything.sh`
