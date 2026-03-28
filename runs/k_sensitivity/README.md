This folder contains the PRISM 8B panel-size sensitivity experiment.

It trains and evaluates:
- Hard trio: original `hard-panel`, `hard-panel-control-k50`, `hard-panel-control-k100`
- Soft trio: original `soft-panel`, `soft-panel-control-k50`, `soft-panel-control-k100`

Main entrypoint:
- `runs/experiments/k_sensitivity/run_everything.sh`

Individual scripts:
- `train_hard_k_50.sh`
- `train_hard_k_100.sh`
- `train_soft_k_50.sh`
- `train_soft_k_100.sh`
- `generate_hard.sh`
- `judge_hard.sh`
- `score_hard.sh`
- `generate_soft.sh`
- `judge_soft.sh`
- `score_soft.sh`
