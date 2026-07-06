#!/bin/bash

REMOTE_HOST="headnode"
REMOTE_DIR="~/cutsc"

echo "Syncing files to cluster..."
rsync -av \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude ".venv" \
  --exclude "venv" \
  --exclude ".DS_Store" \
  ./ "$REMOTE_HOST:$REMOTE_DIR/"

echo "Submitting Slurm job..."
ssh "$REMOTE_HOST" "cd $REMOTE_DIR && sbatch scripts/run_wp1_check.sh"
