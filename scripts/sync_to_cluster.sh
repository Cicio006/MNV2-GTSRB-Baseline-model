#!/bin/bash

REMOTE_HOST="headnode"
REMOTE_DIR="~/cutsc"

rsync -av \
  --exclude ".git" \
  --exclude "__pycache__" \
  --exclude "*.pyc" \
  --exclude ".venv" \
  --exclude "venv" \
  --exclude ".DS_Store" \
  ./ "$REMOTE_HOST:$REMOTE_DIR/"
