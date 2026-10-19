#!/bin/zsh
# Temporary public link to the workbench, served from this laptop (Gradio share link).
# Usage: deploy/share_local.sh [data folders under ui/]   (default: data_bach data_breakhis, with a dataset switch)
# Every visitor gets their own review queue (--session-decisions); local decisions.json is untouched.
# The link stops working when this script stops (Ctrl+C), the laptop sleeps, or the link expires.
DATA=("${@:-data_bach data_breakhis}")
cd "$(dirname "$0")/../ui" || exit 1
PYTHONUNBUFFERED=1 exec .venv/bin/python main.py --data ${=DATA} --live --session-decisions --share
