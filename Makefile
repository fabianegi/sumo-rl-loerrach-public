# Optional Unix convenience wrapper around run.py.
# run.py is the canonical, OS-agnostic interface - these targets just mirror it.
# Usage: make <target>   (e.g. `make demo`). Override PY to pick an interpreter:
#   make demo PY=python3.11

PY ?= python

.PHONY: help check list demo compare train evaluate setup

help:
	@echo "Targets: check | list | demo | compare | train | evaluate | setup"
	@echo "All targets call: \$(PY) run.py <command>. See README → Bedienung."

check:
	$(PY) run.py check

list:
	$(PY) run.py list

demo:
	$(PY) run.py demo

compare:
	$(PY) run.py compare

# Defaults match run.py; override on the command line, e.g.:
#   make train ARGS="--algo ppo --intersection osm --timesteps 5000"
train:
	$(PY) run.py train $(ARGS)

evaluate:
	$(PY) run.py evaluate $(ARGS)

setup:
	$(PY) -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
