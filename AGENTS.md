# Repo Guide

This repository contains **Atropos**, Nous Research's RL gym and library for controlling language model rollouts. It exposes an API server and a collection of microservice environments used to generate and score trajectories.

## Layout

- `atroposlib/` – core library with the API server, environment base classes, and utilities.
- `environments/` – ready‑to‑use RL environments.
- `example_trainer/` – reference trainer showing how to integrate with the API.
- `testing/` – testing helpers and documentation.

Important documents include [`CONFIG.md`](CONFIG.md) for configuration options, [`atroposlib/envs/README.md`](atroposlib/envs/README.md) for building custom environments, and [`environments/README.md`](environments/README.md) for details about included environments. Development guidelines live in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Development

1. Create a Python 3.10+ virtual environment.
2. Install dependencies in editable mode:
   ```bash
   pip install -e .[dev]
   ```
3. Install pre‑commit hooks:
   ```bash
   pre-commit install
   ```
4. Run tests with `pytest` from the repo root. Lint and formatting checks can be run with:
   ```bash
   pre-commit run --all-files
   ```

## Usage

- Launch the API server with `run-api`.
- Start an environment microservice, e.g. `python environments/gsm8k_server.py serve --slurm false`.
- Trainers fetch batches via the API; see [`example_trainer/`](example_trainer/) for a GRPO example.

Refer back to this file for general guidelines when making changes or adding new environments.
