# Screen Capture Environment Usage Guide

This guide explains how to run the `screen_capture_env.py` environment to train or evaluate models on describing screen recordings.

## Prerequisites

1. Install the repository dependencies in editable mode:
   ```bash
   pip install -e .[dev]
   ```
2. Copy `.env.example` to `.env` and set your API key variables, e.g.:
   ```bash
   export OPENAI_API_KEY=YOUR_KEY
   ```
   The environment uses an OpenAI API compatible server. For local servers without authentication you can set `api_key` to a dummy value like `"x"`.

## Running Locally

Start the API server and launch the environment in two terminals:

```bash
run-api
```
```bash
python environments/screen_capture_env.py serve --slurm false
```

The default configuration in `screen_capture_env.py` expects an inference server at `http://localhost:9004/v1` and will send requests using the API key specified in the environment variable or the value provided in `config_init()`.

## Customising the Server

Edit the `config_init` class method if you need to change model settings:

```python
server_configs = [
    APIServerConfig(
        model_name="gemini-2.5-pro",
        base_url="http://localhost:9004/v1",
        api_key=os.environ.get("OPENAI_API_KEY"),
    )
]
```

You can also override these values from the command line:

```bash
python environments/screen_capture_env.py serve \
  --model_name gemini-2.5-pro \
  --base_url https://api.openai.com/v1 \
  --api_key $OPENAI_API_KEY
```

## Offline Data Generation

For quick testing without a trainer, you can use the `process` subcommand to save rollout groups and an HTML preview:

```bash
python environments/screen_capture_env.py process \
  --env.data_path_to_save_groups screen_capture.jsonl
```

## Next Steps

Once rollouts are generated, you can train a model using the example GRPO trainer in `example_trainer/`.

