"""
Run the Trajectory API server.
"""

import argparse

import uvicorn


def main(host: str, port: int, reload: bool):
    """
    Run the API server.
    Args:
        host: The host to run the API server on.
        port: The port to run the API server on.
        reload: Whether to reload the API server on code changes.
    """
    uvicorn.run("atroposlib.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", type=bool, default=False)
    args = parser.parse_args()
    main(args.host, args.port, args.reload)
