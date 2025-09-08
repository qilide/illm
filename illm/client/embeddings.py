import argparse
import json
from typing import Any, Dict, List, Union

import requests


def parse_args():
    p = argparse.ArgumentParser(description="Embeddings demo client")
    p.add_argument("--host", type=str, default="http://127.0.0.1", help="Server host")
    p.add_argument("--port", type=int, default=8000, help="Server port")
    p.add_argument("--framework", type=str, default="transformers", help="Backend framework")
    p.add_argument("--model", type=str, default="sshleifer/tiny-gpt2", help="Model name for generation (not required for embeddings)")
    p.add_argument("--embedding_model", type=str, default="sshleifer/tiny-distilbert-base-cased", help="Embedding model id")
    p.add_argument("--inputs", type=str, nargs="+", default=["hello", "world"], help="Texts to embed")
    return p.parse_args()


def main():
    args = parse_args()
    url = f"{args.host}:{args.port}/v1/embeddings"
    body: Dict[str, Any] = {
        "framework": args.framework,
        "model": args.model,
        "embedding_model": args.embedding_model,
        "input": args.inputs,
    }
    resp = requests.post(url, json=body, timeout=120)
    resp.raise_for_status()
    print(json.dumps(resp.json(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()