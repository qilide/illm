import argparse
import json
import sys
import time
from typing import Any, Dict, List, Optional

import requests


def parse_args():
    p = argparse.ArgumentParser(description="Chat Completions demo client")
    p.add_argument("--host", type=str, default="http://127.0.0.1", help="Server host")
    p.add_argument("--port", type=int, default=8000, help="Server port")
    p.add_argument("--framework", type=str, default="transformers", help="Backend framework")
    p.add_argument("--model", type=str, default="sshleifer/tiny-gpt2", help="Model name or id")
    p.add_argument("--prompt", type=str, default="Hello", help="Prompt text")
    p.add_argument("--stream", action="store_true", help="Enable streaming")
    p.add_argument("--max_tokens", type=int, default=32)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    url = f"{args.host}:{args.port}/v1/chat/completions"
    body: Dict[str, Any] = {
        "framework": args.framework,
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": args.stream,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
    }

    if not args.stream:
        resp = requests.post(url, json=body, timeout=120)
        resp.raise_for_status()
        print(json.dumps(resp.json(), ensure_ascii=False, indent=2))
        return

    with requests.post(url, json=body, stream=True) as r:
        r.raise_for_status()
        print("Streaming chunks:")
        for line in r.iter_lines(decode_unicode=True):
            if not line:
                continue
            if line.startswith("data: "):
                data = line[len("data: ") :]
                if data == "[DONE]":
                    print("\n[STREAM DONE]")
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    obj = {"raw": data}
                print(json.dumps(obj, ensure_ascii=False))
        print("Finished.")


if __name__ == "__main__":
    main()