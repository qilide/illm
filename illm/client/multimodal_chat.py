import argparse
import base64
import json
from typing import Any, Dict, List, Optional

import requests


def parse_args():
    p = argparse.ArgumentParser(description="Multimodal Chat (image + text) demo client")
    p.add_argument("--host", type=str, default="http://127.0.0.1", help="Server host")
    p.add_argument("--port", type=int, default=8000, help="Server port")
    p.add_argument("--framework", type=str, default="ollama", help="Backend framework (ollama recommended)")
    p.add_argument("--model", type=str, default="llava:7b", help="Vision model name (e.g., llava / minicpm-v)")
    p.add_argument("--text", type=str, default="Describe this image.", help="Text prompt")
    p.add_argument("--image_url", type=str, default=None, help="Image URL")
    p.add_argument("--image_path", type=str, default=None, help="Local image path to base64 encode")
    p.add_argument("--stream", action="store_true", help="Enable streaming")
    return p.parse_args()


def load_b64_from_path(path: str) -> Optional[str]:
    try:
        with open(path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception:
        return None


def main():
    args = parse_args()
    url = f"{args.host}:{args.port}/v1/chat/completions"

    parts: List[Dict[str, Any]] = []
    if args.text:
        parts.append({"type": "text", "text": args.text})

    if args.image_url:
        parts.append({"type": "image_url", "image_url": args.image_url})

    if args.image_path:
        b64 = load_b64_from_path(args.image_path)
        if b64:
            parts.append({"type": "image_base64", "image_base64": b64})

    if not parts:
        print("No content provided. Use --text and/or --image_url/--image_path.")
        return

    body: Dict[str, Any] = {
        "framework": args.framework,
        "model": args.model,
        "messages": [
            {"role": "user", "content": parts}
        ],
        "stream": args.stream,
    }

    if not args.stream:
        resp = requests.post(url, json=body, timeout=300)
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