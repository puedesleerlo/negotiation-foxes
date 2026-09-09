"""Check the model configuration before running agents.

Usage:  python scripts/check_llm.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.llm import CredentialsMissing, LLMClient, load_env  # noqa: E402


def main() -> None:
    env = load_env()
    print("== .env ==")
    for key in ("LLM_API_KEY", "LLM_ENDPOINT", "LLM_MODEL"):
        value = env.get(key)
        if not value:
            print(f"  {key:14} MISSING")
        elif key == "LLM_API_KEY":
            print(f"  {key:14} present ({len(value)} characters, ends in …{value[-4:]})")
        else:
            print(f"  {key:14} {value}")

    client = LLMClient()
    if not client.configured:
        print("\nCannot continue without complete credentials. Fill in .env with:")
        print("  LLM_API_KEY=...\n  LLM_ENDPOINT=https://.../v1\n  LLM_MODEL=kimi-k3")
        raise SystemExit(1)

    print(f"\n== models available at {client.endpoint} ==")
    try:
        models = client.list_models()
        for model_id in models[:25]:
            mark = " <- configured" if model_id == client.model else ""
            print(f"  {model_id}{mark}")
        if client.model not in models:
            print(f"\n  WARNING: LLM_MODEL='{client.model}' is not in the list.")
    except Exception as exc:
        print(f"  could not list models ({type(exc).__name__}: {exc}); "
              f"this may be normal depending on the provider")

    print(f"\n== test call to {client.model} ==")
    try:
        resp = client.complete(
            [{"role": "user", "content": "Reply with exactly the word: ready"}],
            max_tokens=64, temperature=0.0)
        print(f"  reply: {resp.text.strip()[:80]!r}")
        print(f"  tokens: {resp.usage.prompt_tokens} input + "
              f"{resp.usage.completion_tokens} output · {resp.usage.seconds:.2f} s")
        if not client.supports_temperature:
            print("  note: this model does not accept a fixed temperature; the client omits it.")
            print("        Consequence for reproducibility: the model's outputs are NOT")
            print("        deterministic, so the trace stores the raw reply of every call in")
            print("        order to rebuild a program from what the model actually said.")
        print("\nReady to run agents.")
    except CredentialsMissing as exc:
        print(f"  {exc}")
        raise SystemExit(1)
    except Exception as exc:
        print(f"  failed: {type(exc).__name__}: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
