"""Annotate verifier-baseline results with the provider that served each call.

OpenRouter routes every request for one model to one of many providers,
which run it at different precisions. Knowing which provider answered each
scenario lets a run's errors be traced to a provider rather than to the
model. The lookup must run after the calls, from the generation ids the
baseline records.

    python -m evaluation.openrouter_generation_info results.json [...]
"""

import json
import os
import sys
import time
import urllib.request

from dotenv import load_dotenv

load_dotenv(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 ".env"),
    override=True,
)


def generation_info(generation_id: str) -> dict:
    request = urllib.request.Request(
        f"https://openrouter.ai/api/v1/generation?id={generation_id}",
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"},
    )
    return json.load(urllib.request.urlopen(request))["data"]


def annotate(path: str) -> None:
    with open(path, encoding="utf-8") as f:
        results = json.load(f)
    for scenario in results["scenarios"]:
        generation_id = scenario.get("generation_id")
        if not generation_id or scenario.get("provider_name"):
            continue
        # The generation record appears a few seconds after the call.
        for attempt in range(3):
            try:
                data = generation_info(generation_id)
                scenario["provider_name"] = data.get("provider_name")
                scenario["reasoning_tokens"] = data.get(
                    "native_tokens_reasoning")
                break
            except Exception:
                time.sleep(2 * (attempt + 1))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    served = [s.get("provider_name") for s in results["scenarios"]]
    failed = [(s["id"], s.get("provider_name"))
              for s in results["scenarios"] if not s["pass"]]
    print(f"{path}: {len(set(served))} providers, failures: {failed}")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        annotate(path)
