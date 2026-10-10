"""Canonicalizer stress test: 50 informal activity names mapped to
canonical actions by the LLM, compared against ground truth.

This evaluates the accuracy of the LLM-based Action Canonicalizer
(Section 5.2) on a diverse set of informal, colloquial, and noisy
patient utterances. The symbolic layer's fail-closed mechanism
(unknown -> blocked) means that a wrong mapping is worse than no
mapping, so we track both accuracy and "unknown rate".

Metrics:
  - accuracy: fraction of activities mapped to the correct canonical action
  - unknown_rate: fraction mapped to "unknown" (fail-closed)
  - wrong_known_rate: fraction mapped to a valid but wrong action
  - per-category breakdown

Output: canonicalizer_stress_test_results.json
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
load_dotenv(dotenv_path, override=True)

from planning.canonical_actions import CANONICAL_ACTIONS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from utils.models import define_model
# The node modules and graph_init import each other, so the graph has to be
# imported first for the node import below to resolve.
import graph.graph_init  # noqa: F401
from nodes.action_canonicalizer_node import (
    INITIAL_PROMPT as NODE_PROMPT,
    _catalogue as node_catalogue,
    _parse_batch as parse_node_answer,
)


# ------------------------------------------------------------------
# 50 informal activity names with ground-truth canonical action.
# Categories:
#   - exact_synonym: close synonym of the canonical action name
#   - colloquial: everyday patient speech
#   - paraphrase: longer description
#   - noisy: contains typos, extra words, or ambiguity
#   - out_of_scope: no matching canonical action (expected: unknown)
# ------------------------------------------------------------------

STRESS_TEST_ITEMS = [
    # --- exact_synonym (10) ---
    {"input": "Take my pills",              "expected": "take_medication",       "category": "exact_synonym"},
    {"input": "Take antibiotics",           "expected": "take_antibiotic",       "category": "exact_synonym"},
    {"input": "Drink a glass of water",     "expected": "drink_water",           "category": "exact_synonym"},
    {"input": "Eat my meal",                "expected": "eat_meal",              "category": "exact_synonym"},
    {"input": "Have lunch",                 "expected": "have_lunch",            "category": "exact_synonym"},
    {"input": "Do some exercise",           "expected": "exercise",              "category": "exact_synonym"},
    {"input": "Read a book",                "expected": "reading_activity",      "category": "exact_synonym"},
    {"input": "Rest for a while",           "expected": "rest",                  "category": "exact_synonym"},
    {"input": "Check my vitals",            "expected": "check_vitals",          "category": "exact_synonym"},
    {"input": "Walk in the park",           "expected": "walk_in_park",          "category": "exact_synonym"},

    # --- colloquial (10) ---
    {"input": "I need to pop my pills",     "expected": "take_medication",       "category": "colloquial"},
    {"input": "Time for my meds",           "expected": "take_medication",       "category": "colloquial"},
    {"input": "Gotta take my antibiotics",  "expected": "take_antibiotic",       "category": "colloquial"},
    {"input": "I'm thirsty, need water",    "expected": "drink_water",           "category": "colloquial"},
    {"input": "Let's grab some food",       "expected": "eat_meal",              "category": "colloquial"},
    {"input": "Time to eat lunch",          "expected": "have_lunch",            "category": "colloquial"},
    {"input": "I wanna work out",           "expected": "exercise",              "category": "colloquial"},
    {"input": "Call my daughter on phone",  "expected": "call_relative",         "category": "colloquial"},
    {"input": "Video chat with my son",     "expected": "video_call_relative",   "category": "colloquial"},
    {"input": "I want to talk to the nurse","expected": "speak_with_nurse",      "category": "colloquial"},

    # --- paraphrase (10) ---
    {"input": "I should swallow my morning medicine",          "expected": "take_medication",       "category": "paraphrase"},
    {"input": "I need to take the antibiotic the doctor gave", "expected": "take_antibiotic",       "category": "paraphrase"},
    {"input": "Please bring me something to drink",            "expected": "drink_water",           "category": "paraphrase"},
    {"input": "I would like to have my dinner now",            "expected": "eat_meal",              "category": "paraphrase"},
    {"input": "Let's do the midday meal",                      "expected": "have_lunch",            "category": "paraphrase"},
    {"input": "I feel like doing some physical activity",      "expected": "exercise",              "category": "paraphrase"},
    {"input": "I'd love to read something for a while",        "expected": "reading_activity",      "category": "paraphrase"},
    {"input": "I need to lie down and rest my legs",           "expected": "rest",                  "category": "paraphrase"},
    {"input": "Can someone measure my blood pressure",         "expected": "check_vitals",          "category": "paraphrase"},
    {"input": "I want to go for a stroll outside",             "expected": "walk_in_park",          "category": "paraphrase"},

    # --- noisy (10) ---
    {"input": "Take my pillz",                                "expected": "take_medication",       "category": "noisy"},
    {"input": "antibiotcs",                                   "expected": "take_antibiotic",       "category": "noisy"},
    {"input": "drink wter",                                   "expected": "drink_water",           "category": "noisy"},
    {"input": "eat food meal please now",                     "expected": "eat_meal",              "category": "noisy"},
    {"input": "have lunchtime meal",                          "expected": "have_lunch",            "category": "noisy"},
    {"input": "do exercise workout fitness",                  "expected": "exercise",              "category": "noisy"},
    {"input": "read book reading novel",                      "expected": "reading_activity",      "category": "noisy"},
    {"input": "rest sleep nap tired",                         "expected": "rest",                  "category": "noisy"},
    {"input": "check blood pressure heart rate vitals",       "expected": "check_vitals",          "category": "noisy"},
    {"input": "walk park outdoor stroll nature",              "expected": "walk_in_park",          "category": "noisy"},

    # --- out_of_scope (10) ---
    {"input": "I want to watch TV",                           "expected": "unknown",               "category": "out_of_scope"},
    {"input": "Turn on the radio",                            "expected": "unknown",               "category": "out_of_scope"},
    {"input": "I want to play chess",                         "expected": "unknown",               "category": "out_of_scope"},
    {"input": "Can you open the window",                      "expected": "unknown",               "category": "out_of_scope"},
    {"input": "I want to go shopping",                        "expected": "unknown",               "category": "out_of_scope"},
    {"input": "Help me get dressed",                          "expected": "unknown",               "category": "out_of_scope"},
    {"input": "I want to paint a picture",                    "expected": "unknown",               "category": "out_of_scope"},
    {"input": "Can you call a taxi",                          "expected": "unknown",               "category": "out_of_scope"},
    {"input": "I want to knit a sweater",                     "expected": "unknown",               "category": "out_of_scope"},
    {"input": "Let me water the plants",                      "expected": "unknown",               "category": "out_of_scope"},
]


CANONICALIZER_SYSTEM_PROMPT = """\
You are an action canonicalization module.

Your task is to map a patient activity to ONE canonical action.

Canonical actions:

{canonical_actions}

Return ONLY the canonical action name. It must be a string. No List, no JSON. no explanations.
If the activity does not match any canonical action, return "unknown".
"""


def run_canonicalizer_stress_test(
    output_file: str = "evaluation/canonicalizer_stress_test_results.json",
    prompt_variant: str = "old",
    temperature: float | None = None,
) -> dict:
    """Run the canonicalizer stress test with the LLM.

    ``prompt_variant`` selects which prompt is measured: "old" is the bare
    list of action names originally evaluated, "new" is the prompt the node
    now sends, which also carries each action's description. Keeping both
    separates the effect of the prompt from the effect of the model.
    """
    model = define_model(temperature=temperature)

    if prompt_variant == "new":
        system_prompt = PromptTemplate.from_template(NODE_PROMPT).invoke(
            {"canonical_actions": node_catalogue()}
        ).to_string()
    else:
        canonical_actions_list = "\n".join(sorted(CANONICAL_ACTIONS.keys()))
        system_prompt = PromptTemplate.from_template(
            CANONICALIZER_SYSTEM_PROMPT
        ).invoke({"canonical_actions": canonical_actions_list}).to_string()

    results = []
    total_correct = 0
    total_unknown = 0
    total_wrong_known = 0
    total_latency = 0.0
    total_tokens = 0

    category_stats = {}

    for item in STRESS_TEST_ITEMS:
        input_text = item["input"]
        expected = item["expected"]
        category = item["category"]

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(
                content=f"1. {input_text}" if prompt_variant == "new"
                else input_text
            ),
        ]

        start_time = time.time()
        try:
            response = model.invoke(messages)
            latency = time.time() - start_time
            total_latency += latency

            raw_output = response.content
            if isinstance(raw_output, list):
                raw_output = raw_output[0] if raw_output else ""
            if isinstance(raw_output, dict):
                raw_output = raw_output.get("text") or raw_output.get("content") or ""
            if raw_output is None:
                raw_output = ""

            if prompt_variant == "new":
                parsed = parse_node_answer(response.content, expected=1)
                predicted = parsed[0] if parsed else "unknown"
            else:
                predicted = str(raw_output).strip()

            # Apply same fail-closed logic as the real node
            if predicted not in CANONICAL_ACTIONS and predicted != "unknown":
                predicted = "unknown"

            token_usage = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                token_usage = {
                    "prompt_tokens": response.usage_metadata.get("input_tokens", 0),
                    "completion_tokens": response.usage_metadata.get("output_tokens", 0),
                    "total_tokens": response.usage_metadata.get("total_tokens", 0),
                }
                total_tokens += token_usage.get("total_tokens", 0)

        except Exception as e:
            latency = time.time() - start_time
            total_latency += latency
            predicted = "unknown"
            raw_output = f"ERROR: {e}"
            token_usage = {}

        is_correct = (predicted == expected)
        is_unknown = (predicted == "unknown")
        is_wrong_known = (not is_correct and not is_unknown)

        if is_correct:
            total_correct += 1
        if is_unknown:
            total_unknown += 1
        if is_wrong_known:
            total_wrong_known += 1

        # Per-category stats
        if category not in category_stats:
            category_stats[category] = {"total": 0, "correct": 0, "unknown": 0, "wrong_known": 0}
        category_stats[category]["total"] += 1
        if is_correct:
            category_stats[category]["correct"] += 1
        if is_unknown:
            category_stats[category]["unknown"] += 1
        if is_wrong_known:
            category_stats[category]["wrong_known"] += 1

        results.append({
            "input": input_text,
            "expected": expected,
            "predicted": predicted,
            "correct": is_correct,
            "category": category,
            "latency_seconds": round(latency, 3),
            "token_usage": token_usage,
            "raw_output": str(raw_output)[:200],
        })

        status = "OK" if is_correct else ("UNK" if is_unknown else "WRONG")
        print(f"  [{status}] '{input_text}' -> '{predicted}' (expected: '{expected}') [{category}]")

    total = len(STRESS_TEST_ITEMS)
    accuracy = total_correct / total
    unknown_rate = total_unknown / total
    wrong_known_rate = total_wrong_known / total
    avg_latency = total_latency / total

    # Per-category accuracy
    for cat in category_stats:
        cs = category_stats[cat]
        cs["accuracy"] = round(cs["correct"] / cs["total"], 4) if cs["total"] > 0 else 0

    summary = {
        "total_items": total,
        "correct": total_correct,
        "unknown": total_unknown,
        "wrong_known": total_wrong_known,
        "accuracy": round(accuracy, 4),
        "unknown_rate": round(unknown_rate, 4),
        "wrong_known_rate": round(wrong_known_rate, 4),
        "avg_latency_seconds": round(avg_latency, 3),
        "total_latency_seconds": round(total_latency, 3),
        "total_tokens": total_tokens,
        "model": (
            os.getenv("OPENROUTER_MODEL")
            if os.getenv("LLM_PROVIDER") == "openrouter"
            else os.getenv("OPENAI_MODEL", "unknown")
        ),
        "provider": os.getenv("LLM_PROVIDER", "unknown"),
        "prompt_variant": prompt_variant,
        "category_breakdown": category_stats,
    }

    output = {
        "summary": summary,
        "items": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)

    return output


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Canonicalizer stress test")
    parser.add_argument("--prompt", choices=("old", "new"), default="old",
                        help="which canonicalizer prompt to measure")
    parser.add_argument("--temperature", type=float, default=None,
                        help="override the sampling temperature")
    parser.add_argument("--provider", default=None,
                        help="override LLM_PROVIDER for this run")
    parser.add_argument("--model", default=None,
                        help="override the model id for this run")
    parser.add_argument("--out", default=None, help="results file")
    args = parser.parse_args()

    # load_dotenv ran at import with override=True, so command-line
    # environment variables were already replaced by the .env values.
    # Applying the overrides here makes the choice explicit and final.
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.model:
        provider = os.environ.get("LLM_PROVIDER", "")
        key = ("OPENROUTER_MODEL" if provider == "openrouter"
               else "GROQ_MODEL" if provider == "groq"
               else "OLLAMA_MODEL" if provider == "ollama"
               else "OPENAI_MODEL")
        os.environ[key] = args.model

    output_file = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "canonicalizer_stress_test_results.json",
    )

    print("=" * 60)
    print("  CANONICALIZER STRESS TEST")
    print("  50 informal activity names -> canonical actions")
    print("=" * 60)
    print()

    results = run_canonicalizer_stress_test(
        output_file, prompt_variant=args.prompt, temperature=args.temperature
    )

    s = results["summary"]
    print()
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Total items:           {s['total_items']}")
    print(f"  Correct:               {s['correct']} ({s['accuracy']:.1%})")
    print(f"  Unknown (fail-closed): {s['unknown']} ({s['unknown_rate']:.1%})")
    print(f"  Wrong but known:       {s['wrong_known']} ({s['wrong_known_rate']:.1%})")
    print(f"  Avg latency:           {s['avg_latency_seconds']:.3f}s")
    print(f"  Total tokens:          {s['total_tokens']}")
    print(f"  Model:                 {s['model']}")
    print()
    print("  Per category:")
    for cat, cs in s["category_breakdown"].items():
        print(f"    {cat:20s} {cs['correct']}/{cs['total']} ({cs['accuracy']:.0%}) "
              f"[unknown: {cs['unknown']}, wrong_known: {cs['wrong_known']}]")
    print()
    print(f"  Results saved to: {output_file}")
    print("=" * 60)
