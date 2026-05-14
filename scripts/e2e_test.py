"""
End-to-end pipeline test — runs against the live HF Space backend.

Usage:
    python scripts/e2e_test.py
    python scripts/e2e_test.py --url http://localhost:8000   # local dev
    python scripts/e2e_test.py --verbose                     # print full responses
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import requests

BASE_URL = "https://k2p-neuro-gym-rag.hf.space"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def ok(msg):  print(f"  {GREEN}✓{RESET} {msg}")
def fail(msg): print(f"  {RED}✗{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def info(msg): print(f"  {CYAN}→{RESET} {msg}")


# ── Result tracking ───────────────────────────────────────────────────────────
@dataclass
class Result:
    name: str
    passed: bool
    detail: str = ""
    retrieval_ms: int = 0
    generation_ms: int = 0
    sources: int = 0


results: list[Result] = []

def record(name: str, passed: bool, detail: str = "", **kw) -> bool:
    results.append(Result(name, passed, detail, **kw))
    if passed:
        ok(f"{name}  {detail}")
    else:
        fail(f"{name}  {detail}")
    return passed


# ── HTTP helpers ──────────────────────────────────────────────────────────────
def get(url: str, **kwargs) -> requests.Response:
    return requests.get(url, timeout=30, **kwargs)

def chat(url: str, query: str, session_id: str = "e2e",
         config: str = "F — all + BM25") -> dict[str, Any] | None:
    try:
        r = requests.post(
            f"{url}/chat",
            data={"query": query, "session_id": session_id, "config_name": config},
            timeout=120,
        )
        if r.status_code == 200:
            return r.json()
        warn(f"HTTP {r.status_code}: {r.text[:200]}")
        return None
    except Exception as exc:
        warn(f"Request error: {exc}")
        return None


# ── Test suites ───────────────────────────────────────────────────────────────

def test_health(url: str) -> bool:
    print(f"\n{BOLD}── 1. Health check ──────────────────────────────────────{RESET}")
    try:
        r = get(f"{url}/health")
        d = r.json()
        record("HTTP 200",       r.status_code == 200)
        record("status=ok",      d.get("status") == "ok",    d.get("status","?"))
        record("qdrant loaded",  isinstance(d.get("qdrant"), dict) and len(d["qdrant"]) == 3,
               str(d.get("qdrant", {})))
        record("bm25 loaded",    d.get("bm25_loaded") is True)
        record("gemini loaded",  d.get("gemini_loaded") is True)

        qdrant = d.get("qdrant", {})
        for col, count in qdrant.items():
            record(f"  {col} non-empty", isinstance(count, int) and count > 0, f"{count:,} vectors")

        return d.get("status") == "ok"
    except Exception as exc:
        record("health reachable", False, str(exc))
        return False


def test_configs(url: str) -> None:
    print(f"\n{BOLD}── 2. All 8 retrieval configs ───────────────────────────{RESET}")
    configs = [
        "A — images only",
        "B — text only",
        "C — tables only",
        "D — all dense",
        "E — tables + BM25",
        "F — all + BM25",
        "G — hybrid + rerank",
        "H — BM25 only",
    ]
    query = "What was athlete_00042 squat weight in week 8?"
    for cfg in configs:
        info(f"Config {cfg}")
        d = chat(url, query, session_id=f"cfg-{cfg[0]}", config=cfg)
        passed = (
            d is not None
            and bool(d.get("response", "").strip())
            and len(d.get("sources", [])) > 0
        )
        record(
            cfg,
            passed,
            f"sources={len(d.get('sources',[]))}  ret={d.get('retrieval_ms')}ms  gen={d.get('generation_ms')}ms"
            if d else "no response",
            retrieval_ms  = d.get("retrieval_ms", 0) if d else 0,
            generation_ms = d.get("generation_ms", 0) if d else 0,
            sources       = len(d.get("sources", [])) if d else 0,
        )
        time.sleep(1)


def test_intents(url: str) -> None:
    print(f"\n{BOLD}── 3. Intent coverage ───────────────────────────────────{RESET}")
    cases = [
        ("factual — specific week",
         "What was athlete_00042 squat weight and RPE in week 8?"),
        ("factual — athlete profile",
         "What is athlete_00089 Dots score and training level?"),
        ("comparison",
         "Compare the peak bench press of an intermediate and an advanced athlete."),
        ("analytical — trends",
         "At what point in the 12-week block do most advanced athletes show an RPE spike?"),
        ("coaching — advice",
         "My RPE on deadlift is consistently hitting 9.5 in week 6. Is that normal?"),
        ("ranking",
         "Who are the strongest deadlifters in the advanced category ranked by Dots score?"),
        ("visual — chart description",
         "Can you describe what athlete_00042 progression chart looks like?"),
        ("general knowledge",
         "What does a good deload week look like for an intermediate lifter?"),
    ]
    for name, query in cases:
        info(name)
        d = chat(url, query, session_id=f"intent-{name[:8]}")
        passed = d is not None and bool(d.get("response", "").strip())
        detail = ""
        if d:
            detail = (f"intent={d.get('intent','?')}  "
                      f"sources={len(d.get('sources',[]))}  "
                      f"ret={d.get('retrieval_ms')}ms")
        record(name, passed, detail)
        time.sleep(1)


def test_multiturn(url: str) -> None:
    print(f"\n{BOLD}── 4. Multi-turn memory ─────────────────────────────────{RESET}")
    sid = "e2e-memory"

    info("Turn 1 — establish athlete context")
    d1 = chat(url, "Tell me about athlete_00117 training program.", session_id=sid)
    record("turn 1 response", d1 is not None and bool(d1.get("response", "").strip()))

    info("Turn 2 — pronoun reference (requires memory)")
    d2 = chat(url, "How did their deadlift progress from week 1 to week 12?", session_id=sid)
    passed = (
        d2 is not None
        and bool(d2.get("response", "").strip())
        and "athlete_00117" in d2.get("retrieval_query", "") + d2.get("response", "")
    )
    record("turn 2 references prior athlete", passed,
           f"retrieval_query={d2.get('retrieval_query','?')[:60]}" if d2 else "no response")

    info("Turn 3 — follow-up detail")
    d3 = chat(url, "What accessories did they do on lower body days?", session_id=sid)
    record("turn 3 coherent follow-up", d3 is not None and bool(d3.get("response", "").strip()))


def test_edge_cases(url: str) -> None:
    print(f"\n{BOLD}── 5. Edge cases ────────────────────────────────────────{RESET}")

    info("Non-existent athlete")
    d = chat(url, "What is athlete_99999 squat max?", session_id="e2e-edge-1")
    record("non-existent athlete — graceful",
           d is not None and bool(d.get("response", "").strip()),
           "responded without crash")

    info("Very short query")
    d = chat(url, "squat", session_id="e2e-edge-2")
    record("short query — responded", d is not None and bool(d.get("response", "").strip()))

    info("Out-of-domain question")
    d = chat(url, "What is the capital of France?", session_id="e2e-edge-3")
    record("out-of-domain — responded", d is not None and bool(d.get("response", "").strip()))

    info("Query at max length (2000 chars)")
    long_q = ("Tell me about athlete training " * 65)[:2000]
    d = chat(url, long_q, session_id="e2e-edge-4")
    record("max-length query — no 422", d is not None)

    info("Exceeded max length (2001 chars)")
    too_long = "x" * 2001
    try:
        r = requests.post(
            f"{url}/chat",
            data={"query": too_long, "session_id": "e2e-edge-5",
                  "config_name": "F — all + BM25"},
            timeout=30,
        )
        record("over-limit query — 422 returned", r.status_code == 422,
               f"got {r.status_code}")
    except Exception as exc:
        record("over-limit query — 422 returned", False, str(exc))


def test_latency(url: str) -> None:
    print(f"\n{BOLD}── 6. Latency benchmarks ────────────────────────────────{RESET}")
    queries = [
        ("factual",  "What is athlete_00042 squat in week 8?"),
        ("compare",  "Compare bench press between intermediate and advanced athletes."),
        ("coaching", "What should I do if my squat has stalled for 3 weeks?"),
    ]
    for name, query in queries:
        d = chat(url, query, session_id=f"lat-{name}")
        if d:
            ret = d.get("retrieval_ms", 0)
            gen = d.get("generation_ms", 0)
            total = ret + gen
            passed = total < 30_000   # 30s hard limit
            record(
                f"{name} latency",
                passed,
                f"retrieval={ret}ms  generation={gen}ms  total={total}ms",
                retrieval_ms=ret, generation_ms=gen,
            )
        else:
            record(f"{name} latency", False, "no response")
        time.sleep(1)


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(verbose: bool) -> None:
    print(f"\n{BOLD}{'─'*55}{RESET}")
    total  = len(results)
    passed = sum(1 for r in results if r.passed)
    failed = total - passed

    print(f"{BOLD}  Results: {GREEN}{passed} passed{RESET}  {RED}{failed} failed{RESET}  / {total} total{BOLD}{RESET}")

    if failed:
        print(f"\n{RED}  Failed tests:{RESET}")
        for r in results:
            if not r.passed:
                print(f"    {RED}✗{RESET} {r.name}  {r.detail}")

    # Latency summary
    timed = [r for r in results if r.retrieval_ms or r.generation_ms]
    if timed:
        avg_ret = sum(r.retrieval_ms for r in timed) / len(timed)
        avg_gen = sum(r.generation_ms for r in timed) / len(timed)
        print(f"\n  Avg retrieval : {avg_ret:.0f} ms")
        print(f"  Avg generation: {avg_gen:.0f} ms")

    print(f"{'─'*55}")
    sys.exit(0 if failed == 0 else 1)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Neuro Gym E2E test suite")
    parser.add_argument("--url", default=BASE_URL, help="Backend base URL")
    parser.add_argument("--verbose", action="store_true", help="Print full responses")
    parser.add_argument(
        "--suite", nargs="+",
        choices=["health", "configs", "intents", "multiturn", "edge", "latency", "all"],
        default=["all"],
    )
    args = parser.parse_args()
    url = args.url.rstrip("/")
    suites = set(args.suite)
    run_all = "all" in suites

    print(f"\n{BOLD}Neuro Gym RAG — End-to-End Test Suite{RESET}")
    print(f"Target: {CYAN}{url}{RESET}\n")

    if run_all or "health" in suites:
        alive = test_health(url)
        if not alive:
            print(f"\n{RED}Backend not reachable — aborting remaining tests.{RESET}")
            print_summary(args.verbose)

    if run_all or "configs"   in suites: test_configs(url)
    if run_all or "intents"   in suites: test_intents(url)
    if run_all or "multiturn" in suites: test_multiturn(url)
    if run_all or "edge"      in suites: test_edge_cases(url)
    if run_all or "latency"   in suites: test_latency(url)

    print_summary(args.verbose)


if __name__ == "__main__":
    main()
