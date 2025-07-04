#!/usr/bin/env python3
"""
finops_chat_agent.py  —  OpenAI v1 interactive REPL (read-only)
================================================================
Investigate AWS cost spikes from a CLI chat. 100 % read-only.
This patch fixes a Cost Explorer edge-case: AWS requires the **Start** date to
be **strictly before** the **End** date when `Granularity="DAILY"`. Asking for
`window_hours < 24` therefore blew up. The function now coerces any request
shorter than 24 h into a 24-hour window.

Usage unchanged:
```bash
python finops_chat_agent.py
```
Then type questions or `quit`.
"""
from __future__ import annotations

import json
import os
import sys
import datetime as dt
from typing import Any, Dict, List

import boto3
from dateutil import parser as dt_parser
from dateutil import tz
from tabulate import tabulate

try:
    from openai import OpenAI  # ≥1.0 client
except ImportError:
    sys.stderr.write("ERROR: run 'pip install --upgrade openai'\n")
    sys.exit(1)

if not os.getenv("OPENAI_API_KEY"):
    sys.stderr.write("ERROR: set OPENAI_API_KEY first.\n")
    sys.exit(1)

client = OpenAI()
MODEL = "gpt-4o-mini"

_ce = boto3.client("ce", region_name="us-east-1")
_cw = boto3.client("cloudwatch")

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _now() -> dt.datetime:
    return dt.datetime.utcnow().replace(tzinfo=tz.tzutc())


def _iso(t: dt.datetime) -> str:
    return t.astimezone(tz.tzutc()).strftime("%Y-%m-%dT%H:%M:%SZ")


def _hdr(title: str) -> None:
    print("\n" + title)
    print("=" * len(title))


def _parse_date(s: str) -> dt.date:
    """Parse YYYY‑MM‑DD or natural month like 'May 2025'."""
    d: dt.datetime = dt_parser.parse(s, default=dt.datetime.utcnow())
    return d.date()

# ──────────────────────────────────────────────────────────────────────────────
# LLM-exposed functions
# ──────────────────────────────────────────────────────────────────────────────

def query_cost_explorer(
        start_date: str | None = None,
        end_date:   str | None = None,
        group_by_type: str | str = "DIMENSION",
        group_by_key: str | str = "SERVICE",
        granularity: str | str = "DAILY"
    ) -> Dict[str, Any]:
    #Return top‑10 services by cost within the specified period.

    # ------------------------------------------------------------
    #  Choose a date window exactly once, in priority order
    # ------------------------------------------------------------
    today = dt.date.today()

    if start_date and end_date:                          # explicit dates win
        start = dt.date.fromisoformat(start_date)
        end   = dt.date.fromisoformat(end_date)
    else:                                                 # default: 24 h
        hours = 24
        start = today - dt.timedelta(hours=hours)
        end   = today

    # Cost Explorer expects End **exclusive**, so +1 day
    end_excl = end + dt.timedelta(days=1)

    time_period = {
        "Start": start.strftime("%Y-%m-%d"),
        "End":   end_excl.strftime("%Y-%m-%d"),
    }

    res = _ce.get_cost_and_usage(
        TimePeriod=time_period,
        Granularity=granularity,
        Metrics=["BlendedCost"],
        GroupBy=[{"Type": group_by_type, "Key": group_by_key}],
    )

    #rows = [
    #    {
    #        "service": g["Keys"][0],
    #        "cost_usd": round(float(g["Metrics"]["BlendedCost"]["Amount"]), 2),
    #    }
    #    for g in res["ResultsByTime"][0]["Groups"]
    #]
    #rows.sort(key=lambda r: r["cost_usd"], reverse=True)
    return res


def query_cloudwatch(metric: str, namespace: str, dimension_name: str, dimension_value: str, period: int = 300, window_hours: int = 1) -> Dict[str, Any]:
    end = _now()
    start = end - dt.timedelta(hours=window_hours)
    res = _cw.get_metric_statistics(
        Namespace=namespace,
        MetricName=metric,
        Dimensions=[{"Name": dimension_name, "Value": dimension_value}],
        StartTime=start,
        EndTime=end,
        Period=period,
        Statistics=["Average"],
    )
    pts = sorted(res["Datapoints"], key=lambda d: d["Timestamp"])
    return {
        "metric": metric,
        "namespace": namespace,
        "dimension": {dimension_name: dimension_value},
        "period": period,
        "values": [{"timestamp": _iso(dp["Timestamp"]), "average": dp["Average"]} for dp in pts],
    }


def rightsizing_recommendations() -> Dict[str, Any]:
    return {
        "ec2": [{"instance_id": "i-demo", "current_type": "m5a.large", "recommended_type": "t4g.medium", "monthly_savings_usd": 25}],
        "rds": [],
    }

# ──────────────────────────────────────────────────────────────────────────────
# Tool schemas (OpenAI v1)
# ──────────────────────────────────────────────────────────────────────────────
FUNCTION_SCHEMAS: List[Dict[str, Any]] = [
    {
        "name": "query_cost_explorer",
        "description": "Get AWS cost by service for a custom period (dates or hours).",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "YYYY‑MM‑DD or month name"},
                "end_date": {"type": "string", "description": "YYYY‑MM‑DD or month name"},
                "group_by_type": {"type": "string", "description": "Group results by a specific type", "default": "DIMENSION"},
                "group_by_key": {"type": "string", "description": "Which key use in the group", "default": "SERVICE"},
                "granularity": {"type": "string", "description": "Which time granularity use in the report", "default": "DAILY"}
            },
        },
    },
    {
        "name": "query_cloudwatch",
        "description": "Fetch CloudWatch datapoints for a metric/dimension.",
        "parameters": {
            "type": "object",
            "properties": {
                "metric": {"type": "string"},
                "namespace": {"type": "string"},
                "dimension_name": {"type": "string"},
                "dimension_value": {"type": "string"},
                "period": {"type": "integer", "default": 300},
                "window_hours": {"type": "integer", "default": 1},
            },
            "required": ["metric", "namespace", "dimension_name", "dimension_value"],
        },
    },
    {
        "name": "rightsizing_recommendations",
        "description": "List potential rightsizing wins (demo stub).",
        "parameters": {"type": "object", "properties": {}},
    },
]
TOOLS = [{"type": "function", "function": s} for s in FUNCTION_SCHEMAS]
FUNC_DISPATCH = {
    "query_cost_explorer": query_cost_explorer,
    "query_cloudwatch": query_cloudwatch,
    "rightsizing_recommendations": rightsizing_recommendations,
}

# ──────────────────────────────────────────────────────────────────────────────
# Agent core
# ──────────────────────────────────────────────────────────────────────────────
messages: List[Dict[str, Any]] = [
    {
        "role": "system",
        "content": (
            "You are a FinOps assistant for AWS.  "
            "When the user asks questions about cost, "
            "call the function 'query_cost_explorer' with the right date window.  "
            "After tool calls, summarise the results clearly."
        ),
    }
]
def run_agent(prompt: str, max_steps: int = 6) -> None:
    messages.append({"role": "user", "content": prompt})
    for step in range(max_steps):
        rsp = client.chat.completions.create(model=MODEL, messages=messages, tools=TOOLS, tool_choice="auto")
        msg = rsp.choices[0].message

        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                args = json.loads(tc.function.arguments or "{}")
                _hdr(f"Step {step+1}: {fn_name} {args}")
                result = FUNC_DISPATCH[fn_name](**args)  # type: ignore[arg-type]

                messages.append({"role": "assistant", "content": "", "tool_calls": [tc]})
                messages.append({"role": "tool", "tool_call_id": tc.id, "name": fn_name, "content": json.dumps(result)})

                #if isinstance(result, dict) and "services" in result:
                #    rows = [
                #        {
                #            "service": g["Keys"][0],
                #            "cost_usd": round(float(g["Metrics"]["BlendedCost"]["Amount"]), 2),
                #        }
                #        for g in result["services"]["ResultsByTime"][0]["Groups"]
                #    ]
                #    rows.sort(key=lambda r: r["cost_usd"], reverse=True)
                #    print(tabulate(rows[:5], headers="keys"))
                #else:
                #    print(json.dumps(result, indent=2))
        else:
            _hdr("Assistant response")
            print(msg.content)
            break
    else:
        print("Reached max_steps without a final answer.")

# ──────────────────────────────────────────────────────────────────────────────
# REPL
# ──────────────────────────────────────────────────────────────────────────────

def repl() -> None:
    print("FinOps Chat Agent — type your question, or 'quit' / 'exit' to leave.")
    while True:
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if prompt.lower() in {"quit", "exit"}:
            break
        if not prompt:
            continue

        run_agent(prompt)


if __name__ == "__main__":
    repl()