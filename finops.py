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
MODEL = "gpt-4.1-mini"

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
        filters: dict | None = None,
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

    if filters is not None:
        return _ce.get_cost_and_usage(
            TimePeriod=time_period,
            Granularity=granularity,
            Filter=filters,
            Metrics=["BlendedCost"],
            GroupBy=[{"Type": group_by_type, "Key": group_by_key}],
        )
    else:
        return _ce.get_cost_and_usage(
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

def query_cost_forecast(start_date: str, end_date: str, metric: str = "BLENDED_COST", granularity: str = "MONTHLY"):
    return _ce.get_cost_forecast(
        TimePeriod={"Start": start_date, "End": end_date},
        Metric=metric,
        Granularity=granularity,
    )
def query_usage_forecast(start_date: str, end_date: str, metric: str = "USAGE_QUANTITY", granularity: str = "DAILY"):
    return _ce.get_usage_forecast(
        TimePeriod={"Start": start_date, "End": end_date},
        Metric=metric,
        Granularity=granularity,
    )

def list_dimension_values(dimension: str, time_period: dict):
    return _ce.get_dimension_values(
        TimePeriod=time_period,
        Dimension=dimension,
    )

def query_reservation_utilization(time_period: dict, granularity="MONTHLY"):
    return _ce.get_reservation_utilization(
        TimePeriod=time_period,
        Granularity=granularity,
    )
def query_reservation_coverage(time_period: dict, granularity="MONTHLY"):
    return _ce.get_reservation_coverage(
        TimePeriod=time_period,
        Granularity=granularity,
    )

def query_savings_plans_utilization(time_period: dict):
    return _ce.get_savings_plans_utilization(TimePeriod=time_period)

def query_savings_plans_coverage(time_period: dict):
    return _ce.get_savings_plans_coverage(TimePeriod=time_period)

def purchase_savings_plans_recommendation():
    return _ce.get_savings_plans_purchase_recommendation()

def reservation_purchase_recommendation():
    return _ce.get_reservation_purchase_recommendation(
        Service="AmazonEC2",
        AccountScope="PAYER",
        LookbackPeriodInDays="SIXTY_DAYS",
        TermInYears="ONE_YEAR",
        PaymentOption="NO_UPFRONT",
    )

def list_anomaly_monitors():
    return _ce.describe_anomaly_monitors()

def get_cost_anomalies(time_period: dict):
    return _ce.get_anomalies(TimePeriod=time_period)

def list_cost_categories():
    return _ce.list_cost_category_definitions()

def list_metrics(namespace: str, metric_name_prefix: str = None):
    args = {"Namespace": namespace}
    if metric_name_prefix:
        args["MetricName"] = metric_name_prefix
    return _cw.list_metrics(**args)

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
                "filter": {"type": "object", "description": "Represents a filter to the get_cost_usage call. Ex: {\"Key\": \"SERVICE\", \"Values\": [\"EC2\"]}"},
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
        "name": "query_cost_forecast",
        "description": "Get AWS cost forecast for a custom period",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD start of forecast window"
                },
                "end_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD end of forecast window"
                },
                "metric": {
                    "type": "string",
                    "description": "Which cost metric to forecast",
                    "default": "BLENDED_COST"
                },
                "granularity": {
                    "type": "string",
                    "description": "Time grain of forecast",
                    "default": "MONTHLY"
                }
            },
            "required": ["start_date", "end_date"]
        }
    },
    {
        "name": "query_usage_forecast",
        "description": "Get AWS usage forecast (e.g. EC2 hours) for a custom period",
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD start of usage forecast window"
                },
                "end_date": {
                    "type": "string",
                    "description": "YYYY-MM-DD end of usage forecast window"
                },
                "metric": {
                    "type": "string",
                    "description": "Which usage metric to forecast",
                    "default": "USAGE_QUANTITY"
                },
                "granularity": {
                    "type": "string",
                    "description": "Time grain of usage forecast",
                    "default": "DAILY"
                }
            },
            "required": ["start_date", "end_date"]
        }
    },
    {
        "name": "list_dimension_values",
        "description": "List possible values for a given Cost Explorer dimension",
        "parameters": {
            "type": "object",
            "properties": {
                "dimension": {
                    "type": "string",
                    "description": "Dimension name (e.g. SERVICE, REGION, TAG)"
                },
                "time_period": {
                    "type": "object",
                    "description": "TimePeriod dict with Start and End YYYY-MM-DD",
                    "properties": {
                        "Start": {"type": "string"},
                        "End": {"type": "string"}
                    },
                    "required": ["Start", "End"]
                }
            },
            "required": ["dimension", "time_period"]
        }
    },
    {
        "name": "query_reservation_utilization",
        "description": "Get Reservation Utilization report for a time period",
        "parameters": {
            "type": "object",
            "properties": {
                "time_period": {
                    "type": "object",
                    "description": "TimePeriod dict with Start and End YYYY-MM-DD",
                    "properties": {
                        "Start": {"type": "string"},
                        "End": {"type": "string"}
                    },
                    "required": ["Start", "End"]
                },
                "granularity": {
                    "type": "string",
                    "description": "Granularity of report",
                    "default": "MONTHLY"
                }
            },
            "required": ["time_period"]
        }
    },
    {
        "name": "query_reservation_coverage",
        "description": "Get Reservation Coverage report for a time period",
        "parameters": {
            "type": "object",
            "properties": {
                "time_period": {
                    "type": "object",
                    "description": "TimePeriod dict with Start and End YYYY-MM-DD",
                    "properties": {
                        "Start": {"type": "string"},
                        "End": {"type": "string"}
                    },
                    "required": ["Start", "End"]
                },
                "granularity": {
                    "type": "string",
                    "description": "Granularity of report",
                    "default": "MONTHLY"
                }
            },
            "required": ["time_period"]
        }
    },
    {
        "name": "query_savings_plans_utilization",
        "description": "Get Savings Plans Utilization for a time period",
        "parameters": {
            "type": "object",
            "properties": {
                "time_period": {
                    "type": "object",
                    "description": "TimePeriod dict with Start and End YYYY-MM-DD",
                    "properties": {
                        "Start": {"type": "string"},
                        "End": {"type": "string"}
                    },
                    "required": ["Start", "End"]
                }
            },
            "required": ["time_period"]
        }
    },
    {
        "name": "query_savings_plans_coverage",
        "description": "Get Savings Plans Coverage for a time period",
        "parameters": {
            "type": "object",
            "properties": {
                "time_period": {
                    "type": "object",
                    "description": "TimePeriod dict with Start and End YYYY-MM-DD",
                    "properties": {
                        "Start": {"type": "string"},
                        "End": {"type": "string"}
                    },
                    "required": ["Start", "End"]
                }
            },
            "required": ["time_period"]
        }
    },
    {
        "name": "purchase_savings_plans_recommendation",
        "description": "Get AWS Savings Plans purchase recommendations",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "reservation_purchase_recommendation",
        "description": "Get AWS Reserved Instances purchase recommendations",
        "parameters": {
            "type": "object",
            "properties": {
                "Service": {
                    "type": "string",
                    "description": "Service code, e.g. AmazonEC2"
                },
                "AccountScope": {
                    "type": "string",
                    "description": "PAYER or LINKED"
                },
                "LookbackPeriodInDays": {
                    "type": "string",
                    "description": "Look-back window, e.g. SIXTY_DAYS"
                },
                "TermInYears": {
                    "type": "string",
                    "description": "ONE_YEAR or THREE_YEARS"
                },
                "PaymentOption": {
                    "type": "string",
                    "description": "NO_UPFRONT, PARTIAL_UPFRONT, or ALL_UPFRONT"
                }
            },
            "required": ["Service", "AccountScope", "LookbackPeriodInDays", "TermInYears", "PaymentOption"]
        }
    },
    {
        "name": "list_anomaly_monitors",
        "description": "List all Cost Explorer anomaly monitors",
        "parameters": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_cost_anomalies",
        "description": "Retrieve cost anomalies for a given period",
        "parameters": {
            "type": "object",
            "properties": {
                "time_period": {
                    "type": "object",
                    "description": "TimePeriod dict with Start and End YYYY-MM-DD",
                    "properties": {
                        "Start": {"type": "string"},
                        "End": {"type": "string"}
                    },
                    "required": ["Start", "End"]
                }
            },
            "required": ["time_period"]
        }
    },
    {
        "name": "list_cost_categories",
        "description": "List all defined Cost Categories",
        "parameters": {
            "type": "object",
            "properties": {}
        }
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
            "You need to assist the user how to optimize the AWS bill, "
            "You have some tools that you can use to retrieve information.  "
            "You can call multiple tools per interaction. Summarize results from the tool when it makes sense"
            "User might be interested in disglose specific charges."
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