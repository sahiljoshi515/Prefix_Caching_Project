"""Milestone 3 Task 2: build the results page from summary.json.

    python -m analysis.m3t2_report --summary results/m3-study/summary.json \\
        --out results/m3-study/mixed-workload-reuse.html

Writes one self-contained HTML page (spec 007 FR-011, constitution V) that
answers six questions about LRU, LFU and FIFO on the mixed workload, in
the order the course asked them:

1. the reuse time interval each policy serves (the mix);
2. the reuse portion each policy serves (the mix, and per workload);
3. how a request's workload can be told from what a server sees;
4. usable reuse against cache size, with offline Belady (the mix);
5. reuse portion and interval of each workload run alone;
6. each workload alone against sharing the cache;

and a seventh on the new policies, TTDF, RACE-RP
(``vllm/core/reuse_evictors.py``) and Sampled LRFU
(``vllm/core/sampled_lrfu.py``), which also join the mixed-workload
charts of points 1, 2 and 4 once their runs are in ``summary.json``.

Every number in its text and charts is read from ``summary.json``
(``analysis/m3t2_study.py``, with ``analysis/identify.py``'s report); the
charts are drawn in the browser by ``analysis/m3t2_page/charts.js`` with
Observable Plot, styled by ``analysis/m3t2_page/style.css``. Cache-replay
figures carry a "cache replay" badge; everything else comes from
``vllm serve`` runs.
"""
import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PAGE = Path(__file__).with_name("m3t2_page")
TITLE = "Mixed-Workload Reuse Study"
PLOT = ("https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js",
        "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.16/dist/"
        "plot.umd.min.js")
FONTS = ("https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+"
         "Mono:wght@400;600&family=Atkinson+Hyperlegible+Next:ital,wght@0,400;"
         "0,600;1,400&family=Schibsted+Grotesk:wght@500;700&display=swap")
Row = Dict[str, Any]

POLICIES = ("lru", "lfu", "fifo")
# The new policies, explored on the mixed workload only.
NEW_POLICIES = ("ttdf", "race_rp", "sampled_lrfu")
GROUPS = ("sharegpt", "agentbank", "ccbench", "bailian")
LABELS = {
    "sharegpt": "ShareGPT",
    "agentbank": "AgentBank",
    "ccbench": "CC-Bench",
    "bailian": "Bailian",
    "all": "Whole mix",
}
POLICY_LABELS = {
    "lru": "LRU",
    "lfu": "LFU",
    "fifo": "FIFO",
    "ttdf": "TTDF",
    "race_rp": "RACE-RP",
    "ttdf_tuned": "TTDF, tuned",
    "race_rp_tuned": "RACE-RP, tuned",
    "sampled_lrfu": "Sampled LRFU",
    "sampled_lrfu_tuned": "Sampled LRFU, tuned",
    "belady": "Belady",
    "ceiling": "No eviction",
}
# The identification models, in the order the course lists the signals.
MODELS = {
    "prompt_length": "Prompt length",
    "prefix_hashes": "Prefix hashes",
    "reuse_history": "Reuse history",
    "interarrival": "Inter-arrival times",
    "all": "All four together",
}
# An interval bin is drawn only with at least this many reuses on offer.
MIN_OFFERED = 1000
# Capacity chart: the plot area's height in pixels (charts.js: 340 high,
# 22 above and 48 below) and the least gap between two end labels.
PLOT_PX = 270
LABEL_GAP_PX = 13


def pct(x: Optional[float], digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.{digits}f}%"


def pts(x: Optional[float], digits: int = 1) -> str:
    """A difference of two shares in percentage points."""
    if x is None:
        return "–"
    v = 100 * x
    unit = "point" if round(abs(v), digits) == 1 else "points"
    return f"{abs(v):.{digits}f} {unit}"


def secs(s: Optional[float]) -> str:
    if s is None:
        return "–"
    if s < 1:
        return f"{1000 * s:.0f} ms"
    if s < 60:
        return f"{s:.0f} s" if s >= 10 else f"{s:.1f} s"
    m = s / 60
    if abs(m - round(m)) < 0.05 or m >= 10:
        return f"{m:.0f} min"
    return f"{m:.1f} min"


def tokens(n: int) -> str:
    """A cache size in tokens: 131072 -> 128K (1K = 1,024 tokens)."""
    return f"{n // 1048576}M" if n >= 1048576 else f"{n // 1024}K"


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def mean(v: Optional[Row]) -> Optional[float]:
    return None if not v else v.get("mean")


def label_offsets(ends: Dict[str, Optional[float]],
                  span: float) -> Dict[str, int]:
    """Vertical pixel offsets (positive: down) that keep end-of-line labels
    `LABEL_GAP_PX` apart on a y axis `span` units tall, highest value first;
    a line without an end value gets none."""
    order = sorted((k for k in ends if ends[k] is not None),
                   key=lambda k: -ends[k])
    out: Dict[str, int] = {}
    last_y = -math.inf
    for k in order:
        y = (1 - ends[k]) * PLOT_PX / span
        placed = max(y, last_y + LABEL_GAP_PX)
        out[k] = round(placed - y)
        last_y = placed
    return out


# --- What each chart reads ----------------------------------------------


def mixed_policies(s: Row) -> List[str]:
    """The policies of the mixed-workload engine views: the baselines, then
    each new policy whose runs the summary holds."""
    have = s["mixed_groups"]["all"]
    return [*POLICIES, *(p for p in NEW_POLICIES if p in have)]


def q2_view(s: Row, pols: Optional[List[str]] = None) -> Row:
    """Point 2: the mixed runs' usable reuse, whole mix and per workload."""
    mg = s["mixed_groups"]
    rows = ["all", *GROUPS]
    return {
        "rows": rows,
        "values":
        {p: {k: mg[k][p]["usable"]
             for k in rows}
         for p in (pols or POLICIES)},
        "ceiling": {k: mg[k]["ceiling"]["usable"]
                    for k in rows},
    }


def q3_view(c: Row) -> Row:
    """Point 3: each model's scores and the whole model's confusion matrix
    (rows: the true workload; shares of the row)."""
    models = [{
        "id": k,
        "label": label,
        "accuracy": c["models"][k]["accuracy"],
        "balanced": c["models"][k]["balanced_accuracy"],
        "tokens": c["models"][k]["token_accuracy"],
    } for k, label in MODELS.items()]
    confusion = c["models"]["all"]["confusion"]
    counts = [[confusion.get(t, {}).get(p, 0) for p in GROUPS] for t in GROUPS]
    share = [[v / sum(row) if sum(row) else 0.0 for v in row]
             for row in counts]
    return {
        "models": models,
        "majority": {
            **c["majority"], "balanced": 1 / len(GROUPS)
        },
        "confusion": {
            "classes": list(GROUPS),
            "counts": counts,
            "share": share
        },
    }


def y_range(values: List[Optional[float]]) -> List[float]:
    """A y domain on 5-point steps with room around `values`."""
    vals = [v for v in values if v is not None]
    lo = math.floor((min(vals) - 0.02) * 20) / 20
    hi = math.ceil((max(vals) + 0.03) * 20) / 20
    return [max(0.0, lo), min(1.0, hi)]


def curve_policies(c: Row, names: Sequence[str]) -> List[str]:
    """The policies among `names` with a complete replay curve."""
    return [
        p for p in names
        if p in c["curve"] and None not in c["curve"][p]["hit_rate"]
    ]


def q4_view(c: Row, names: Optional[Sequence[str]] = None) -> Row:
    """Point 4: the replay curves (`names`, default the baselines, then
    Belady), the ceiling, and the engine runs (with the replay of the same
    seeds) averaged per policy and size; engine runs without a replay value
    are left out."""
    pols = curve_policies(c, [*(names or POLICIES), "belady"])
    curve = {p: c["curve"][p]["hit_rate"] for p in pols}
    engine: Dict[tuple, List[Row]] = {}
    for e in c["engine"]:
        if (e["engine"] is not None and e["replay"] is not None
                and e["policy"] in pols):
            engine.setdefault((e["policy"], e["capacity"]), []).append(e)
    marks = [{
        "policy": p,
        "capacity": cap,
        "engine": sum(e["engine"] for e in es) / len(es),
        "replay": sum(e["replay"] for e in es) / len(es),
        "seeds": sorted(e["seed"] for e in es),
    } for (p, cap), es in sorted(engine.items())]
    values = [v for p in pols for v in curve[p]]
    values += [m["engine"] for m in marks] + [c["ceiling"]]
    domain = y_range(values)
    ends = {p: curve[p][-1] for p in pols}
    return {
        "policies": pols,
        "capacities": c["capacities"],
        "curve": curve,
        "ceiling": c["ceiling"],
        "fixed": c["fixed"],
        "engine": marks,
        "yDomain": domain,
        "labelDy": label_offsets(ends, domain[1] - domain[0]),
    }


def q5_view(iso: Row) -> Row:
    """Point 5: each workload alone, usable reuse and served intervals."""
    rows = iso["rows"]
    return {
        "values":
        {p: {g: rows[g][p]["alone"]
             for g in GROUPS}
         for p in POLICIES},
        "ceiling": {g: rows[g]["ceiling"]
                    for g in GROUPS},
        "curves": {
            g: {p: iso["intervals"][g][p]["curve"]
                for p in POLICIES}
            for g in GROUPS
        },
    }


# A verdict on alone minus sharing, as the chart's table words it.
AGREE = {
    "higher": "yes: lower when sharing",
    "lower": "yes: higher when sharing",
    "tie": "no (a tie by the rule)",
}
# What the page needs from summary.json.
REQUIRED = ("mixed_intervals", "mixed_groups", "ceiling_intervals_by_group",
            "capacity_window", "isolation", "mix", "config")


def q6_view(iso: Row) -> Row:
    """Point 6: alone against sharing, per workload and policy."""
    rows = iso["rows"]
    return {
        "rows": [{
            "group": g,
            "policy": p,
            "alone": mean(rows[g][p]["alone"]),
            "mixed": mean(rows[g][p]["mixed"]),
            "agree": AGREE[rows[g][p]["verdict"]],
        } for g in GROUPS for p in POLICIES],
        "ceiling": {g: mean(rows[g]["ceiling"])
                    for g in GROUPS},
    }


def q7_view(s: Row, pols: Sequence[str]) -> Optional[Row]:
    """Point 7, the new policies: each workload's replay curve against cache
    size (one panel each, with its no-eviction share), evicted blocks per
    second by workload (engine runs), and the tuned settings' curves."""
    c = s.get("capacity_window") or {}
    npol = s.get("new_policies") or {}
    names = curve_policies(c, pols) if c else []
    if not any(p in NEW_POLICIES for p in names) or "groups" not in c:
        return None
    panels = {}
    for g in GROUPS:
        values = [v for p in names for v in c["groups"][p][g]]
        values.append(c["group_ceiling"][g])
        panels[g] = {
            "curve": {p: c["groups"][p][g]
                      for p in names},
            "ceiling": c["group_ceiling"][g],
            "yDomain": y_range(values),
        }
    ev = npol.get("evictions") or {}
    tuned = curve_policies(c, [f"{p}_tuned" for p in NEW_POLICIES])
    return {
        "policies": names,
        "capacities": c["capacities"],
        "fixed": c["fixed"],
        "panels": panels,
        "evictions": {
            p: {
                k: {
                    "mean": ev[p][k]["per_s"],
                    "min": min(ev[p][k]["seeds"]),
                    "max": max(ev[p][k]["seeds"]),
                    "regretted": ev[p][k]["regretted"],
                }
                for k in ("all", *GROUPS)
            }
            for p in pols if p in ev
        },
        "tuned": {p: c["curve"][p]["hit_rate"]
                  for p in tuned},
    }


def build_data(s: Row) -> Row:
    """The compact data object the charts read (window.M3T2)."""
    iso = s.get("isolation")
    mixed = mixed_policies(s)
    replay = [*mixed, *(p for p in NEW_POLICIES if p not in mixed)]
    return {
        "policies": list(POLICIES),
        "mixedPolicies": mixed,
        "groups": list(GROUPS),
        "labels": LABELS,
        "policyLabels": POLICY_LABELS,
        "minOffered": MIN_OFFERED,
        "q1": {
            "curves": {p: s["mixed_intervals"][p]["curve"]
                       for p in mixed}
        },
        "q2": q2_view(s, mixed),
        "q3": q3_view(s["identify"]["characteristics"]),
        "q4": q4_view(s["capacity_window"], replay),
        "q5": q5_view(iso) if iso else None,
        "q6": q6_view(iso) if iso else None,
        "q7": q7_view(s, replay),
    }


# --- The page -------------------------------------------------------------


def figure(fig_id: str,
           claim: str,
           badges: List[str],
           takeaway: str,
           intro: str = "") -> str:
    """One finding: the claim with its badges ("measured:..." or
    "sim:..."), an optional intro, the chart, and what it shows."""
    tags = "".join(f'<span class="badge {kind}">{esc(label)}</span>'
                   for kind, label in (b.split(":", 1) for b in badges))
    intro_html = f'<p class="prose">{intro}</p>' if intro else ""
    return (f'<div class="finding"><div class="claim"><h3>{claim}</h3>{tags}'
            f'</div>{intro_html}<figure class="chart" id="{fig_id}"></figure>'
            f'<div class="takeaway"><b class="label">What this shows</b>'
            f'{takeaway}</div></div>')


def render(data: Row, sections: List[str]) -> str:
    css = (PAGE / "style.css").read_text()
    js = (PAGE / "charts.js").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    scripts = "".join(f'<script src="{u}"></script>' for u in PLOT)
    body = "\n".join(sections)
    return f"""<title>{TITLE}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{css}</style>
<main class="page">
{body}
</main>
{scripts}
<script>window.M3T2 = {payload};</script>
<script>{js}</script>
"""


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary",
                        type=Path,
                        default=Path("results/m3-study/summary.json"))
    parser.add_argument("--out",
                        type=Path,
                        default=Path("results/m3-study/"
                                     "mixed-workload-reuse.html"))
    args = parser.parse_args(argv)
    s = json.loads(args.summary.read_text())
    missing = [k for k in REQUIRED if not s.get(k)]
    if not (s.get("identify") or {}).get("characteristics"):
        missing.append("identify.characteristics")
    if missing:
        raise SystemExit(f"summary.json lacks {', '.join(missing)}: run "
                         "the isolated grid, analysis.tables, "
                         "analysis.identify, the windowed sweep and "
                         "analysis.m3t2_study first")
    data = build_data(s)
    from analysis.m3t2_narrative import sections
    page = render(data, sections(s, data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"Wrote {args.out} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
