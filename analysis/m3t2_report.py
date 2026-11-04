"""Milestone 3 Task 2, Phase 1: build the results page from summary.json.

    python -m analysis.m3t2_report --summary results/m3-study/summary.json \\
        --out results/m3-study/mixed-workload-reuse.html

Writes one self-contained HTML page (spec 007 FR-011, constitution V):
every number in its text and charts is read from ``summary.json``
(``analysis/m3t2_study.py``); the charts are drawn in the browser by
``analysis/m3t2_page/charts.js`` with Observable Plot, styled by
``analysis/m3t2_page/style.css``. Simulator figures carry a "simulator"
badge; everything else comes from ``vllm serve`` runs.
"""
import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PAGE = Path(__file__).with_name("m3t2_page")
TITLE = "Mixed-Workload Reuse Study"
PLOT = ("https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js",
        "https://cdn.jsdelivr.net/npm/@observablehq/plot@0.6.16/dist/"
        "plot.umd.min.js")
FONTS = ("https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible+"
         "Mono:wght@400;600&family=Atkinson+Hyperlegible+Next:ital,wght@0,400;"
         "0,600;1,400&family=Schibsted+Grotesk:wght@500;700&display=swap")
Row = Dict[str, Any]


def pct(x: Optional[float], digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{100 * x:.{digits}f}%"


def pts(x: Optional[float]) -> str:
    if x is None:
        return "–"
    return f"{100 * x:+.1f} points"


def secs(s: Optional[float]) -> str:
    if s is None:
        return "–"
    if s < 1:
        return f"{1000 * s:.0f} ms"
    if s < 90:
        return f"{s:.0f} s" if s >= 10 else f"{s:.1f} s"
    return f"{s / 60:.1f} min" if s < 600 else f"{s / 60:.0f} min"


def mids(edges: List[float]) -> List[float]:
    """A bin's display position: the geometric middle (the first bin, which
    starts at 0, at half its upper edge)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(hi / 2 if lo <= 0 else math.sqrt(lo * hi))
    return out


def _mean(v: Optional[Row]) -> Optional[float]:
    return None if v is None else v.get("mean")


def build_data(s: Row) -> Row:
    """The compact data object the charts read (window.M3T2)."""
    cfg = s["config"]
    W = cfg["workloads"]
    seeds = [str(x) for x in cfg["reported_seeds"]]
    mix = {}
    for w in W:
        per = [s["mix"][k]["workloads"][w] for k in seeds]
        pause = [p["pause_s"].get("p50") for p in per if p.get("pause_s")]
        kept = [
            p.get("kept_session_fraction") for p in per
            if p.get("kept_session_fraction") is not None
        ]
        mix[w] = {
            "requests": sum(p["request_share"] for p in per) / len(per),
            "tokens": sum(p["prompt_token_share"] for p in per) / len(per),
            "sessions": sum(p["sessions"] for p in per) / len(per),
            "pause50": sum(pause) / len(pause) if pause else None,
            "kept": sum(kept) / len(kept) if kept else None,
        }
    q1 = s["q1"]
    retention = {
        p: s["policy_views"][p]["eviction_ages"]["all"]["quantiles"]
        for p in cfg["policies"]
    }
    hbi = {
        "edges": q1["hit_by_interval"]["lru"]["edges"],
        "mids": mids(q1["hit_by_interval"]["lru"]["edges"])
    }
    for p in cfg["policies"]:
        hbi[p] = {w: q1["hit_by_interval"][p][w]["usable_share"] for w in W}
    lost: Row = {}
    for p in cfg["policies"]:
        lost[p] = {}
        for w in W + ["all"]:
            vals = [s["lost"][p][k][w] for k in seeds]
            lost[p][w] = {
                key: sum(v[key] for v in vals)
                for key in ("kept", "evicted", "cut_off")
            }
    pos = s["q2"]["position"]
    edges = pos["all"]["edges"]
    position = {
        "mids":
        mids(edges),
        "labels": [
            f"{int(a)}–{int(b) - 1}" if b - a > 1 else f"{int(a)}"
            for a, b in zip(edges[:-1], edges[1:])
        ],
        "ceiling": {w: pos[w]["ceiling_share"]
                    for w in W},
    }
    density = {
        p: {
            w: {
                "pool":
                s["policy_views"][p]["hit_density"][w]["pool_time_share"],
                "revive":
                s["policy_views"][p]["hit_density"][w]["revive_share"],
                "rate":
                s["policy_views"][p]["hit_density"][w]
                ["revives_per_block_minute"]
            }
            for w in W
        }
        for p in cfg["policies"]
    }
    evictions = {p: s["policy_views"][p]["evictions"] for p in cfg["policies"]}
    data: Row = {
        "workloads": W,
        "labels": cfg["workload_labels"],
        "loop": cfg["workload_loop"],
        "mix": mix,
        "headline": s["headline"],
        "comeBack": {
            "grid": q1["come_back"]["grid"],
            "series": {w: q1["come_back"][w]["came_back"]
                       for w in W}
        },
        "retention": retention,
        "intervals":
        {w: q1["intervals_all"][w].get("quantiles", {})
         for w in W},
        "hitByInterval": hbi,
        "distance": {w: q1["reuse_distance"][w]["beyond_capacity"]
                     for w in W},
        "sources": s["q2"]["sources"],
        "lost": lost,
        "position": position,
        "density": density,
        "evictions": evictions,
    }
    sim = s.get("sim", {}).get("balanced-human")
    if sim:
        caps = sorted(int(k) for k in sim["lru"] if k != "ceiling")
        data["capacity"] = {
            "capacities": caps,
            "fixed": cfg["capacity_tokens"],
            "curves": {
                p: [_mean(sim[p][str(c)]["all"]) for c in caps]
                for p in ("belady", "lru", "lfu", "fifo")
            },
            "ceiling": _mean(sim["ceiling"]["ceiling"]["all"]),
        }
        cap = str(cfg["capacity_tokens"])
        data["beladySplit"] = {
            p: {
                w: _mean((sim[p][cap]
                          if p != "ceiling" else sim["ceiling"]["ceiling"])[w])
                for w in W
            }
            for p in ("ceiling", "belady", "lru")
        }
        data["beladySplit"]["belady_all"] = _mean(sim["belady"][cap]["all"])
        data["beladySplit"]["lru_all"] = _mean(sim["lru"][cap]["all"])
    if "identify" in s:
        data["identify"] = identify_view(s["identify"], W)
    if "signals" in s:
        data["signals"] = signals_view(s["signals"])
    if "paper" in s:
        data["paper"] = paper_view(s["paper"], cfg["workload_labels"])
    return data


# Report views (identification, signals, paper checks) ------------------

TRACE = {
    "bailian-chat": "A (chat)",
    "bailian-api": "B (API)",
    "bailian-thinking": "thinking",
    "bailian-coder": "coder"
}
FEATURE_SETS = [
    ("idle_age", "Idle time only (all LRU uses)"),
    ("+inferred_class", "+ inferred workload"),
    ("+oracle_workload", "+ true workload label"),
    ("+inferred_turn", "+ inferred turn number"),
    ("+index", "+ block position in the prompt"),
    ("+family_popularity", "+ popularity of the prompt's family"),
    ("+hits_before", "+ past hits (what LFU uses)"),
    ("label_free", "All label-free signals together"),
    ("label_free+oracle", "All label-free + true label"),
]


def identify_view(r: Row, workloads: List[str]) -> Row:
    """Ladder shares, confusion matrix and the sentences for question 3."""
    p = r["with_timing"]["pooled"]
    tree = p["tree"]
    steps = {x["step"]: x for x in p["steps"]}
    b1, f2 = steps["block1"]["coverage"], steps["family2"]["coverage"]
    shares = p["step_shares"]
    values = {
        "all": {
            "b1": b1,
            "f2": f2 - b1,
            "tree": tree["accuracy"] - f2,
            "wrong": 1 - tree["accuracy"]
        }
    }
    for w in workloads:
        sh = shares.get(w, {})
        lb1 = sh.get("block1", 0.0)
        lf = sh.get("family2", 0.0) + sh.get("family4", 0.0)
        rec = tree["recall"][w]
        values[w] = {
            "b1": lb1,
            "f2": lf,
            "tree": max(rec - lb1 - lf, 0.0),
            "wrong": 1 - rec
        }
    conf = tree["confusion"]
    share = []
    for w in workloads:
        row = conf.get(w, {})
        tot = sum(row.values()) or 1
        share.append([row.get(c, 0) / tot for c in workloads])
    worst = sorted(((share[i][j], workloads[i], workloads[j])
                    for i in range(len(workloads))
                    for j in range(len(workloads)) if i != j),
                   reverse=True)[:2]
    no_timing = r["without_timing"]["pooled"]["tree"]["accuracy"]
    return {
        "ladder": {
            "rows": [{
                "id": "all",
                "label": "All test requests"
            }] + [{
                "id": w,
                "label": w
            } for w in workloads],
            "segments": [
                {
                    "id": "b1",
                    "label": "Named by its first block"
                },
                {
                    "id": "f2",
                    "label": "Named by its first two blocks"
                },
                {
                    "id": "tree",
                    "label": "Named by the decision tree"
                },
                {
                    "id": "wrong",
                    "label": "Named wrongly"
                },
            ],
            "values":
            values,
        },
        "confusion": {
            "labels": workloads,
            "predicted": workloads,
            "share": share
        },
        "numbers": {
            "fingerprint":
            f2,
            "block1":
            b1,
            "accuracy":
            tree["accuracy"],
            "accuracy_no_timing":
            no_timing,
            "first":
            tree["first_accuracy"],
            "later":
            tree["later_accuracy"],
            "propagation":
            p["propagation"]["accuracy"] if "accuracy" in p.get(
                "propagation", {}) else None,
            "ccbench_known":
            tree["ccbench_known_task_accuracy"],
            "ccbench_new":
            tree["ccbench_new_task_accuracy"],
            "worst":
            worst,
            "parent":
            min(v["parent_is_previous_turn"] for v in r["parents"].values())
        },
    }


def signals_view(r: Row) -> Row:
    """AUC of each feature set, pooled over idle ages, per horizon."""
    horizons = [int(h) for h in r["settings"]["horizons_s"]]
    auc = {
        fs:
        {str(h): r["models"][fs][f"H={h}"]["pooled"]["auc"]
         for h in horizons}
        for fs, _ in FEATURE_SETS if fs in r["models"]
    }
    base = r["base_rates"]

    def rate(h: int, a: int, w: str) -> Optional[float]:
        try:
            return base[f"H={h}"][f"a={a}"]["test"][w]["reused"]
        except KeyError:
            return None

    return {
        "horizons":
        horizons,
        "rows": [{
            "id": fs,
            "label": label
        } for fs, label in FEATURE_SETS if fs in auc],
        "auc":
        auc,
        "base": {
            f"{a}": {
                w: rate(60, a, w)
                for w in ("ccbench", "agentbank", "sharegpt", "bailian-chat",
                          "bailian-coder", "bailian-api", "bailian-thinking")
            }
            for a in (0, 30, 120)
        },
    }


def _mix_mean(mix: Row, *path: str) -> Optional[float]:
    vals = []
    for run in mix.values():
        v: Any = run
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


def _decile_ends(d: Row) -> str:
    dec = (d.get("prefix") or d).get("by_decile") or []
    return f"{pct(dec[0])} → {pct(dec[-1])}" if dec else "–"


def _future(d: Row) -> str:
    f = d.get("future_reuse_by_past_accesses") or {}
    return "/".join(pct(f.get(k)) for k in ("0", "1", "2", "3+"))


def _claim_text(c: int, raw: Row, mix: Row,
                labels: Dict[str, str]) -> Tuple[str, str]:
    """The raw-trace and mix cells of one scorecard claim, in words."""

    def get(w: str, *k: str) -> Any:
        return _dig(raw.get(w, {}), *k)

    def mm(*k: str) -> Optional[float]:
        return _mix_mean(mix, *k)

    a, b, t, c_ = ("bailian-chat", "bailian-api", "bailian-thinking",
                   "bailian-coder")
    if c == 1:
        rtxt = ", ".join(f"{TRACE[w]} {pct(get(w, 'ideal_hit_ratio'))}"
                         for w in TRACE)
        ws = [w for w in labels if mm(w, "ideal_hit_ratio") is not None]
        lo = min(ws, key=lambda w: mm(w, "ideal_hit_ratio"))
        hi = max(ws, key=lambda w: mm(w, "ideal_hit_ratio"))
        mtxt = (f"whole mix {pct(mm('all', 'ideal_hit_ratio'))}; from "
                f"{labels[lo]} {pct(mm(lo, 'ideal_hit_ratio'))} to "
                f"{labels[hi]} {pct(mm(hi, 'ideal_hit_ratio'))}")
    elif c == 2:
        same = "same_session_share"
        top = "top10_block_share_of_reuses"
        rtxt = (f"same session: A {pct(get(a, same))}, thinking "
                f"{pct(get(t, same))}, coder {pct(get(c_, same))} (B has "
                f"one request per session); top 10% of blocks give A "
                f"{pct(get(a, top))} and B {pct(get(b, top))} of reuses")
        mtxt = (f"same session {pct(mm('all', same))}; top 10% of blocks "
                f"give {pct(mm('all', top))}")
    elif c == 3:
        key = "turn1_share_of_hits"
        rtxt = (f"B is all turn 1, so 100% by construction; A "
                f"{pct(get(a, key))}, thinking {pct(get(t, key))}, coder "
                f"{pct(get(c_, key))}")
        mtxt = (f"Bailian API {pct(mm(b, key))}; whole mix "
                f"{pct(mm('all', key))}")
    elif c == 4:
        rtxt = "80% of reuses within: " + ", ".join(
            f"{TRACE[w]} {secs(get(w, 'p80'))}" for w in TRACE)
        mtxt = (f"whole mix {secs(mm('all', 'p80'))}; Bailian API "
                f"{secs(mm(b, 'p80'))} after thinning")
    elif c == 5:
        ks = [get(a, "categories", "text-1", k) for k in ("ks", "cure_ks")]
        kb = [get(b, "categories", "api-1", k) for k in ("ks", "cure_ks")]
        rtxt = (f"turn-1 fits are poor: KS {ks[0]:.2f} (A), {kb[0]:.2f} "
                f"(B); allowing a share never reused helps ({ks[1]:.2f}, "
                f"{kb[1]:.2f}) but not enough")
        mtxt = ("same pattern: reuse within seconds (shared prompts) and "
                "within minutes (history) mix in one category")
    elif c == 6:
        pairs = ((a, "text"), (b, "api"), (t, "thinking"), (c_, "coder"))
        rr = {
            w: get(w, "categories", f"{k}-1", "rate_ratio")
            for w, k in pairs
        }
        rtxt = "hour 2's rate against hour 1's: " + ", ".join(
            f"{TRACE[w]} {v:.2f}×" for w, v in rr.items() if v)
        mtxt = "holds between the two halves of a run"
    elif c == 7:
        parts = []
        for w in (a, t, c_):
            bt = get(w, "by_turn") or {}
            parts.append(f"{TRACE[w]} {pct(bt.get('1'))} at turn 1 → "
                         f"{pct(bt.get('6+'))} at turn 6+")
        rtxt = "; ".join(parts)
        mtxt = ("rises for human traffic; flat for agent loops, which "
                "nearly always continue")
    elif c == 8:
        rtxt = (f"first vs last tenth of a prompt: A "
                f"{_decile_ends(raw[a])}, B {_decile_ends(raw[b])}")
        firsts = [(v.get("all", {}).get("by_decile") or [None])[0]
                  for v in mix.values()]
        lasts = [(v.get("all", {}).get("by_decile") or [None])[-1]
                 for v in mix.values()]
        mtxt = (f"whole mix {pct(_avg(firsts))} → {pct(_avg(lasts))}")
    elif c == 9:
        rtxt = ("chance a block is used again after 0/1/2/3+ past uses: "
                f"A {_future(raw[a])}, B {_future(raw[b])}")
        m0 = next(iter(mix.values()), {}).get("all", {})
        mtxt = f"whole mix {_future(m0)}: frequency is informative"
    else:
        m0 = next(iter(mix.values()), {}).get("live_blocks", {})
        rtxt = "blocks with a future reuse (full rate): " + ", ".join(
            f"{TRACE[w]} {get(w, 'live_blocks', 'mean') or 0:,.0f}"
            for w in TRACE)
        mtxt = (f"{m0.get('mean', 0):,.0f} on average against 8,192 "
                "cached; see the capacity chart")
    return rtxt, mtxt


def _avg(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def paper_view(r: Row, labels: Dict[str, str]) -> List[Row]:
    """One plain row per claim: the paper's number, ours on the raw traces
    and in our mix, and the verdict."""
    out = []
    for row in r["rows"]:
        rtxt, mtxt = _claim_text(row["claim"],
                                 row.get("raw") or {},
                                 row.get("mix") or {}, labels)
        verdict = row.get("verdict", "")
        chip = {
            "holds": "holds",
            "partly": "partly",
            "no": "no"
        }.get(verdict, "na")
        out.append({
            "claim": row["statement"],
            "paper": row["paper"],
            "raw": rtxt,
            "mix": mtxt,
            "verdict": verdict,
            "chip": chip
        })
    return out


def _dig(d: Any, *keys: str) -> Any:
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


# Page ---------------------------------------------------------------------


def esc(text: str) -> str:
    return html.escape(text, quote=False)


def figure(fig_id: str,
           claim: str,
           badges: List[str],
           takeaway: str,
           note: str = "") -> str:
    tags = "".join(f'<span class="badge {b}">{esc(label)}</span>'
                   for b, label in [(x.split(":")[0], x.split(":", 1)[1])
                                    for x in badges])
    note_html = f'<p class="prose">{note}</p>' if note else ""
    return (f'<div class="finding"><div class="claim"><h3>{claim}</h3>{tags}'
            f'</div>{note_html}<figure class="chart" id="{fig_id}"></figure>'
            f'<p class="takeaway"><b>For the design</b>{takeaway}</p></div>')


def render(s: Row, data: Row, sections: Dict[str, str]) -> str:
    css = (PAGE / "style.css").read_text()
    js = (PAGE / "charts.js").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    scripts = "".join(f'<script src="{u}"></script>' for u in PLOT)
    return f"""<title>{TITLE}</title>
<meta name="claude:viewport-fit-sides" content="cover">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONTS}">
<style>{css}</style>
<main class="page">
{sections["header"]}
{sections["answers"]}
{sections["glossary"]}
{sections["mix"]}
{sections["headline"]}
{sections["q1"]}
{sections["q2"]}
{sections["q3"]}
{sections["paper"]}
{sections["headroom"]}
{sections["signals"]}
{sections["checks"]}
{sections["method"]}
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
    data = build_data(s)
    from analysis.m3t2_narrative import sections
    page = render(s, data, sections(s, data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"Wrote {args.out} ({len(page) // 1024} KB)")


if __name__ == "__main__":
    main()
