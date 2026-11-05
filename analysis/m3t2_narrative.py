"""The results page's words (``analysis/m3t2_report.py``).

Every number below is read from ``summary.json`` or from the chart data
built from it; the sentences around them were written for the Phase 1
results and are checked against them by review before the page is
published (spec 007 FR-011). Claims that compare two numbers are worded
from the numbers (for example, a difference whose sign the seeds disagree
on is called a tie, spec 007 SC-003).
"""
from typing import Any, Dict, List, Optional, Tuple

from analysis.m3t2_report import esc, figure, pct, pts, secs

Row = Dict[str, Any]
HUMAN = ("sharegpt", "bailian-chat", "bailian-thinking", "bailian-coder")
AGENTS = ("agentbank", "ccbench")


def _u(s: Row, policy: str, w: str) -> Optional[float]:
    v = s["headline"][policy][w]["usable"]
    return None if v is None else v["mean"]


def _vs(s: Row, policy: str, w: str = "all") -> Row:
    return s["headline"][policy][w]["vs_lru"]


def _tie(v: Row) -> bool:
    """A paired difference is a tie when the seeds disagree on its sign."""
    return v["min"] < 0 < v["max"]


def _same_p50(s: Row, w: str) -> Optional[float]:
    q = s["q1"]["intervals"][w]["same_session"].get("quantiles") or {}
    return q.get("p50")


def _range(s: Row, ws: Tuple[str, ...]) -> Tuple[float, float]:
    vals = [_same_p50(s, w) for w in ws if _same_p50(s, w) is not None]
    return min(vals), max(vals)


def _labels(s: Row) -> Dict[str, str]:
    return s["config"]["workload_labels"]


def _strip(states: List[str]) -> str:
    return "".join(f'<i class="cell {x}"></i>' for x in states)


def _lfu_phrase(s: Row) -> Tuple[str, str]:
    """How LFU compares with LRU overall, in words, and the workloads where
    it loses on every seed (by more than 3 points on average)."""
    v = _vs(s, "lfu")
    if _tie(v):
        overall = "ties with LRU"
    elif abs(v["mean"]) < 0.025:
        overall = "is close to LRU"
    else:
        overall = "beats LRU" if v["mean"] > 0 else "trails LRU"
    lab = _labels(s)
    loses = [
        lab[w] for w in s["config"]["workloads"]
        if max(s["headline"]["lfu"][w]["vs_lru"]["seeds"]) < 0
        and s["headline"]["lfu"][w]["vs_lru"]["mean"] < -0.03
    ]
    return overall, ", ".join(loses)


def _came_back(data: Row, w: str, t: float) -> float:
    g = data["comeBack"]["grid"]
    return data["comeBack"]["series"][w][g.index(t)]


# Header and answers --------------------------------------------------------


def _example_strip(s: Row) -> str:
    """One real request, block by block, under every policy."""
    cfg = s["config"]
    e = s.get("example") or {}
    strip = ""
    if e:
        n = len(e["ceiling"])
        rows = [("No eviction", e["ceiling"])
                ] + [(p.upper(), e[p]) for p in cfg["policies"] if p in e]
        cells = "".join(
            f'<div class="strip-row"><span class="strip-label">{esc(name)}'
            f'</span><span class="cells" role="img" aria-label="{esc(name)}: '
            f'{states.count("hit")} of {n} blocks served from the cache">'
            f'{_strip(states)}</span></div>' for name, states in rows)
        kept = e["lru"].count("hit") if "lru" in e else 0
        strip = f"""
<section class="section strip" aria-labelledby="example"><h2 id="example">One
request, block by block</h2>
  <p class="prose">One real request from the runs: turn {e['turn'] + 1} of
  a ShareGPT conversation, {e['prompt_tokens']:,} prompt tokens in {n}
  blocks of 16 tokens. The person sent it {secs(e.get('since_previous_s'))}
  after the previous answer. A cache that never evicts would serve
  {e['ceiling'].count('hit')} of its {n} blocks; every policy we tested
  kept {kept}, the system prompt all conversations share.</p>
  {cells}
  <div class="legend-inline"><span><i class="cell hit"></i>served from the
  cache</span><span><i class="cell miss"></i>recomputed</span></div>
</section>"""
    return strip


def header(s: Row, data: Row) -> str:
    cfg = s["config"]
    commit = (s.get("provenance") or {}).get("commit", "")[:8]
    runs = (len(cfg["policies"]) + 1) * len(cfg["reported_seeds"])
    seeds = ", ".join(str(x) for x in cfg["reported_seeds"])
    return f"""
<header class="top">
  <span class="eyebrow">COMP 536 · Milestone 3 · Task 2 · Phase 1</span>
  <h1>Mixed-Workload Reuse Study</h1>
  <p class="prose">Seven kinds of LLM traffic share one prefix cache: a
  memory that keeps the model's work on recent prompts (131,072 tokens,
  roughly 100,000 words) so a new prompt that starts the same way can skip
  it. We replayed the traffic through vLLM with three eviction policies and
  measured, workload by workload, how soon cached blocks are needed again,
  how much of each prompt the cache can serve, and whether a server can tell
  the workloads apart. Each finding ends with what it means for the eviction
  policy we design next.</p>
  <p class="prose">When the cache is full, an eviction policy picks which
  idle block to drop. <b>LRU</b>, vLLM's default, drops the block idle the
  longest; <b>LFU</b> the block used the fewest times; <b>FIFO</b> the block
  that became idle first. A <b>freed</b> block is one no running request is
  using; it stays cached until it is evicted.</p>
  <div class="meta"><span>{runs} reported engine runs on seeds {seeds}
  (three independent replays of different traffic)</span>
  <span>Qwen2.5-0.5B-Instruct on vLLM's CPU simulator</span><span>16-token
  blocks</span><span>branch milestone3_task2 · commit <span
  class="num">{esc(commit)}</span></span></div>
</header>"""


def example(s: Row, data: Row) -> str:
    return _example_strip(s)


def answers(s: Row, data: Row) -> str:
    h_lo, h_hi = _range(s, HUMAN)
    a_lo, a_hi = _range(s, AGENTS)
    keep = data["retention"]["lru"]["p50"]
    lfu_keep = data["retention"]["lfu"]["p50"]
    ceil, lru = _u(s, "ceiling", "all"), _u(s, "lru", "all")
    lfu, fifo = _u(s, "lfu", "all"), _u(s, "fifo", "all")
    lfu_word = _lfu_phrase(s)[0]
    idn = (data.get("identify") or {}).get("numbers") or {}
    age = (data.get("signals") or {}).get("by_age", {})
    q3 = ""
    if idn:
        q3 = (f"The first two blocks of a prompt, a fingerprint of its "
              f"system prompt or task, identify {pct(idn['fingerprint'])} of "
              f"test requests without a mistake. A few if-then rules on "
              f"prompt length, turn number and timing identify the rest, for "
              f"{pct(idn['accuracy'])} overall. After training once on one "
              "labelled half hour, it needs no labels while serving; prompt "
              "families never seen in training fall to the rules.")
    design = ""
    if age:
        hb, lf, lo = (age["+hits_before"], age["label_free"],
                      age["label_free+oracle"])
        design = (f"Among blocks that have sat idle equally long, which is "
                  f"all LRU can see, past hits pick out the ones that will "
                  f"come back (AUC {hb['0']:.2f} for blocks just freed, "
                  f"{hb['30']:.2f} after 30 idle seconds; 0.5 is a coin "
                  f"flip). The true workload label adds "
                  f"{lo['0'] - lf['0']:.2f} for fresh blocks and up to "
                  f"{max(lo['30'] - lf['30'], lo['120'] - lf['120']):.2f} "
                  "for blocks idle 30 seconds to 2 minutes.")
    return f"""
<section class="answers" aria-label="Short answers">
  <div class="answer"><span class="q">Question 1 · reuse interval</span>
  <h3>Agents return in a blink; people return in minutes</h3>
  <p>Agent loops reuse their session's blocks a median of
  <mark>{secs(a_lo)} to {secs(a_hi)}</mark> after the previous use. People
  come back to their own conversation after a median of
  <mark>{secs(h_lo)} to {secs(h_hi)}</mark>. At this cache size LRU keeps a
  freed block for about {secs(keep)} and LFU for about {secs(lfu_keep)}, so
  the agents are served and most human returns find their history gone.
  ShareGPT's and AgentBank's pauses are borrowed from recorded traces (see
  What we replayed), and the timing is from a CPU run.</p></div>
  <div class="answer"><span class="q">Question 2 · reuse portion</span>
  <h3>The cache could serve {pct(ceil)} of prompt tokens; LRU serves
  {pct(lru)}</h3>
  <p>A cache that never evicts would serve {pct(ceil)} of all prompt
  tokens. LRU serves {pct(lru)}; LFU ({pct(lfu)}) {lfu_word}; FIFO serves
  {pct(fifo)}. CC-Bench keeps {pct(_u(s, "lru", "ccbench"))} of a possible
  {pct(_u(s, "ceiling", "ccbench"))} under LRU. ShareGPT keeps
  {pct(_u(s, "lru", "sharegpt"))} of a possible
  {pct(_u(s, "ceiling", "sharegpt"))}.</p></div>
  <div class="answer">
  <span class="q">Question 3 · telling workloads apart</span>
  <h3>A cheap prompt fingerprint identifies most requests</h3>
  <p>{q3}</p></div>
  <div class="answer"><span class="q">For the design</span>
  <h3>Keep blocks by their chance of coming back, not by age alone</h3>
  <p>{design}</p></div>
</section>"""


def glossary(s: Row, data: Row) -> str:
    return """
<section class="section" aria-labelledby="terms"><h2 id="terms">Terms used on
this page</h2>
<dl class="glossary">
  <dt>KV cache block</dt><dd>The model's working memory for 16 tokens of a
  prompt. Computing it is the expensive part of reading a prompt; keeping it
  lets a later prompt that starts the same way skip that work.</dd>
  <dt>Prefix cache</dt><dd>vLLM keeps finished blocks after a request ends
  and reuses them when a new prompt starts with the same tokens. Reuse stops
  at the first block that is not cached: the blocks after it must be
  recomputed even if they are still in memory.</dd>
  <dt>Usable reuse</dt><dd>The share of prompt tokens served from the cache
  before that first missing block. It is the number every chart calls
  "reuse".</dd>
  <dt>Eviction policy</dt><dd>When the cache is full, the rule that picks
  which idle block to throw away. <b>LRU</b> drops the block idle the
  longest; <b>LFU</b> drops the block used the fewest times; <b>FIFO</b>
  drops the block that became idle first.</dd>
  <dt>No-eviction ceiling</dt><dd>The same traffic replayed on a cache too
  big to ever fill. It shows how much reuse each workload offers at all, so
  it measures the workload, not a policy.</dd>
  <dt>Reuse interval</dt><dd>The time between two uses of the same block.
  The <b>idle time</b> is the part of it the block spends unused in the
  cache, which is what an eviction policy sees.</dd>
  <dt>Session</dt><dd>One conversation, one agent task or one production
  user thread: the requests that build on each other.</dd>
  <dt>Reuse distance</dt><dd>The number of other blocks used between two
  uses of a block. LRU keeps a block roughly when this is below the cache's
  8,192 blocks.</dd>
  <dt>Working set</dt><dd>All the blocks that will be used again. A cache
  that holds them all never needs to choose what to evict.</dd>
  <dt>Prompt family</dt><dd>Requests whose prompts start with the same
  blocks, such as one system prompt or one agent task.</dd>
  <dt>Hit density</dt><dd>Reuses a block earns per minute it sits idle in
  the cache.</dd>
  <dt>AUC</dt><dd>Pick one block that comes back and one that does not: the
  AUC is how often a signal ranks the returning one higher. 0.5 is a coin
  flip, 1.0 is always right.</dd>
  <dt>True label</dt><dd>The workload a request really came from. Our
  replay knows it; a real server does not.</dd>
  <dt>Measured / cache replay</dt><dd>Measured numbers come from real
  <code>vllm serve</code> runs, with vLLM's CPU simulator standing in for
  the GPU, on seeds 2, 3 and 4 (three independent replays of different
  traffic). Cache-replay numbers come from our fast replay of vLLM's own
  cache code and only guide the design; they are marked.</dd>
</dl></section>"""


# Sections ------------------------------------------------------------------


def mix(s: Row, data: Row) -> str:
    m = data["mix"]
    return f"""
<section class="section" aria-labelledby="mix"><h2 id="mix">What we
replayed</h2>
<p class="prose">Each run replays 30 minutes of traffic at about 3 requests
per second, one seventh of the requests from each workload. The first 10
minutes warm the cache and the last 20 are measured. Requests are equal,
tokens are not: CC-Bench agent prompts are long, so it brings
{pct(m["ccbench"]["tokens"])} of all prompt tokens.</p>
{figure("fig-mix", "Who brings the work", ["input:inputs to the study"],
        "Equal request shares hide very unequal cache demand. Any per-workload "
        "rule must be judged by tokens and blocks, not by requests.",
        note="Pauses between turns: ShareGPT borrows real gaps from the "
        "Bailian chat trace, and AgentBank borrows CC-Bench's recorded tool "
        "times, because their datasets have no timestamps. Their timing is "
        "an input to the study; the Bailian traces' and CC-Bench's timing is "
        "measured from real use. The Bailian traces are sampled down to fit "
        "the mix, which thins their reuse across users (see Checks).")}
</section>"""


def _tiles(s: Row) -> str:
    lfu_v, fifo_v = _vs(s, "lfu"), _vs(s, "fifo")
    return "".join(
        f'<div class="tile"><span class="label">{label}</span><span '
        f'class="value">{pct(_u(s, p, "all"))}</span><span class="sub">'
        f'{sub}</span></div>' for p, label, sub in [
            ("ceiling", "No eviction", "what the traffic allows"),
            ("lru", "LRU", "vLLM's default"),
            ("lfu", "LFU", pts(lfu_v["mean"]) + " vs LRU" +
             (", a tie" if _tie(lfu_v) else "")),
            ("fifo", "FIFO", pts(fifo_v["mean"]) + " vs LRU"),
        ])


def _cut(served: List[Optional[float]], edges: List[float],
         below: float) -> Optional[float]:
    """The lower edge of the first interval bin whose served share falls
    below `below`, or None if none does."""
    for v, e in zip(served, edges):
        if v is not None and v < below:
            return e
    return None


def _tokens(n: float) -> str:
    return (f"{n / 1048576:g}M" if n >= 1048576 else f"{n / 1024:g}K")


def _verdicts(seeds_a: List[List[float]],
              seeds_b: List[List[float]]) -> List[str]:
    """Per cache size, "a", "b" or "tie": a difference counts only when
    every seed agrees on its sign and the mean exceeds half a point."""
    out = []
    for xa, xb in zip(seeds_a, seeds_b):
        if not xa or not xb or None in xa or None in xb:
            out.append("tie")
            continue
        d = [p - q for p, q in zip(xa, xb)]
        mean = sum(d) / len(d)
        if abs(mean) >= 0.005 and (min(d) > 0 or max(d) < 0):
            out.append("a" if mean > 0 else "b")
        else:
            out.append("tie")
    return out


def _runs(caps: List[int], verdicts: List[str], want: str) -> List[str]:
    """Contiguous stretches of cache sizes with the verdict `want`, as
    "48K" or "256K to 1M"."""
    out, start = [], None
    for i, v in enumerate(verdicts + ["end"]):
        if v == want and start is None:
            start = i
        elif v != want and start is not None:
            lo, hi = caps[start], caps[i - 1]
            out.append(
                _tokens(lo) if lo == hi else f"{_tokens(lo)} to {_tokens(hi)}")
            start = None
    return out


def capacity_figure(s: Row, data: Row) -> str:
    """Cache size against hit rate on the mix: the cache replay's curves
    with the engine runs on top."""
    C = data.get("capCurve")
    if not C:
        return ""
    caps, cv, ceil = C["capacities"], C["curve"], C["ceiling"]
    lru, lfu, fifo = (cv[p]["hit_rate"] for p in ("lru", "lfu", "fifo"))
    lfu_v = _verdicts(cv["lfu"]["seeds"], cv["lru"]["seeds"])
    fifo_v = _verdicts(cv["fifo"]["seeds"], cv["lru"]["seeds"])
    lfu_wins, lru_wins = _runs(caps, lfu_v, "a"), _runs(caps, lfu_v, "b")
    fifo_all_behind = all(v == "b" for v in fifo_v)
    fifo_gap = [
        100 * (q - p)
        for xs, ys in zip(cv["fifo"]["seeds"], cv["lru"]["seeds"])
        for p, q in zip(xs, ys)
    ]
    fifo_txt = ("FIFO trails LRU at every size"
                if fifo_all_behind else "FIFO mostly trails LRU")
    lfu_txt = (
        (f"LFU beats LRU only at {' and '.join(lfu_wins)}"
         if lfu_wins else "LFU never beats LRU") +
        (f" and trails it from {' and '.join(lru_wins)}" if lru_wins else ""))
    title = f"{fifo_txt}; {lfu_txt} tokens"
    parts = []
    if C["fixed"] in caps:
        i = caps.index(C["fixed"])
        parts.append(
            f"At the fixed cache ({caps[i]:,} tokens) LRU serves "
            f"{pct(lru[i])}, LFU {pct(lfu[i])} and FIFO {pct(fifo[i])} of "
            "prompt tokens over the whole run.")
        if 2 * caps[i] in caps:
            j = caps.index(2 * caps[i])
            parts.append(f"Doubling it gives LRU {pts(lru[j] - lru[i])}.")
    parts.append(
        "Compared seed by seed (a difference counts only when all three "
        "seeds agree and it exceeds half a point): "
        f"{fifo_txt}, by "
        f"{min(fifo_gap):.0f} to {max(fifo_gap):.0f} points; "
        f"{lfu_txt}; elsewhere LFU and LRU tie.")
    if ceil is not None:
        near = next((caps[k] for k in range(len(caps))
                     if lru[k] is not None and ceil - lru[k] < 0.02), None)
        parts.append(
            f"LRU comes within 2 points of the no-eviction rate "
            f"({pct(ceil)}) only at {_tokens(near)} tokens, "
            f"{near / C['fixed']:.0f} times the fixed cache; below that the "
            "cache cannot hold the working set and the policy matters."
            if near else f"Even at {_tokens(caps[-1])} tokens LRU stays "
            f"more than 2 points below the no-eviction rate ({pct(ceil)}).")
    eng = [e for e in C["engine"] if e["replay"] is not None]
    if eng:
        diff = max(abs(e["engine"] - e["replay"]) for e in eng)
        sizes = sorted({e["capacity"] for e in eng})
        parts.append(
            f"The {len(eng)} engine runs at "
            f"{', '.join(_tokens(c) for c in sizes)} tokens land within "
            f"{100 * diff:.1f} points of the replay for the same plan, so the "
            f"curve can be trusted at least from {_tokens(sizes[0])} to "
            f"{_tokens(sizes[-1])}.")
    return figure(
        "fig-capcurve",
        title,
        ["sim:cache replay, seeds 2–4", "measured:engine runs (markers)"],
        " ".join(parts),
        note="Hit rate: the share of prompt tokens served from the cache, "
        "what vLLM reports as cached tokens. Here it counts the whole "
        "30-minute run, warm-up included, as the cache replay does, so the "
        "fixed-cache values sit a little below the totals above, which "
        "leave out the first 10 minutes. Lines: our cache replay at "
        f"{len(caps)} sizes (1K = 1,024 tokens), averaged over seeds 2 to 4. "
        "Hollow markers: single engine runs (vllm serve).")


def baselines(s: Row, data: Row) -> str:
    M = data.get("mixRuns")
    if not M:
        return ""
    ceil, lru = _u(s, "ceiling", "all"), _u(s, "lru", "all")
    lfu, fifo = _u(s, "lfu", "all"), _u(s, "fifo", "all")
    eff = M["efficiency"]
    lfu_v, fifo_v = _vs(s, "lfu"), _vs(s, "fifo")
    lfu_overall = _lfu_phrase(s)[0]
    fifo_word = ("ties with LRU" if _tie(fifo_v) else "beats LRU on every seed"
                 if fifo_v["min"] > 0 else "trails LRU on every seed"
                 if fifo_v["max"] < 0 else "is mixed against LRU")

    # Over the run: where LRU first falls 5 points below the no-eviction run.
    mins, tl = M["minutes"], M["timeline"]
    split = next((m for m, c, x in zip(mins, tl["ceiling"], tl["lru"])
                  if c is not None and x is not None and c - x > 0.05), None)
    meas = [i for i, m in enumerate(mins) if m >= M["w0"]]

    def count(a: str, b: str) -> int:
        return sum(1 for i in meas if tl[a][i] is not None
                   and tl[b][i] is not None and tl[a][i] >= tl[b][i])

    gaps = [
        tl["ceiling"][i] - tl["lru"][i] for i in meas
        if tl["ceiling"][i] is not None and tl["lru"][i] is not None
    ]
    n_close = int(split) if split is not None else 0
    split_txt = (("Only in the first minute" if n_close == 1 else
                  f"Only in the first {n_close} minutes") +
                 " does every policy serve about "
                 "what the no-eviction run does; then evictions start to cost "
                 "reuse and the lines separate. " if n_close else "")
    lf, lfi = count("lru", "fifo"), count("lru", "lfu")
    time_take = (
        f"{split_txt}In every measured minute LRU stays "
        f"{pts(min(gaps)).lstrip('+')} to {pts(max(gaps)).lstrip('+')} below "
        "the no-eviction run. " +
        (f"LRU is above FIFO in all {len(meas)} measured minutes"
         if lf == len(meas) else f"LRU is above FIFO in {lf} of {len(meas)} "
         "measured minutes") + f", but above LFU in only {lfi} of them; LFU "
        f"is ahead in the other {len(meas) - lfi}. The ups and downs come from "
        "the traffic, since the no-eviction line moves with them.")

    # Reuse interval for the mix.
    IV, cdf = M["interval"], M["intervalCdf"]
    q, within = cdf.get("quantiles") or {}, cdf.get("within") or {}
    c90 = {p: _cut(IV["served"][p], IV["edges"], 0.9) for p in IV["served"]}
    late = {
        p: max((v for v, e in zip(IV["served"][p], IV["edges"])
                if e >= 30 and v is not None),
               default=0.0)
        for p in IV["served"]
    }
    by_cut: Dict[float, List[str]] = {}
    for p in ("lru", "lfu", "fifo"):
        if c90[p] is not None:
            by_cut.setdefault(c90[p], []).append(p.upper())
    cut_txt = "; ".join(
        f"{' and '.join(ps)} serve"
        f"{'s' if len(ps) == 1 else ''} at least 90% of reuses "
        f"up to {secs(c)}" for c, ps in sorted(by_cut.items(), reverse=True))
    keeps_late = [p.upper() for p in late if late[p] >= 0.05]
    late_txt = (
        f" {' and '.join(keeps_late)} still serve"
        f"{'s' if len(keeps_late) == 1 else ''} some reuses after "
        f"30 s (up to {pct(max(late[p.lower()] for p in keeps_late))}), "
        "because blocks with many past hits stay; "
        f"{' and '.join(p.upper() for p in late if late[p] < 0.05)} "
        "serve almost none." if keeps_late else "")
    int_take = (
        f"Half the mix's reuses come within {secs(q.get('p50'))} of the "
        f"block's previous use and {pct(within.get('60'))} within a minute "
        f"(left). {cut_txt} (right).{late_txt} The "
        f"{pct(1 - within.get('60', 1))} of reuses that come back after a "
        "minute are mostly lost under every baseline.")

    ret = data["retention"]
    age_take = (
        "Median idle time at eviction: " + ", ".join(
            f"{p.upper()} {secs(M['evictP50'][p])}"
            for p in sorted(M["evictP50"], key=lambda p: -M["evictP50"][p])) +
        f". Under every policy 90% of evicted blocks had been idle less "
        f"than {secs(max(ret[p]['p90'] for p in ret))}, far shorter than "
        "the minutes people take to come back.")

    L = data["lost"]

    def kept(p: str) -> float:
        t = L[p]["all"]
        return t["kept"] / max(1, t["kept"] + t["evicted"] + t["cut_off"])

    f_all = L["fifo"]["all"]
    cut_share = f_all["cut_off"] / max(1, sum(f_all.values()))
    lost_take = (
        "Of the blocks the no-eviction run reuses, " +
        ", ".join(f"{p.upper()} reuses {pct(kept(p))}"
                  for p in ("lru", "lfu", "fifo")) +
        ". LRU and LFU lose blocks only by evicting them; FIFO also leaves "
        f"{pct(cut_share)} cached but unusable behind an evicted block.")
    return f"""
<section class="section" aria-labelledby="baselines"><h2 id="baselines">
Baselines on the mixed workload: LRU, LFU and FIFO</h2>
<p class="prose">This is the experiment the task asks for. All seven
workloads are replayed together, interleaved in time, through one shared
cache of 131,072 tokens, once per policy. Within a seed every policy replays
exactly the same requests, so the differences come only from what each
policy evicts. This section treats the mix as one stream of requests; the
sections after it split the same runs by workload.</p>
<div class="tiles">{_tiles(s)}</div>
{figure("fig-mixbars", f"On the mix, LRU serves {pct(lru)} of prompt "
        f"tokens, LFU {pct(lfu)} and FIFO {pct(fifo)}; with no eviction, "
        f"{pct(ceil)}", ["measured:measured, 3 seeds"],
        f"LRU keeps {pct(eff['lru'])} of what the mix offers, LFU "
        f"{pct(eff['lfu'])} and FIFO {pct(eff['fifo'])}. LFU {lfu_overall} "
        f"({pts(lfu_v['min'])} to {pts(lfu_v['max'])} per seed); FIFO "
        f"{fifo_word} ({pts(fifo_v['min'])} to {pts(fifo_v['max'])}). The "
        "new policy has to raise this number without giving up any one "
        "workload, which the sections below check.")}
{capacity_figure(s, data)}
{figure("fig-timeline", "Every policy falls below the no-eviction run "
        "within minutes and stays there", ["measured:measured, 3 seeds"],
        time_take,
        note="Each point is the share of prompt tokens served from the "
        "cache for the requests sent in that minute, all workloads "
        "together, averaged over the three seeds. The first 10 minutes "
        "warm the cache and are not counted in the totals.")}
{figure("fig-mixinterval", "On the mix, a reuse is served only if it "
        "comes back quickly", ["measured:measured, 3 seeds"], int_take,
        note="Reuse interval: the time between two uses of the same "
        "block. Left: how the mix's reuses spread over that time, in runs "
        "that never evict. Right: for each policy, the share of reuses "
        "whose block was still cached, by the same time.")}
{figure("fig-mixages", "How long each policy keeps an idle block",
        ["measured:measured, 3 seeds"], age_take)}
{figure("fig-mixlost", "Where the mix's lost reuse goes",
        ["measured:measured, 3 seeds"], lost_take)}
</section>"""


def headline(s: Row, data: Row) -> str:
    lfu_v, fifo_v = _vs(s, "lfu"), _vs(s, "fifo")

    def word(v: Row) -> str:
        return ("ties with"
                if _tie(v) else "beats" if v["mean"] > 0 else "trails")

    lfu_overall, lfu_loses = _lfu_phrase(s)
    claim = (f"LFU {lfu_overall}"
             f"{' but loses ' + lfu_loses if lfu_loses else ''}; FIFO "
             f"{word(fifo_v)} LRU; all three are far from the ceiling")
    lab, sg = _labels(s), "sharegpt"
    wins = [
        lab[w] for w in s["config"]["workloads"]
        if min(s["headline"]["lfu"][w]["vs_lru"]["seeds"]) > 0
    ]
    win_txt = (f", yet beats LRU on {' and '.join(wins)} in every seed"
               if wins else "")
    lead = ("Overall ties hide large swings per workload"
            if _tie(lfu_v) else "The totals hide large swings per workload")
    return f"""
<section class="section" aria-labelledby="headline"><h2 id="headline">The
same runs, workload by workload</h2>
<p class="prose">The mixed runs above, split by the workload each request
came from. Each mark is that workload's share of prompt tokens served from
the cache, averaged over the three reported seeds; the thin line through
each mark spans the seeds. The top row is the whole mix again. The grey bar
is what LRU leaves on the table compared with a cache that never
evicts.</p>
{figure("fig-headline", claim, ["measured:measured, 3 seeds"],
        f"{lead}: LFU serves "
        f"{lab[sg]} {pct(_u(s, 'lfu', sg))} against LRU's "
        f"{pct(_u(s, 'lru', sg))}{win_txt}. The agents keep most of their "
        f"reuse under every policy (CC-Bench {pct(_u(s, 'lru', 'ccbench'))} "
        f"of a possible {pct(_u(s, 'ceiling', 'ccbench'))} under LRU); the "
        "human-paced workloads lose most of theirs. The new policy has to "
        "win back human returns without giving up the agents' reuse, and it "
        "must be judged per workload.")}
</section>"""


def q1(s: Row, data: Row) -> str:
    keep = data["retention"]["lru"]
    lab = _labels(s)
    W = data["workloads"]
    t = 300.0
    seen = [w for w in W if _came_back(data, w, t) is not None]
    low = sorted([w for w in seen if _came_back(data, w, t) < 0.5],
                 key=lambda w: _came_back(data, w, t))
    low_txt = ("Many blocks are not needed again soon: within 5 minutes "
               "only " + ", ".join(f"{lab[w]} {pct(_came_back(data, w, t))}"
                                   for w in low) +
               " of freed blocks were needed again. ") if low else ""
    far = [lab[w] for w in W if (data["distance"][w] or 0) > 0.5]
    far_txt = (f"That is the case for most reuses of {', '.join(far)}. "
               if far else "")
    return f"""
<section class="section" aria-labelledby="q1"><h2 id="q1">Question 1: how
soon do blocks come back?</h2>
{figure("fig-comeback", "Workloads differ in return time by hundreds of "
        "times", ["measured:measured, no-eviction runs"],
        f"LRU keeps a freed block for {secs(keep['p10'])} to "
        f"{secs(keep['p90'])} (the middle 80% of its evictions). Agent "
        "blocks come back well inside that window; most human returns come "
        f"after it. {low_txt}A policy that keeps human history longer must "
        "find the space by dropping blocks that will not be used again.",
        note="Each line is the share of freed blocks needed again within "
        "the time on the axis, in runs whose cache never fills. A line that "
        "stays low marks blocks that are rarely reused. A line stops where "
        "fewer than 1% of a workload's blocks could still be followed before "
        "the run ended. The labels mark the median time each policy kept an "
        "idle block before evicting it.")}
{figure("fig-hitinterval", "LRU serves almost every reuse under 10 seconds "
        "and none after a minute", ["measured:measured, 3 seeds"],
        "Under LRU every workload's line falls at the same place: LRU sees "
        "only time, not who the block belongs to. LFU evicts a block that "
        "has no hits yet within seconds, so it loses even quick reuses of "
        "fresh conversation history; that is what wipes out ShareGPT under "
        "LFU. FIFO behaves like LRU on time.",
        note="For each reuse in the policy's own run: whether the block was "
        "still cached, by how long it had been since the block's previous "
        "use. Reuses of a block that another running request was holding "
        "are left out, since no policy can lose them.")}
{figure("fig-distance", "Most human-paced reuses are out of reach for an "
        "LRU cache of this size", ["measured:measured, no-eviction runs"],
        "When the reuse distance is below the cache's 8,192 blocks, LRU "
        "roughly keeps the block; above it, LRU cannot (blocks held by "
        "running requests make the usable cache a little smaller). "
        f"{far_txt}Winning them back means choosing differently which "
        "8,192 blocks to keep.",
        note="Reuse distance: for each reuse, the number of other distinct "
        "blocks used since the block's previous use.")}
</section>"""


def q2(s: Row, data: Row) -> str:
    lab = _labels(s)
    W = data["workloads"]
    d = data["density"]["lru"]
    heavy = sorted(W, key=lambda w: -d[w]["pool"])[:2]
    pool = d[heavy[0]]["pool"] + d[heavy[1]]["pool"]
    rev = d[heavy[0]]["revive"] + d[heavy[1]]["revive"]
    ev = data["evictions"]["lru"]["counts"]
    tot = sum(map(sum, ev)) or 1
    driver = sorted(range(len(W)), key=lambda i: -sum(ev[i]))[:2]
    cross = 1 - sum(ev[i][i] for i in range(len(W))) / tot
    causes = [sum(ev[i]) / tot for i in range(len(W))]
    suffers = [
        sum(ev[i][j] for i in range(len(W))) / tot for j in range(len(W))
    ]
    chance = 1 - sum(a * b for a, b in zip(causes, suffers))
    balanced = max(abs(a - b) for a, b in zip(causes, suffers)) < 0.02
    src = s["q2"]["sources"]
    shared = [w for w in W if (src[w]["other_session"] or 0) > 0.5]
    own = [w for w in W if (src[w]["same_session"] or 0) > 0.5]
    shared_txt = ", ".join(f"{lab[w]} {pct(src[w]['other_session'])}"
                           for w in shared)
    fifo = data["lost"]["fifo"]["all"]
    cut = fifo["cut_off"] / max(1, sum(fifo.values()))
    cut_lost = fifo["cut_off"] / max(1, fifo["cut_off"] + fifo["evicted"])
    return f"""
<section class="section" aria-labelledby="q2"><h2 id="q2">Question 2: how
much of each prompt is reused?</h2>
{figure("fig-sources", "Most workloads reuse their own history; "
        f"{', '.join(lab[w] for w in shared)} reuse mostly other users' "
        "prompts", ["measured:measured, no-eviction runs"],
        f"For {', '.join(lab[w] for w in own)}, most reuse is the session's "
        f"own history. For the others it is prompts shared across users "
        f"({shared_txt}). Shared prompts are few blocks read by many users; "
        "history is many blocks read by one. They need different "
        "treatment. Thinning the Bailian traces to fit the mix lowers their "
        "shared reuse; the full-rate traces are in the paper section.")}
{figure("fig-lost", "Where the lost reuse goes", ["measured:measured, 3 "
        "seeds"],
        f"Under LRU and LFU every lost block was evicted. FIFO also leaves "
        f"{pct(cut)} of the reusable blocks cached but unusable behind an "
        f"evicted block, {pct(cut_lost)} of all it loses. A new policy must "
        "never evict a block while the blocks after it in the same prompt "
        "stay cached.",
        note="Blocks that the no-eviction run reused, split by what happened "
        "to them under each policy.")}
{figure("fig-position", "Production prompts share their head; histories "
        "are reused all the way down", ["measured:measured, no-eviction "
        "runs"],
        "In the Bailian traces the first blocks (system prompts) are reused "
        "far more than later ones, which hold one user's text. In "
        "conversation and agent histories nearly every block is reused. "
        "Evicting a prompt's deepest blocks first, as vLLM already does "
        "between blocks idle equally long, suits the first kind.")}
{figure("fig-density", "Some workloads fill the idle cache without using "
        "it", ["measured:measured, LRU, 3 seeds"],
        f"Under LRU, {lab[heavy[0]]} and {lab[heavy[1]]} hold {pct(pool)} of "
        f"the idle cache's time but earn {pct(rev)} of its reuses. CC-Bench "
        f"earns {pct(d['ccbench']['revive'])} from "
        f"{pct(d['ccbench']['pool'])}. That idle time is what a better "
        "policy can hand to blocks that will be used. The Bailian traces "
        "are thinned to fit the mix, which removes part of their reuse "
        "across users, so their waste here is somewhat overstated.")}
{figure("fig-evictions", "Who evicts whom", ["measured:measured, 3 seeds"],
        f"{lab[W[driver[0]]]} and {lab[W[driver[1]]]} cause the most "
        f"evictions, and {pct(cross)} of all evictions remove another "
        f"workload's block, the same as random mixing would give "
        f"({pct(chance)})." + (" Each workload causes about as many "
        "evictions as it suffers: evictions follow block demand, with no "
        "workload preying on another. Dropping blocks that never return "
        "therefore costs no one." if balanced else " Every rule must be "
        "checked per workload, not only in total."),
        note="Rows: the workload whose new blocks needed space. Columns: "
        "the workload whose idle block was evicted. Each cell is a share of "
        "all evictions.")}
</section>"""


def q3(s: Row, data: Row) -> str:
    idn = (data.get("identify") or {}).get("numbers") or {}
    sig = data.get("signals") or {}
    lab = _labels(s)
    if not idn:
        return ""
    confused = "; ".join(f"{pct(x)} of {lab[a]} requests look like {lab[b]}"
                         for x, a, b in idn["worst"])
    age, base = sig.get("by_age", {}), sig.get("base", {})
    base_txt = ""
    if base:
        b0, b30 = base["0"], base["30"]
        base_txt = (f"Of blocks just freed, {pct(b0['ccbench'])} of "
                    f"CC-Bench's and {pct(b0['bailian-thinking'])} of Bailian "
                    f"thinking's come back within a minute; after 30 idle "
                    f"seconds, {pct(b30['ccbench'])} and "
                    f"{pct(b30['bailian-thinking'])}. Each mark compares "
                    "blocks that have sat idle equally long, so idle time, "
                    "the only thing LRU looks at, cannot separate them (it "
                    "would score 0.5).")
    sig_take = ""
    if age:
        hb, ow = age["+hits_before"], age["+oracle_workload"]
        lf, lo = age["label_free"], age["label_free+oracle"]
        beats = [a for a in ("0", "5", "30", "120") if hb[a] > ow[a]]
        when = ("at every idle time"
                if len(beats) == 4 else f"for blocks idle up to {beats[-1]} s"
                if beats and beats == ["0", "5", "30", "120"][:len(beats)] else
                "at some idle times")
        sig_take = (f"Past hits are the best single signal {when}: AUC "
                    f"{hb['0']:.2f} for blocks just freed and {hb['30']:.2f} "
                    f"after 30 idle seconds, against {ow['0']:.2f} and "
                    f"{ow['30']:.2f} for the true workload label. All "
                    f"label-free signals together reach {lf['0']:.2f} and "
                    f"{lf['30']:.2f}; adding the true label gains "
                    f"{lo['0'] - lf['0']:.2f} for fresh blocks and "
                    f"{lo['30'] - lf['30']:.2f} after 30 s. LFU counts past "
                    "hits too, but evicts fresh blocks before they have had "
                    "a chance to be reused.")
    return f"""
<section class="section" aria-labelledby="q3"><h2 id="q3">Question 3: can a
server tell which workload a request belongs to?</h2>
<p class="prose">A server sees only tokens and timing. We gave a classifier
only what a server could know when a request arrives: whether each block's
hash, which the cache computes anyway, was seen before, how long prompts are
and how they grow, and the time since the session's last turn. It was
trained once on seed 1, where the true labels are known, and tested on seeds
2 to 4 with no labels. The test seeds replay later half hours of the same
Bailian traces, so system prompts seen in training come back; a new
deployment would need its own training pass.</p>
{figure("fig-ladder", "A prompt fingerprint identifies most requests; a "
        "few if-then rules identify the rest", ["measured:plans of seeds "
        "2–4, trained on seed 1"],
        f"The fingerprint resolves {pct(idn['fingerprint'])} of requests "
        f"with no errors. With the rules, accuracy is {pct(idn['accuracy'])} "
        f"({pct(idn['first'])} for a session's first request, "
        f"{pct(idn['later'])} for later ones). The inferred parent request "
        f"is the true previous turn for at least {pct(idn['parent'])} of "
        "later turns, so a policy can follow a session without any label.")}
{figure("fig-confusion", "Which workloads get confused", ["measured:plans "
        "of seeds 2–4"],
        f"A session's first request is wrong {pct(1 - idn['first'])} of the "
        f"time, a later one {pct(1 - idn['later'])}. The largest confusions: "
        f"{confused}. The rules' guess when unsure is "
        f"{lab.get(idn.get('fallback'), idn.get('fallback'))}: "
        f"{pct(idn.get('fallback_wrong'))} of requests it names that way are "
        "something else. A policy should treat that name as weak evidence "
        "and revise it as the session's next turns arrive.")}
{figure("fig-signals", "What predicts that an idle block will come back",
        ["measured:no-eviction runs, trained on seed 1"], sig_take,
        note=base_txt)}
</section>"""


def paper(s: Row, data: Row) -> str:
    rows = data.get("paper") or []
    body = "".join(
        f"<tr><td>{esc(r['claim'])}</td><td>{esc(r['paper'])}</td>"
        f"<td>{esc(r['raw'])}</td><td>{esc(r['mix'])}</td>"
        f'<td><span class="chip {r["chip"]}">{esc(r["verdict"])}</span>'
        f"</td></tr>" for r in rows)
    return f"""
<section class="section" aria-labelledby="paper"><h2 id="paper">Do the
paper's findings hold here?</h2>
<p class="prose">The Bailian traces come from "KVCache Cache in the Wild"
(Wang et al., USENIX ATC '25). We tested ten of its claims on the four raw
traces at full rate (A is chat, B is API; the coder and thinking traces are
newer and not in the paper) and in our mix. The verdict is on traces A and
B. KS, used in some cells, is the largest gap between a fitted curve and the
real one: 0 is a perfect fit, 1 the worst.</p>
<div class="tablewrap"><table class="signals"><thead><tr><th>Claim</th>
<th>Paper</th><th>Raw traces</th><th>Our mix</th><th>Verdict</th></tr>
</thead><tbody>{body}</tbody></table></div>
<p class="takeaway"><b>For the design</b>Two findings change the plan. Past
use does predict future use, so frequency belongs in the policy, with
protection for blocks too young to have been reused. And reuse times within
one category are not exponential: shared prompts come back within seconds
and history within minutes, so a single curve per category, as the paper's
policy uses, would misjudge both.</p>
</section>"""


def headroom(s: Row, data: Row) -> str:
    b = data.get("beladySplit") or {}
    if not b:
        return ""
    gain = b["belady_all"] - b["lru_all"]
    share = b.get("token_share_of_gain", {})
    human_share = sum(v for w, v in share.items() if w in HUMAN)
    cap = data.get("capacity") or {}
    lru_curve = cap.get("curves", {}).get("lru", [])
    caps = cap.get("capacities", [])
    # The smallest swept cache at which LRU catches Belady at the fixed one.
    catch = next((c for c, v in zip(caps, lru_curve)
                  if v is not None and v >= b["belady_all"]), None)
    if catch is None and caps:
        title = (f"A policy that knew the future would beat "
                 f"{caps[-1] / cap['fixed']:.0f} times the memory")
        size_txt = (f" LRU does not reach that even with "
                    f"{caps[-1]:,} tokens, {caps[-1] / cap['fixed']:.0f} "
                    "times the cache.")
    elif catch:
        k = catch / cap["fixed"]
        prev = max((c for c in caps if c < catch), default=None)
        lo = prev / cap["fixed"] if prev else None
        title = ((f"LRU needs {lo:g} to {k:g} times the memory" if lo
                  and lo >= 1 else f"LRU needs up to {k:g} times the memory") +
                 " to match a policy that knew the future")
        size_txt = (f" LRU falls short of that at {prev:,} tokens and "
                    f"reaches it at {catch:,} tokens, {k:g} times the cache."
                    if prev else f" LRU reaches that at {catch:,} tokens.")
    else:
        title = "A policy that knew the future would equal a bigger cache"
        size_txt = ""
    gaps = [
        bel - lru for bel, lru in zip(
            cap.get("curves", {}).get("belady", []), lru_curve)
        if bel is not None and lru is not None
    ]
    gap_txt = ""
    if gaps:
        gap_txt = (f" Across the sizes tried the gap runs from "
                   f"{100 * max(gaps):.0f} points down to "
                   f"{100 * gaps[-1]:.0f} at {caps[-1]:,} tokens" +
                   (", where the cache holds nearly the whole working set and "
                    "the policy stops mattering." if gaps[-1] < 0.02 else
                    ": even the largest cache tried holds less than the mix's "
                    "working set."))
        bel = cap.get("curves", {}).get("belady", [])
        full = next((c for c, v in zip(caps, bel)
                     if v is not None and cap.get("ceiling") is not None
                     and v >= cap["ceiling"] - 0.002), None)
        lru_full = next((c for c, v in zip(caps, lru_curve)
                         if v is not None and cap.get("ceiling") is not None
                         and v >= cap["ceiling"] - 0.01), None)
        if full and lru_full and lru_full > full:
            gap_txt += (f" Belady already serves everything the traffic "
                        f"allows at {full:,} tokens, while LRU needs about "
                        f"{lru_full:,}: most of LRU's extra memory holds "
                        "blocks that never come back.")
    return f"""
<section class="section" aria-labelledby="headroom"><h2 id="headroom">How
much room is there?</h2>
<p class="prose">Belady's rule evicts the block needed again furthest in
the future. It needs to know the future, so no real server can run it; it
serves as a near-best reference for a given cache size. These numbers come
from our cache replay, which runs vLLM's own cache code and matched the
engine runs closely (see Checks).</p>
{figure("fig-capacity", title, ["sim:cache replay"],
        f"At the fixed cache, Belady serves {pct(b['belady_all'])} against "
        f"LRU's {pct(b['lru_all'])}: {pts(gain)} are there for a better "
        f"policy to win, and a real policy will win back only part of it."
        f"{size_txt}{gap_txt} These runs average seeds 1 to 4 with the "
        f"replay's own timing, so LRU reads {pct(b['lru_all'])} here "
        f"against {pct(_u(s, 'lru', 'all'))} in the engine runs above.")}
{figure("fig-belady", "Where the headroom is", ["sim:cache replay"],
        f"Counted in tokens, {pct(human_share)} of Belady's gain is in the "
        f"human-paced workloads and {pct(share.get('ccbench', 0))} in "
        "CC-Bench alone, whose long prompts make a few points worth many "
        "tokens. Belady maximises the total, so its split across workloads "
        "is one good allocation, not a limit for each workload.")}
</section>"""


def signals(s: Row, data: Row) -> str:
    lab = _labels(s)
    h_lo, h_hi = _range(s, HUMAN)
    a_lo, a_hi = _range(s, AGENTS)
    keep = data["retention"]["lru"]["p50"]
    d = data["density"]["lru"]
    fr = data.get("freq_ceiling") or {}
    age = (data.get("signals") or {}).get("by_age", {})
    idn = (data.get("identify") or {}).get("numbers") or {}
    src = s["q2"]["sources"]
    label_row = ""
    if idn and age:
        lf, lo = age["label_free"], age["label_free+oracle"]
        late = max(lo["30"] - lf["30"], lo["120"] - lf["120"])
        label_row = (f"After training on one labelled half hour, a "
                     f"fingerprint plus a small tree names "
                     f"{pct(idn['accuracy'])} of requests. Among blocks idle "
                     f"equally long, the true label adds "
                     f"{lo['0'] - lf['0']:.2f} AUC over label-free signals "
                     f"for fresh blocks and up to {late:.2f} for blocks idle "
                     "30 s to 2 min.")
    hb = age.get("+hits_before", {})
    it = age.get("+inferred_turn", {})
    turn_ev = ((f"Among blocks idle equally long, the inferred turn ranks "
                f"returners with AUC {it['0']:.2f} just freed and "
                f"{it['30']:.2f} after 30 s. " if it else "") +
               "In the Bailian chat trace the chance of a next turn rises "
               "from turn 1 to turn 6+ (see the paper section).")
    rows = [
        ("How soon a session returns",
         f"Agents return in {secs(a_lo)} to {secs(a_hi)} (median); people in "
         f"{secs(h_lo)} to {secs(h_hi)}. LRU keeps idle blocks about "
         f"{secs(keep)}.",
         "Give a session's history a retention time that fits its loop: "
         "short for agent loops, minutes for people, dropped once the "
         "session has been quiet well past its usual gap.",
         "No: the time since the session's previous turn is visible to the "
         "server.",
         "Holding human history uses space the agents could have used. A "
         "server cannot see when a session ends, and a first turn has no gap "
         "history. The human gaps are borrowed for ShareGPT, and pacing "
         "decides who LRU serves (see Checks)."),
        ("Turn number", turn_ev,
         "Trust a session's history more as its turn count grows.",
         "No: the turn is inferred from the matching earlier prompt.",
         "A long session can still end at any turn."),
        ("Past hits",
         (f"In the no-eviction runs, blocks with 0, 1, 2 and "
          f"3+ past uses were used again {pct(fr.get('0'))}, "
          f"{pct(fr.get('1'))}, {pct(fr.get('2'))} and {pct(fr.get('3+'))} "
          "of the time (mean of seeds 2–4)" if fr else "No-eviction runs "
          "missing") +
         (f"; among blocks idle equally long, AUC "
          f"{hb['0']:.2f} just freed and {hb['30']:.2f} after 30 s."
          if hb else "."),
         "Rank idle blocks by recent hits, decayed over time, and protect "
         "blocks younger than their session's usual return time.", "No.",
         "Counts go stale; LFU without protection collapses on fresh "
         "history."),
        ("Dead blocks", "Within 5 minutes only " + ", ".join(
            f"{lab[w]} {pct(_came_back(data, w, 300))}"
            for w in ("bailian-thinking", "bailian-api", "bailian-chat")) +
         " of freed blocks came back. Under LRU, Bailian thinking holds "
         f"{pct(d['bailian-thinking']['pool'])} of idle cache time and earns "
         f"{pct(d['bailian-thinking']['revive'])} of reuses.",
         "Evict first the blocks whose prompt family or class rarely "
         "returns: rank by expected hits per unit of cache time (hit "
         "density).",
         "A class helps; prompt-family statistics work without one.",
         "A rare long return, such as a user back after an hour, is lost. "
         "Thinning removed part of the Bailian traces' reuse across users, "
         "so the rule may evict shared Bailian prompts too eagerly."),
        ("Shared prompt heads", "Share of reuse across users: " +
         ", ".join(f"{lab[w]} {pct(src[w]['other_session'])}"
                   for w in ("bailian-api", "bailian-thinking",
                             "bailian-coder", "bailian-chat")) +
         ". Shared heads are few blocks with many readers.",
         "Keep a prompt family's first blocks while the family is popular; "
         "evict deep blocks first.",
         "No: block position and family popularity are visible.",
         "A burst of one family can crowd out others."),
        ("Prefix closure",
         "FIFO leaves reusable blocks cached but unusable behind an evicted "
         "block; LRU and LFU never do.",
         "Never evict a block while a later block of the same prompt stays "
         "cached (the spec's prefix-closure rule).", "No.",
         "Restricts which blocks a smarter score can pick."),
        ("Who the request is", label_row,
         "Run label-free, with true labels only as the upper bound to "
         "compare against (the spec's label modes).", "No, by design.",
         "A session's first request is the least certain, and the "
         "fingerprints come from the same traces' earlier half hour."),
    ]
    body = "".join("<tr>" + "".join(f"<td>{esc(x)}</td>" for x in r) + "</tr>"
                   for r in rows)
    return f"""
<section class="section" aria-labelledby="signals"><h2 id="signals">Signals
for the policy design</h2>
<p class="prose">What the measurements suggest a workload-aware policy
should use, the evidence for each, and what could go wrong. Phase 2 turns
these into one scoring rule and tests it against LRU, LFU and FIFO on new
seeds.</p>
<div class="tablewrap"><table class="signals"><thead><tr><th>Signal</th>
<th>Evidence</th><th>Rule it suggests</th><th>Needs the workload
label?</th><th>Risk</th></tr></thead><tbody>{body}</tbody></table></div>
</section>"""


def thinning(s: Row) -> str:
    """The Bailian thinning sentence, from the plans and the paper report."""
    seeds = [str(x) for x in s["config"]["reported_seeds"]]
    lab = _labels(s)
    kept = {}
    for w in ("bailian-api", "bailian-coder", "bailian-chat",
              "bailian-thinking"):
        vals = [
            s["mix"][k]["workloads"][w]["kept_session_fraction"] for k in seeds
        ]
        kept[w] = sum(vals) / len(vals)
    text = ("Thinning: to fit the mix, each Bailian trace keeps only part of "
            "its sessions (seed average: " +
            ", ".join(f"{lab[w]} {pct(v, 1)}" for w, v in kept.items()) + ").")
    thin = (s.get("paper") or {}).get("thinning") or {}
    ratio = [
        thin[f"{s['config']['mix']}-{s['config']['pacing']}-s{k}"]
        ["bailian-api"]["kept_over_full_hit_ratio"] for k in seeds
        if f"{s['config']['mix']}-{s['config']['pacing']}-s{k}" in thin
    ]
    if ratio:
        loss = 1 - sum(ratio) / len(ratio)
        text += (f" That removes {pct(loss)} of the API trace's reuse in the "
                 "same window, all of it reuse across users, so the Bailian "
                 "reuse here is a lower bound of the production value.")
    return text


def checks(s: Row, data: Row) -> str:
    items: List[str] = []
    integ = s.get("integrity") or {}
    if integ:
        same = all(v["ceiling_lookups"] == v.get("lru_lookups") == v.get(
            "lfu_lookups") == v.get("fifo_lookups") for v in integ.values())
        ev = sum(v["ceiling_evictions"] for v in integ.values())
        items.append(
            "Same requests for every policy: within each seed, every run "
            f"looked up {'exactly the same' if same else 'a different'} "
            "number of prompt blocks, and the no-eviction runs evicted "
            f"{ev} blocks.")
    val = s.get("validation") or {}
    if val.get("runs"):
        items.append(
            "Cache replay against engine: on the same request plans, the "
            "replay's usable reuse is within "
            f"{100 * val['max_diff']:.1f} points of the engine runs overall.")
    vt = s.get("validation_engine_timing") or []
    if vt:
        diff = max(abs(r["engine"] - r["sim"]) for r in vt)
        overlap = min(r["evicted_overlap"] for r in vt)
        items.append(
            "Cache replay with the engine's own timing (seed 1, LRU, LFU and "
            "FIFO): the same prompt blocks, usable reuse within "
            f"{100 * diff:.2f} points, and {pct(overlap, 3)} of evicted "
            "blocks the same.")
    sens = s.get("sensitivity") or {}
    noise = sens.get("noise")
    if noise:
        items.append(
            "Run-to-run noise: replaying seed 2 under LRU a second time moved "
            f"usable reuse by {pts(noise['all']['diff'])} overall.")
    load = sens.get("load") or []
    if load:
        by = {r["run"]: r for r in load}
        lat_l = max(r["loaded_latency_p50"] for r in load)
        lat_q = max(r["quiet_latency_p50"] for r in load)
        fifo = [r for r in load if "-fifo-" in r["run"]]
        rest = [r for r in load if "-fifo-" not in r["run"]]
        moved = max(abs(r["quiet_usable"] - r["loaded_usable"]) for r in rest)
        fifo_moved = max(
            (abs(r["quiet_usable"] - r["loaded_usable"]) for r in fifo),
            default=None)
        fifo_txt = (f"; FIFO moved by up to {100 * fifo_moved:.1f} points"
                    if fifo_moved is not None else "")
        flip = ""
        lru4 = by.get("balanced-human-s4-lru-cap131072")
        lfu4 = by.get("balanced-human-s4-lfu-cap131072")
        if lru4 and lfu4:
            before = lfu4["loaded_usable"] - lru4["loaded_usable"]
            after = lfu4["quiet_usable"] - lru4["quiet_usable"]
            if before * after < 0:
                flip = (f" Seed 4's LFU minus LRU was {pts(before)} under "
                        f"load and {pts(after)} quiet, so LFU's small deficit "
                        "to LRU is within the replay's timing noise.")
        items.append(
            f"Machine load: {len(by)} runs first ran while analysis jobs "
            f"loaded the laptop (median request latency up to "
            f"{1000 * lat_l:.0f} ms instead of {1000 * lat_q:.0f} ms) and "
            "were replayed on a quiet machine; the page reports the quiet "
            "runs. LRU, LFU and the ceiling moved by at most "
            f"{100 * moved:.1f} points{fifo_txt}, so FIFO is the most "
            f"timing-sensitive.{flip}")
    warm = sens.get("warmup") or {}
    lru = [
        warm[k]["lru"]["mean"] for k in sorted(warm, key=float)
        if warm[k].get("lru")
    ]
    if lru:
        items.append("Warm-up: measuring from 5, 10 or 15 minutes gives LRU "
                     f"{', '.join(pct(x) for x in lru)}.")
    pace = sens.get("pacing") or {}
    if (pace.get("fast") or {}).get("lru") and pace["human"].get("lru"):
        hp, fp = pace["human"]["lru"], pace["fast"]["lru"]
        items.append(
            "Pacing (seed 2): with the earlier setup's fixed 5 s pause for "
            "ShareGPT instead of real human gaps, LRU serves ShareGPT "
            f"{pct(fp['sharegpt']['fraction'])} instead of "
            f"{pct(hp['sharegpt']['fraction'])}. The pacing decides who LRU "
            "serves, which is why real gaps are used.")
    items += [
        thinning(s),
        "Synthetic tokens: Bailian prompts are rebuilt from hashed block ids, "
        "so equal blocks match exactly but the text is random; nothing here "
        "depends on the text.",
        "CC-Bench trajectories do not include the agent's tool definitions, "
        "which real coding agents send with every request, so its shared "
        "prefix is understated.",
        "LFU here has no aging; an aged LFU behaves differently.",
        "Timing comes from vLLM's CPU simulator: requests take tens of "
        "milliseconds instead of seconds on a GPU, which matters only for "
        "the sub-second agent gaps.",
        "Three reported seeds: a difference whose sign the seeds disagree on "
        "is called a tie.",
    ]
    lis = "".join(f"<li>{x}</li>" for x in items)
    return f"""
<section class="section" aria-labelledby="checks"><h2 id="checks">Checks
and limits</h2>
<ul class="prose stack">{lis}</ul>
</section>"""


def method(s: Row, data: Row) -> str:
    cfg = s["config"]
    items = [
        "Workloads: ShareGPT conversations, AgentBank agent trajectories, "
        "CC-Bench coding-agent sessions, and four Bailian production traces "
        "(chat, API, thinking, coder) from alibaba-edu/"
        "qwen-bailian-usagetraces-anon.",
        "Plans: analysis/mix.py builds a 30-minute plan per seed with one "
        "seventh of the requests from each workload and whole sessions. "
        "Bailian seed k replays the trace's k-th half hour. No turn is "
        "planned after the 30 minutes, so every policy replays the same "
        "requests.",
        "Engine runs: vllm serve with the simulator executor, Qwen2.5-0.5B-"
        f"Instruct, a {cfg['capacity_tokens']:,}-token cache of 16-token "
        "blocks, the cache event log on, and the client replaying the plan "
        "on its own clock. The ceiling uses a cache of 16.8M tokens.",
        "Tables: analysis/tables.py turns each run's event log into every "
        "block lookup, every idle stay of a block and every request; "
        "analysis/m3t2_study.py computes this page's numbers from them.",
        "Identification and signals: analysis/identify.py and "
        "analysis/signals.py, trained on seed 1 and tested on seeds 2 to 4.",
        "Paper checks: analysis/paper_checks.py on the raw traces at full "
        "rate and on our no-eviction runs.",
        "Headroom: analysis/cachesim.py replays the plans through vLLM's own "
        "allocator; analysis/oracle.py adds Belady's offline rule.",
        "Reproduce: python -m analysis.run_m3_task2 plans --grid phase1; "
        "python -m analysis.run_m3_task2 grid --grid phase1; python -m "
        "analysis.tables results/m3-task2/*; python -m analysis.m3t2_study; "
        "python -m analysis.m3t2_report.",
    ]
    lis = "".join(f"<li>{esc(x)}</li>" for x in items)
    return f"""
<section class="section" aria-labelledby="method"><details><summary><h2
id="method" style="display:inline">How it was measured</h2></summary>
<ul class="prose stack">{lis}</ul></details></section>"""


def sections(s: Row, data: Row) -> Dict[str, str]:
    return {
        "header": header(s, data),
        "answers": answers(s, data),
        "example": example(s, data),
        "glossary": glossary(s, data),
        "mix": mix(s, data),
        "baselines": baselines(s, data),
        "headline": headline(s, data),
        "q1": q1(s, data),
        "q2": q2(s, data),
        "q3": q3(s, data),
        "paper": paper(s, data),
        "headroom": headroom(s, data),
        "signals": signals(s, data),
        "checks": checks(s, data),
        "method": method(s, data),
    }
