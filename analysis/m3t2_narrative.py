"""The results page's words (``analysis/m3t2_report.py``).

Every number below is read from ``summary.json`` or from the chart data
built from it, and every comparison is worded from the numbers: a
difference counts only when all seeds agree on its sign and its mean is at
least half a point (spec 007 SC-003); otherwise it is called a tie. The
page answers six questions; `sections` returns them in order.

Words: an *engine run* is the real vLLM server (measured); the *cache
replay* is our simulator of vLLM's cache code (simulated). "Replay" is kept
for the second.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

from analysis.m3t2_report import (GROUPS, LABELS, MIN_OFFERED, MODELS,
                                  POLICIES, POLICY_LABELS, esc, figure, mean,
                                  pct, pts, secs, tokens)
from analysis.m3t2_study import MIN_GAP

Row = Dict[str, Any]
P = POLICY_LABELS
L = LABELS
# Reuse that waits at least this long counts as a long wait (seconds).
LONG_WAIT_S = 10.0
# A policy "serves almost nothing" beyond the first interval bin from which
# every drawn bin's share served is under this.
NONE_SHARE = 0.01
# Interval lines closer than this in every drawn bin are said to coincide.
COINCIDE = 0.005
TEXT_WORKLOADS = ("sharegpt", "agentbank", "ccbench")
BAILIAN = ("bailian-chat", "bailian-api", "bailian-thinking", "bailian-coder")


def _list(items: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c'."""
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _verb(items: Sequence[str], one: str, many: str) -> str:
    return one if len(items) == 1 else many


def _span(lo: float, hi: float) -> str:
    """Two differences as '4.2 to 5.0 points'."""
    a, b = pts(lo).split()[0], pts(hi)
    return b if a == b.split()[0] else f"{a} to {b}"


def _ul(items: Sequence[str]) -> str:
    return '<ul class="prose">' + "".join(f"<li>{i}</li>"
                                          for i in items) + "</ul>"


def _bins(view: Row) -> List[Row]:
    return view["by_bin"]


def _range_label(b: Row) -> str:
    if b["hi"] is None:
        return f"more than {secs(b['lo'])}"
    if b["lo"] <= 0:
        return f"under {secs(b['hi'])}"
    return f"{secs(b['lo'])} to {secs(b['hi'])}"


def _served_beyond(curve: List[Row], lo: float) -> Optional[float]:
    """Share served of the reuses that waited at least `lo` seconds."""
    off = sum(b["offered"] for b in curve if b["lo"] >= lo)
    srv = sum(b["served"] for b in curve if b["lo"] >= lo)
    return srv / off if off else None


def _cutoff(curve: List[Row]) -> Optional[float]:
    """The wait (a bin's lower edge, seconds) from which a policy serves
    under `NONE_SHARE` of the reuses in every later drawn bin; None if it
    never stops serving."""
    drawn = [b for b in curve if b["offered"] >= MIN_OFFERED]
    for i, b in enumerate(drawn):
        if all((x["share"] or 0) < NONE_SHARE for x in drawn[i:]):
            return b["lo"]
    return None


def _coincide(views: Row) -> bool:
    """Whether every policy's share served is within `COINCIDE` of the
    others in every drawn interval bin."""
    curves = [views[p]["curve"] for p in POLICIES]
    for i, b in enumerate(curves[0]):
        if b["hi"] is None or any(c[i]["offered"] < MIN_OFFERED
                                  for c in curves):
            continue
        shares = [c[i]["share"] or 0.0 for c in curves]
        if max(shares) - min(shares) > COINCIDE:
            return False
    return True


def _cap_verdicts(seeds_a: List[List[float]],
                  seeds_b: List[List[float]]) -> List[str]:
    """Per cache size, "a", "b" or "tie" (the module's rule)."""
    out = []
    for xa, xb in zip(seeds_a, seeds_b):
        if not xa or not xb or None in xa or None in xb:
            out.append("tie")
            continue
        d = [p - q for p, q in zip(xa, xb)]
        m = sum(d) / len(d)
        if abs(m) >= MIN_GAP and (min(d) > 0 or max(d) < 0):
            out.append("a" if m > 0 else "b")
        else:
            out.append("tie")
    return out


def _cap_runs(caps: List[int], verdicts: List[str], want: str) -> List[str]:
    """Stretches of cache sizes with verdict `want`: '48K', '96K to 384K'."""
    out, start = [], None
    for i, v in enumerate(verdicts + ["end"]):
        if v == want and start is None:
            start = i
        elif v != want and start is not None:
            lo, hi = caps[start], caps[i - 1]
            out.append(
                tokens(lo) if lo == hi else f"{tokens(lo)} to {tokens(hi)}")
            start = None
    return out


def _against_lru(verdicts: Dict[str, str]) -> str:
    """Policies' verdicts against LRU in one clause: 'LFU and FIFO are lower
    than LRU', 'LFU beats LRU and FIFO ties with it'."""
    by = {
        v: [P[p] for p, x in verdicts.items() if x == v]
        for v in ("higher", "lower", "tie")
    }
    parts = []
    if by["higher"]:
        parts.append(f"{_list(by['higher'])} "
                     f"{_verb(by['higher'], 'beats', 'beat')} LRU")
    if by["lower"]:
        parts.append(f"{_list(by['lower'])} "
                     f"{_verb(by['lower'], 'is', 'are')} lower than LRU")
    if by["tie"]:
        parts.append(f"{_list(by['tie'])} "
                     f"{_verb(by['tie'], 'ties', 'tie')} with it"
                     if parts else f"{_list(by['tie'])} "
                     f"{_verb(by['tie'], 'ties', 'tie')} with LRU")
    return _list(parts)


def _best_text(vs: Row) -> str:
    """Which policy is best for one workload alone, by the rule."""
    v = {p: vs[p]["verdict"] for p in ("lfu", "fifo")}
    higher = [P[p] for p in v if v[p] == "higher"]
    if higher:
        return f"{_list(higher)} {_verb(higher, 'beats', 'beat')} LRU"
    ties = [P[p] for p in v if v[p] == "tie"]
    if not ties:
        return "LRU is best"
    if len(ties) == len(v):
        return "all three tie"
    return f"LRU and {_list(ties)} tie for best"


def _seeds_txt(s: Row) -> str:
    return _list([str(x) for x in s["config"]["reported_seeds"]])


def _seed_range(s: Row) -> str:
    seeds = s["config"]["reported_seeds"]
    return (f"{seeds[0]}–{seeds[-1]}" if seeds == list(
        range(seeds[0], seeds[-1] + 1)) else _seeds_txt(s))


def badges(s: Row) -> Dict[str, str]:
    """The badge of each kind of number, naming the seeds."""
    seeds = f"seeds {_seed_range(s)}"
    return {
        "engine": f"measured:engine runs, {seeds}",
        "offline": f"offline:computed from the planned requests, {seeds}",
        "replay": f"sim:cache replay (simulated), {seeds}",
        "markers": "measured:engine runs (markers)",
    }


# --- Facts the text quotes --------------------------------------------------


def wait_split(s: Row) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    """The share of the mix's reuses (no eviction) that wait at least
    `LONG_WAIT_S`, and each workload's share of the quick and of the long
    waits."""
    by = s["ceiling_intervals_by_group"]
    long = {
        g: sum(b["offered"] for b in _bins(by[g]) if b["lo"] >= LONG_WAIT_S)
        for g in GROUPS
    }
    quick = {g: by[g]["offered"] - long[g] for g in GROUPS}
    total = sum(v["offered"] for v in by.values())
    n = sum(long.values())
    return (n / total, {g: v / (total - n)
                        for g, v in quick.items()},
            {g: v / n
             for g, v in long.items()})


def lost_split(s: Row, policy: str = "lru") -> Tuple[float, Dict[str, float]]:
    """The points of the mix's prompt tokens `policy` loses against no
    eviction, and each workload's part of them (token-weighted)."""
    mg = s["mixed_groups"]
    total = mg["all"]["prompt_tokens"]
    parts = {
        g: (mean(mg[g]["ceiling"]["usable"]) - mean(mg[g][policy]["usable"])) *
        mg[g]["prompt_tokens"] / total
        for g in GROUPS
    }
    return (mean(mg["all"]["ceiling"]["usable"]) -
            mean(mg["all"][policy]["usable"])), parts


def _token_shares(s: Row) -> Dict[str, float]:
    mg = s["mixed_groups"]
    return {
        g: mg[g]["prompt_tokens"] / mg["all"]["prompt_tokens"]
        for g in GROUPS
    }


def _members(group: str) -> Tuple[str, ...]:
    return BAILIAN if group == "bailian" else (group, )


def _mix_mean(s: Row, group: str, field: str) -> float:
    """A workload's mean over the reported seeds of a plan-manifest count
    (requests, sessions), its traces summed."""
    seeds = [str(x) for x in s["config"]["reported_seeds"]]
    mix = s["mix"]
    vals = [
        sum(mix[x]["workloads"][w][field] for w in _members(group))
        for x in seeds
    ]
    return sum(vals) / len(vals)


# --- Sections ---------------------------------------------------------------


def header(s: Row) -> str:
    cfg = s["config"]
    cap = cfg["capacity_tokens"]
    w0, stop = cfg["w0_s"] / 60, cfg["stop_s"] / 60
    return f"""<header class="top">
  <span class="eyebrow">COMP 536 · Milestone 3 · Task 2 · Results</span>
  <h1>How four workloads share one KV cache</h1>
  <p class="prose">We ran four workloads against one vLLM server:
  <b>ShareGPT</b> (people chatting), <b>AgentBank</b> (agents acting in an
  environment), <b>CC-Bench</b> (a coding agent calling tools) and
  <b>Bailian</b> (four production traces from Alibaba's model service). They
  ran together on a KV cache of {cap:,} tokens, and then each one alone,
  under three eviction policies: LRU, LFU and FIFO. This page answers six
  questions about what each policy keeps.</p>
  <div class="meta"><span>Qwen2.5-0.5B-Instruct on vLLM v0, with the GPU
  replaced by a stand-in (see the last section)</span><span>{cap:,}-token
  cache ({cap // cfg["block_size"]:,} blocks of {cfg["block_size"]}
  tokens)</span><span>{stop:.0f}-minute runs, measured over minutes
  {w0:.0f}–{stop:.0f}</span><span>seeds {_seeds_txt(s)}</span></div>
</header>"""


def answers(facts: Row) -> str:
    cards = [
        ("1 · Reuse time interval", facts["q1_answer"]),
        ("2 · Reuse portion", facts["q2_answer"]),
        ("3 · Which workload sent a request", facts["q3_answer"]),
        ("4 · Cache capacity", facts["q4_answer"]),
        ("5 · Each workload alone", facts["q5_answer"]),
        ("6 · Alone vs. sharing", facts["q6_answer"]),
    ]
    body = "".join(f'<div class="answer"><span class="q">{q}</span><p>{a}</p>'
                   f'<a href="#point-{q[0]}">See point {q[0]}</a></div>'
                   for q, a in cards)
    return (f'<section class="section" aria-labelledby="answers-h">'
            f'<h2 id="answers-h">The six answers</h2>'
            f'<div class="answers">{body}</div></section>')


def terms(s: Row) -> str:
    cfg = s["config"]
    bs = cfg["block_size"]
    rows = [
        ("KV cache", "Memory where the server keeps what it computed for each "
         f"prompt token, in blocks of {bs} tokens. A later request whose "
         "prompt starts with the same tokens can reuse those blocks instead "
         "of computing them again."),
        ("Reuse", "A request needs a block that an earlier request produced "
         "(the system prompt, or the previous turns of the same "
         f"conversation). Each reuse is one {bs}-token block."),
        ("Reuse time interval", "How long the block sat unused before it was "
         "needed again."),
        ("Usable prefix reuse", "The share of prompt tokens served from the "
         "cache, counting only the unbroken run of cached blocks from the "
         "start of each prompt (the server stops at the first missing block). "
         "Weighted by tokens. This is vLLM's prefix cache hit rate."),
        ("Points", "Percentage points: 55.8% against 50.3% is 5.5 points."),
        ("No eviction (ceiling)", "An engine run of the same requests on a "
         "cache too large to ever evict: all the reuse the traffic offers."),
        ("LRU, LFU, FIFO", "When room is needed, which cached block (among "
         "those no running request is using) to throw out: the one used least "
         "recently (LRU), the one reused the fewest times (LFU), or the one "
         "cached earliest, whatever happened since (FIFO)."),
        ("Belady", "An offline reference that knows the future: it throws out "
         "the block needed farthest ahead. No real server can do this. It is "
         "the best possible for plain hit counts; for usable prefix reuse it "
         "is a strong reference, not a proven upper bound."),
        ("Engine run (measured)", "The real vLLM server, its scheduler and "
         "cache code unchanged, with the model replaced by a stand-in that "
         "returns recorded answers and estimates how long each step takes. "
         "What the cache keeps is what vLLM really did; times are "
         "estimates."),
        ("Cache replay (simulated)", "Only vLLM's cache code, fed the same "
         "requests offline, so it can try many cache sizes and run Belady. "
         "It is labelled wherever it is used."),
        ("Seeds and ties",
         f"Independent {cfg['stop_s'] / 60:.0f}-minute mixes (seeds "
         f"{_seeds_txt(s)}); numbers are their mean. A difference counts "
         "only if every seed agrees on its direction and it is at least "
         f"{100 * MIN_GAP:.1f} points; otherwise it is a tie."),
    ]
    dl = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in rows)
    return (f'<section class="section" aria-labelledby="terms-h">'
            f'<h2 id="terms-h">Terms used on this page</h2>'
            f'<dl class="glossary">{dl}</dl></section>')


def part(title: str, text: str) -> str:
    return (f'<div class="part"><span class="eyebrow">{title}</span>'
            f'<h2>{text}</h2></div>')


def point(n: int, kind: str, question: str, body: str) -> str:
    return (f'<section class="section" id="point-{n}" '
            f'aria-labelledby="point-{n}-h"><span class="eyebrow">Point {n} '
            f'of 6 · {kind}</span><h2 id="point-{n}-h">{question}</h2>'
            f'{body}</section>')


def q1(s: Row, facts: Row) -> str:
    mi = s["mixed_intervals"]
    curves = {p: mi[p]["curve"] for p in POLICIES}
    coarse = {p: _bins(mi[p]) for p in POLICIES}
    cut = {p: _cutoff(curves[p]) for p in POLICIES}
    long_share, quick_by, long_by = wait_split(s)
    quick = {p: coarse[p][0]["share"] or 0.0 for p in POLICIES}
    lru_cut = cut["lru"]
    facts["lru_cut"] = lru_cut
    if lru_cut is None:
        beyond: Dict[str, Optional[float]] = {p: None for p in POLICIES}
        none: List[str] = []
        keeps: List[str] = []
        claim = "LRU serves some reuse at every wait"
    else:
        beyond = {p: _served_beyond(curves[p], lru_cut) for p in POLICIES}
        none = [p for p in POLICIES if (beyond[p] or 0) < NONE_SHARE]
        keeps = [p for p in POLICIES if p not in none]
        names = [P[p] for p in none]
        claim = (f"{_list(names)} {_verb(names, 'serves', 'serve')} almost "
                 f"no reuse that waited longer than {secs(lru_cut)}")
        if keeps:
            claim += "; " + _list([
                f"{P[p]} still serves {pct(beyond[p], 0)} of them"
                for p in keeps
            ])
    order = sorted(GROUPS, key=lambda g: -long_by[g])
    top_quick = max(GROUPS, key=lambda g: quick_by[g])
    intro = (
        "For every reuse in the mix we know how long the block had sat "
        f"unused. With no eviction, {pct(1 - long_share, 0)} of reuses happen "
        f"within {secs(LONG_WAIT_S)} of the block's last use, "
        f"{pct(quick_by[top_quick], 0)} of them {L[top_quick]}'s. Of the "
        f"{pct(long_share, 0)} that wait longer, " +
        _list([f"{pct(long_by[g], 0)} are {L[g]}'s" for g in order]) +
        ". The chart sorts reuses by how long they waited and shows the share "
        f"each policy served from its {s['config']['capacity_tokens']:,}-"
        "token cache. Blocks still in use by another running request are "
        "left out (no policy can throw those out), and waits with fewer than "
        f"{MIN_OFFERED:,} reuses are not drawn.")
    head = ["Policy", *[_range_label(b) for b in coarse["lru"]]]
    table = ('<div class="tablewrap"><table><thead><tr>' +
             "".join(f"<th>{h}</th>" for h in head) + "</tr></thead><tbody>" +
             "".join(f"<tr><td>{P[p]}</td>" +
                     "".join(f"<td>{pct(b['share'], 1)}</td>"
                             for b in coarse[p]) + "</tr>"
                     for p in POLICIES) + "</tbody></table></div>")
    why = [
        "Under LRU an unused block is thrown out once newer blocks have "
        "filled the cache behind it" +
        (f", which in the mix took under {secs(lru_cut)}."
         if lru_cut is not None else "."),
        "LFU throws out the block reused the fewest times, so an often-used "
        "block can outlive a long pause, while a block just cached (not yet "
        "reused) goes first" +
        ("; so it serves fewer of the quick reuses than LRU."
         if quick["lfu"] < quick["lru"] else "."),
        "FIFO throws out the block cached earliest, even one used a moment "
        "ago.",
    ]
    take = (f'<p class="prose">Share of reuses served, by how long the block '
            f'waited:</p>{table}<p class="prose">{" ".join(why)}</p>')
    drawn = [
        i for i, b in enumerate(curves["lru"])
        if b["offered"] >= MIN_OFFERED and (b["share"] or 0) >= NONE_SHARE
    ]
    fifo_below = all(
        (curves["fifo"][i]["share"] or 0) < (curves["lru"][i]["share"] or 0)
        for i in drawn)
    if fifo_below and lru_cut is not None and "fifo" in none:
        fifo_txt = (f"FIFO serves less than LRU at every wait up to "
                    f"{secs(lru_cut)}, and almost none after.")
    elif fifo_below:
        fifo_txt = "FIFO serves less than LRU at every wait LRU serves."
    else:
        fifo_txt = ""
    head_txt = (
        f"With the four workloads sharing the cache, LRU serves "
        f"{pct(quick['lru'], 1)} of reuses that happen within "
        f"{secs(LONG_WAIT_S)} of the block's last use, but " +
        (f"almost none of those that waited more than {secs(lru_cut)}."
         if lru_cut is not None else "also some at every longer wait."))
    tail = ""
    if keeps:
        tail = (f"{_list([P[p] for p in keeps])} still "
                f"{_verb(keeps, 'serves', 'serve')} " +
                _list([pct(beyond[p], 0) for p in keeps]) +
                f" of those that waited over {secs(lru_cut)}")
        if "lfu" in keeps and quick["lfu"] < quick["lru"]:
            tail += (("," if keeps == ["lfu"] else ", LFU") +
                     f" but only {pct(quick['lfu'], 1)} of the quick ones")
        tail += "."
    facts["q1_answer"] = " ".join(x for x in (head_txt, fifo_txt, tail) if x)
    body = (f'<p class="prose">{intro}</p>' +
            figure("fig-q1", esc(claim), [badges(s)["engine"]], take))
    return point(
        1, "Mixed workload", "What is the difference in reuse time "
        "interval between FIFO, LRU and LFU?", body)


def q2(s: Row, facts: Row) -> str:
    mg = s["mixed_groups"]
    u = {
        k: {p: mean(mg[k][p]["usable"])
            for p in (*POLICIES, "ceiling")}
        for k in ("all", *GROUPS)
    }
    a = u["all"]
    claim = (f"LRU serves {pct(a['lru'], 1)} of the mix's prompt tokens from "
             f"the cache, LFU {pct(a['lfu'], 1)} and FIFO {pct(a['fifo'], 1)}"
             f", out of {pct(a['ceiling'], 1)} possible")
    gap, parts = lost_split(s)
    share = _token_shares(s)
    order = sorted(GROUPS, key=lambda g: -parts[g])
    eff = {g: u[g]["lru"] / u[g]["ceiling"] for g in GROUPS}
    least = min(GROUPS, key=lambda g: eff[g])
    others = [g for g in sorted(GROUPS, key=lambda g: -eff[g]) if g != least]
    cfg = s["config"]
    window = f"minutes {cfg['w0_s'] / 60:.0f}–{cfg['stop_s'] / 60:.0f}"
    intro = (
        "The share of each request's prompt that came from the cache, over "
        "the whole mix and for each workload inside it. The dashed tick in "
        "each row is the no-eviction ceiling: all the reuse that workload's "
        "traffic offers. The thin line at each bar's end spans the seeds.")
    verdicts = {p: mg["all"][p]["verdict"] for p in ("lfu", "fifo")}
    agree = any(v != "tie" for v in verdicts.values())
    lines = [
        "Over the whole mix, " + _against_lru(verdicts) +
        (f", on every seed ({_seeds_txt(s)})." if agree else "."),
        (f"LRU's usable reuse is {pts(gap)} below no eviction. By workload "
         "that is " + _list([f"{L[g]} {pts(parts[g])}" for g in order]) +
         ", weighted by the share of prompt tokens each sends in " + window +
         " (" + _list([f"{L[g]} {pct(share[g], 0)}" for g in order]) + ")."),
    ]
    p50 = s["ceiling_intervals_by_group"][least]["served"]["quantiles"]["p50"]
    cut = facts.get("lru_cut")
    least_txt = (
        f"{L[least]} keeps the smallest part of what it could: LRU serves "
        f"{pct(u[least]['lru'], 0)} of its prompt tokens from the cache, out "
        f"of {pct(u[least]['ceiling'], 0)} possible (LFU "
        f"{pct(u[least]['lfu'], 0)}, FIFO {pct(u[least]['fifo'], 0)}). Half "
        f"of its reuses wait more than {p50:.0f} s")
    if cut is not None and p50 > cut:
        least_txt += (f", longer than the {secs(cut)} after which LRU serves "
                      "almost nothing in the mix (point 1)")
    lines.append(least_txt + ".")
    lines.append("The others under LRU: " + _list([
        f"{L[g]} {pct(u[g]['lru'], 0)} of {pct(u[g]['ceiling'], 0)}"
        for g in others
    ]) + ".")
    by_policy = []
    for p in ("lfu", "fifo"):
        groups = {
            v: [L[g] for g in GROUPS if mg[g][p]["verdict"] == v]
            for v in ("higher", "lower", "tie")
        }
        bits = []
        if groups["higher"]:
            bits.append(f"beats LRU on {_list(groups['higher'])}")
        if groups["lower"]:
            bits.append(f"is lower than LRU on {_list(groups['lower'])}")
        if groups["tie"]:
            bits.append(f"ties with it on {_list(groups['tie'])}")
        by_policy.append(f"{P[p]} {_list(bits)}")
    lines.append("Workload by workload: " + "; ".join(by_policy) + ".")
    facts["q2_answer"] = (
        f"Of the mix's prompt tokens, LRU serves {pct(a['lru'], 1)} from the "
        f"cache, LFU {pct(a['lfu'], 1)} and FIFO {pct(a['fifo'], 1)}, against "
        f"{pct(a['ceiling'], 1)} with no eviction. The largest part of LRU's "
        f"shortfall is {L[order[0]]}'s ({pts(parts[order[0]]).split()[0]} of "
        f"the {pts(gap)}). LRU serves {L[least]} only "
        f"{pct(u[least]['lru'], 0)} of its prompt tokens, out of "
        f"{pct(u[least]['ceiling'], 0)} possible.")
    body = (f'<p class="prose">{intro}</p>' +
            figure("fig-q2", esc(claim), [badges(s)["engine"]], _ul(lines)))
    return point(
        2, "Mixed workload", "What is the difference in reuse "
        "portion between FIFO, LRU and LFU?", body)


def q3(s: Row, facts: Row) -> str:
    c = s["identify"]["characteristics"]
    m = c["models"]
    full, maj = m["all"], c["majority"]
    single = {k: m[k] for k in MODELS if k != "all"}
    best = max(single, key=lambda k: single[k]["accuracy"])
    worst = min(single, key=lambda k: single[k]["accuracy"])
    rec = full["recall"]
    conf = full["confusion"]
    mistakes = sorted(((t, p, n / sum(conf[t].values())) for t in conf
                       for p, n in conf[t].items() if p != t),
                      key=lambda x: -x[2])[:2]
    guess = (f"{pct(maj['accuracy'], 0)} by always guessing the most common "
             f"workload, {L[maj['label']]}")
    claim = (f"From signals a server can see, a decision tree identifies the "
             f"workload of {pct(full['accuracy'], 0)} of requests, against "
             f"{guess}")
    intro = (
        "No request carries its workload's name, so a policy has to infer "
        "it. A policy inside the server could compute four kinds of signal "
        "for each arriving request, from that request and the ones before it "
        "(never later ones); we computed them offline from the planned "
        "requests:")
    signals = [
        "<b>Prompt length</b>: the number of prompt tokens.",
        "<b>Prefix hashes</b>: a fingerprint (hash) of the prompt's first 1, "
        "2 and 4 blocks, looked up in a table of which workload sent prompts "
        "that start that way.",
        "<b>Reuse history</b>: how many of the prompt's leading blocks the "
        "server has seen before, whether it continues an earlier request and "
        "how many earlier turns of the conversation it follows, how many new "
        "tokens it adds, and how many earlier requests share its first 2 "
        "blocks.",
        "<b>Inter-arrival times</b>: the time since the request it continues, "
        "and since the last request that started with the same 2 blocks.",
    ]
    how = (
        "A decision tree (a learned flowchart of yes/no questions about the "
        "signals) turns them into a workload. Labels are used once, offline: "
        "the tree and the hash table are learned on one profiling run "
        "(seed 1), and the tree's size is chosen on that run alone. At run "
        f"time the policy sees no label. The tree is scored on seeds "
        f"{_seeds_txt(s)}, whose requests it never saw.")
    sessions = {g: _mix_mean(s, g, "sessions") for g in GROUPS}
    requests = {g: _mix_mean(s, g, "requests") for g in GROUPS}
    bailian_sessions = sessions["bailian"] / sum(sessions.values())
    lines = [
        f"Alone, {MODELS[best].lower()} is the strongest signal "
        f"({pct(single[best]['accuracy'], 0)} of requests right) and "
        f"{MODELS[worst].lower()} the weakest "
        f"({pct(single[worst]['accuracy'], 0)}). Together they reach "
        f"{pct(full['accuracy'], 1)}, or {pct(full['balanced_accuracy'], 1)} "
        "when each workload counts equally (the average of the four "
        "workloads' scores, where always guessing would get "
        f"{pct(1 / len(GROUPS), 0)}).",
        f"The first request of each conversation is identified "
        f"{pct(full['first_accuracy'], 0)} of the time, later ones "
        f"{pct(full['later_accuracy'], 0)}. Most conversations are "
        f"Bailian's ({pct(bailian_sessions, 0)} of them, at "
        f"{requests['bailian'] / sessions['bailian']:.1f} requests each on "
        f"average), and Bailian is identified {pct(rec['bailian'], 0)} of the "
        "time.",
        "Seed by seed: " + _list([
            f"seed {name.rsplit('-s', 1)[-1]} {pct(v['accuracy'], 1)}"
            for name, v in full["per_test"].items()
        ]) + ".",
        "ShareGPT and AgentBank have no timestamps of their own (their "
        "pauses are borrowed from other traces), so their timing could "
        "flatter the result. Without the inter-arrival signal the tree still "
        f"identifies {pct(m['all_but_timing']['accuracy'], 1)}.",
        "Counted by prompt tokens instead of requests, reuse history alone "
        f"({pct(m['reuse_history']['token_accuracy'], 1)}) scores about as "
        f"well as all four ({pct(full['token_accuracy'], 1)}).",
    ]
    hashes = m["prefix_hashes"]
    if full.get("ccbench_known_requests") and hashes.get(
            "ccbench_known_requests"):
        n = full["ccbench_known_requests"] + full["ccbench_new_requests"]
        lines.append(
            "CC-Bench's tasks recur across seeds, which could flatter the "
            f"hash table. Of CC-Bench's {n:,} test requests, "
            f"{full['ccbench_known_requests']:,} come from tasks seed 1 also "
            f"ran and {full['ccbench_new_requests']:,} from new tasks. Prefix "
            f"hashes alone get {pct(hashes['ccbench_known_accuracy'], 0)} and "
            f"{pct(hashes['ccbench_new_accuracy'], 0)} of them right; all "
            f"four signals get {pct(full['ccbench_known_accuracy'], 0)} and "
            f"{pct(full['ccbench_new_accuracy'], 0)}. With every signal the "
            "tree leans less on the table: it recognises fewer repeated tasks "
            "but more new ones, which is what matters for unseen traffic.")
    b = badges(s)["offline"]
    fig_a = figure("fig-q3-acc", esc(claim), [b], _ul(lines))
    mistake_txt = [
        f"{pct(x, 0)} of {L[t]}'s requests labelled {L[p]}"
        for t, p, x in mistakes
    ]
    shared = [(t, p) for t, p, _ in mistakes
              if t in TEXT_WORKLOADS and p in TEXT_WORKLOADS]
    fig_b = figure(
        "fig-q3-cm", "Where it goes wrong: " +
        _list([f"{L[t]} taken for {L[p]}" for t, p, _ in mistakes]), [b],
        _ul([
            "Share of each workload's requests identified correctly (its "
            "recall): " + _list([f"{L[g]} {pct(rec[g], 0)}"
                                 for g in GROUPS]) + ".",
            "The largest mistakes: " + _list(mistake_txt) + "." +
            (" ShareGPT, AgentBank and CC-Bench start with the same system "
             "prompt, so their first blocks look alike, which explains " +
             _list([f"{L[t]} taken for {L[p]}"
                    for t, p in shared]) + "." if shared else ""),
        ]))
    gain = full["accuracy"] - maj["accuracy"]
    facts["q3_answer"] = (
        ("It can be inferred" if gain >= 0.1 else "Only partly") +
        " without labels at run time: from prompt length, prefix hashes, "
        "reuse history and inter-arrival times, a decision tree learned on "
        f"one profiling run identifies the workload of "
        f"{pct(full['accuracy'], 0)} of requests on runs it never saw "
        f"({guess}). {MODELS[best]} is the best single signal.")
    body = (f'<p class="prose">{intro}</p>' + _ul(signals) +
            f'<p class="prose">{how}</p>' + fig_a + fig_b)
    return point(3, "Mixed workload", "How do we know which workload a "
                 "request belongs to?", body)


def q4(s: Row, data: Row, facts: Row) -> str:
    c = s["capacity_window"]
    caps, cv = c["capacities"], c["curve"]
    fixed = c["fixed"]
    i = caps.index(fixed)
    hr = {p: cv[p]["hit_rate"] for p in (*POLICIES, "belady")}
    top = len(caps) - 1
    belady_wins = hr["lru"][top] < hr["belady"][i]
    if belady_wins:
        claim = (
            f"More cache helps every policy, but LRU with "
            f"{tokens(caps[top])} tokens ({pct(hr['lru'][top], 1)}) still "
            f"serves less than Belady with {tokens(fixed)} "
            f"({pct(hr['belady'][i], 1)})")
    else:
        claim = (f"More cache helps every policy; Belady stays ahead of LRU "
                 f"by {pts(hr['belady'][top] - hr['lru'][top])} at "
                 f"{tokens(caps[top])}")
    lfu_v = _cap_verdicts(cv["lfu"]["seeds"], cv["lru"]["seeds"])
    fifo_v = _cap_verdicts(cv["fifo"]["seeds"], cv["lru"]["seeds"])
    lfu_runs = {k: _cap_runs(caps, lfu_v, k) for k in ("a", "b", "tie")}
    fifo_runs = {k: _cap_runs(caps, fifo_v, k) for k in ("a", "b", "tie")}
    fifo_gap = [hr["lru"][k] - hr["fifo"][k] for k in range(len(caps))]
    gaps = [hr["belady"][k] - hr["lru"][k] for k in range(len(caps))]
    doubled = caps.index(2 * fixed) if 2 * fixed in caps else None
    marks = data["q4"]["engine"]
    engine = [e for e in c["engine"] if e["replay"] is not None]
    worst = max((abs(e["engine"] - e["replay"]) for e in engine), default=None)
    if all(v == "b" for v in fifo_v):
        fifo_txt = (f"FIFO trails LRU at every size, by "
                    f"{_span(min(fifo_gap), max(fifo_gap))}")
    else:
        bits = []
        if fifo_runs["b"]:
            bits.append(f"trails LRU at {_list(fifo_runs['b'])}")
        if fifo_runs["a"]:
            bits.append(f"beats it at {_list(fifo_runs['a'])}")
        if fifo_runs["tie"]:
            bits.append(f"ties at {_list(fifo_runs['tie'])}")
        fifo_txt = "FIFO " + _list(bits)
    lfu_bits = [(f"beats LRU at {_list(lfu_runs['a'])}"
                 if lfu_runs["a"] else "never beats LRU")]
    if lfu_runs["b"]:
        lfu_bits.append(f"trails it at {_list(lfu_runs['b'])}")
    if lfu_runs["tie"]:
        lfu_bits.append(f"ties at {_list(lfu_runs['tie'])}, where the seeds "
                        "disagree")
    lfu_txt = "LFU " + _list(lfu_bits)
    intro = (
        f"The same mix on caches from {tokens(caps[0])} to {tokens(caps[top])} "
        "tokens (1K = 1,024 tokens). The lines are the cache replay, averaged "
        f"over seeds {_seeds_txt(s)}; it is the only way to run Belady and to "
        "try many sizes. The hollow markers are engine runs at "
        f"{_list([tokens(x) for x in sorted({e['capacity'] for e in marks})])}"
        " tokens.")
    partial = sorted({
        e["capacity"]
        for e in marks if len(e["seeds"]) < len(s["config"]["reported_seeds"])
    })
    partial_seeds = sorted(
        {x
         for e in marks if e["capacity"] in partial for x in e["seeds"]})
    lines = [
        (f"At this study's {tokens(fixed)} cache (cache replay): LRU "
         f"{pct(hr['lru'][i], 1)}, LFU {pct(hr['lfu'][i], 1)}, FIFO "
         f"{pct(hr['fifo'][i], 1)}, Belady {pct(hr['belady'][i], 1)}; no "
         f"eviction {pct(c['ceiling'], 1)}."),
        ((f"Doubling the cache to {tokens(caps[doubled])} adds "
          f"{pts(hr['lru'][doubled] - hr['lru'][i])} for LRU; " if doubled
          is not None else "") + f"going to {tokens(caps[top])} adds "
         f"{pts(hr['lru'][top] - hr['lru'][i])}."),
        f"{fifo_txt}. {lfu_txt}.",
    ]
    engine_lfu = s["mixed_groups"]["all"]["lfu"]["verdict"]
    replay_lfu = {"a": "higher", "b": "lower", "tie": "tie"}[lfu_v[i]]
    if engine_lfu != replay_lfu:
        word = {
            "higher": "higher than LRU on every seed",
            "lower": "lower than LRU on every seed",
            "tie": "tied with LRU"
        }
        lines.append(f"At {tokens(fixed)} the engine runs (point 2) find LFU "
                     f"{word[engine_lfu]}, while the cache replay finds it "
                     f"{word[replay_lfu]}.")
    lines.append(
        f"Belady's lead over LRU is {pts(gaps[0])} at {tokens(caps[0])} and "
        f"{pts(gaps[top])} at {tokens(caps[top])}: reuse a smarter policy "
        "could still win at these sizes.")
    if worst is not None:
        note = (f"The engine runs land within {pts(worst)} of the cache "
                "replay of the same seeds (× beside each marker).")
        if partial:
            note += (f" At {_list([tokens(x) for x in partial])} only seed"
                     f"{'s' if len(partial_seeds) > 1 else ''} "
                     f"{_list([str(x) for x in partial_seeds])} ran on the "
                     "engine, so those markers sit where that seed does, not "
                     "on the line averaged over all seeds.")
        lines.append(note)
    facts["q4_answer"] = (
        f"From {tokens(caps[0])} to {tokens(caps[top])}, LRU rises from "
        f"{pct(hr['lru'][0], 0)} to {pct(hr['lru'][top], 0)}, short of the "
        f"{pct(c['ceiling'], 0)} ceiling. {fifo_txt}; {lfu_txt}. " +
        (f"Belady with {tokens(fixed)} ({pct(hr['belady'][i], 0)}) beats LRU "
         f"with {tokens(caps[top])}." if belady_wins else
         f"Belady stays {pts(gaps[top])} ahead at {tokens(caps[top])}."))
    body = (
        f'<p class="prose">{intro}</p>' +
        figure("fig-q4", esc(claim),
               [badges(s)["replay"], badges(s)["markers"]], _ul(lines)))
    return point(
        4, "Mixed workload", "How sensitive is usable prefix reuse "
        "to the KV-cache capacity?", body)


SAME_TIMES = (
    "the same requests, session starts and pauses as in the mix (for the "
    "closed-loop workloads, ShareGPT, AgentBank and CC-Bench, each next turn "
    "waits for the reply, so its send time can shift by a few milliseconds)")


def q5(s: Row, facts: Row) -> str:
    iso = s["isolation"]
    rows = iso["rows"]
    alone = {
        g: {p: mean(rows[g][p]["alone"])
            for p in POLICIES}
        for g in GROUPS
    }
    ceil = {g: mean(rows[g]["ceiling"]) for g in GROUPS}
    vs = iso["vs_lru"]
    claim = ("Alone on the same cache, LRU serves " + _list([
        f"{L[g]} {pct(alone[g]['lru'], 0)} of {pct(ceil[g], 0)} possible"
        for g in GROUPS
    ]))
    cross = max(v["other_workload"] for v in s["q2"]["sources"].values())
    intro = (
        f"Each workload run by itself on the same "
        f"{s['config']['capacity_tokens']:,}-token cache, with {SAME_TIMES}. "
        "Only the other workloads are gone. The no-eviction mark is each "
        "workload's ceiling from the mixed run; that is safe because, on "
        f"average over the seeds, at most {pct(cross, 1)} of any workload's "
        "reuse came from another workload's blocks.")
    lines = []
    for g in GROUPS:
        verdicts = {p: vs[g][p]["verdict"] for p in ("lfu", "fifo")}
        by = [
            f"{pts(abs(mean(vs[g][p]['diff']))).split()[0]} ({P[p]})"
            for p in verdicts if verdicts[p] != "tie"
        ]
        lines.append(f"<b>{L[g]}</b>: LRU {pct(alone[g]['lru'], 1)}, LFU "
                     f"{pct(alone[g]['lfu'], 1)}, FIFO "
                     f"{pct(alone[g]['fifo'], 1)}; " + _against_lru(verdicts) +
                     (f", by {_list(by)} points" if by else "") + ".")
    full = [
        L[g] for g in GROUPS if all(
            all(
                abs(a - c) < 1e-6 for a, c in zip(rows[g][p]["alone"]["seeds"],
                                                  rows[g]["ceiling"]["seeds"]))
            for p in POLICIES)
    ]
    if full:
        lines.append(
            f"{_list(full)} alone never lost a reuse: every policy reaches "
            "its ceiling on every seed, so the policy never had to choose.")
    cuts = {g: _cutoff(iso["intervals"][g]["lru"]["curve"]) for g in GROUPS}
    endless = [L[g] for g in GROUPS if cuts[g] is None]
    stops = [g for g in GROUPS if cuts[g] is not None]
    mix_cut = facts.get("lru_cut")
    cut_bits = []
    if endless:
        cut_bits.append("keeps serving some of " +
                        _list([f"{x}'s" for x in endless]) +
                        " reuses at every drawn wait")
    if stops:
        cut_bits.append("serves almost nothing that waited longer than " +
                        _list([f"{secs(cuts[g])} for {L[g]}" for g in stops]))
    interval_claim = "Alone, LRU " + "; it ".join(cut_bits)
    same = [L[g] for g in GROUPS if _coincide(iso["intervals"][g])]
    fig_a = figure("fig-q5-portion", esc(claim), [badges(s)["engine"]],
                   _ul(lines))
    fig_b = figure(
        "fig-q5-interval", esc(interval_claim), [badges(s)["engine"]],
        _ul([
            "In the mix as a whole LRU served almost nothing that waited "
            "longer than " +
            (secs(mix_cut) if mix_cut is not None else "any wait") +
            " (point 1).",
            "Reading the panels: a line that stays near the top means the "
            "policy kept almost every block that workload came back for." +
            (f" For {_list(same)} the three lines lie on top of each other."
             if same else ""),
        ]))
    facts["q5_answer"] = (
        "Alone on the same cache, LRU serves " +
        _list([f"{L[g]} {pct(alone[g]['lru'], 0)}" for g in GROUPS]) +
        " of prompt tokens from the cache. Counting only differences every "
        "seed agrees on: " + "; ".join(f"{L[g]}, {_best_text(vs[g])}"
                                       for g in GROUPS) + ".")
    body = f'<p class="prose">{intro}</p>' + fig_a + fig_b
    return point(
        5, "Each workload alone", "Reuse portion and reuse interval "
        "of each workload run alone", body)


def q6(s: Row, facts: Row) -> str:
    iso = s["isolation"]
    rows = iso["rows"]
    cfg = s["config"]
    diff = {(g, p): mean(rows[g][p]["mixed"]) - mean(rows[g][p]["alone"])
            for g in GROUPS for p in POLICIES}
    verdict = {(g, p): rows[g][p]["verdict"] for g in GROUPS for p in POLICIES}
    hurt = [k for k in diff if verdict[k] == "higher"]
    helped = [k for k in diff if verdict[k] == "lower"]
    if hurt:
        worst = min(hurt, key=lambda k: diff[k])
        claim = (f"Sharing the cache costs {L[worst[0]]} the most: "
                 f"{pts(diff[worst])} under {P[worst[1]]}")
    elif helped:
        top = max(helped, key=lambda k: diff[k])
        claim = (f"Sharing the cache never costs a workload by the rule; it "
                 f"helps {L[top[0]]} most ({pts(diff[top])} under "
                 f"{P[top[1]]})")
    else:
        claim = ("Sharing the cache makes no difference that every seed "
                 "agrees on")
    minutes = (cfg["stop_s"] - cfg["w0_s"]) / 60
    rate = {g: mean(rows[g]["prompt_tokens_per_s"]) for g in GROUPS}
    total = sum(rate.values())
    intro = (
        f"Each workload twice on the same cache: once alone, once sharing "
        f"with the other three, with {SAME_TIMES}. So each workload's own "
        "traffic intensity, its requests and prompt tokens per second, is "
        "held fixed. What sharing adds is everyone else's traffic: in the "
        f"mix the cache takes about {round(total, -2):,.0f} prompt tokens per "
        "second, "
        "against " +
        _list([f"{rate[g]:,.0f} for {L[g]} alone" for g in GROUPS]) +
        ". The difference below is the effect of that competition; it does "
        "not separate how much traffic arrives from what kind.")
    lines = []
    for g in sorted(GROUPS, key=lambda g: diff[(g, "lru")]):
        kinds = {verdict[(g, p)] for p in POLICIES}
        if kinds == {"higher"}:
            lines.append(f"<b>{L[g]}</b> when sharing: lower by " + _list(
                [f"{pts(diff[(g, p)]).split()[0]} ({P[p]})"
                 for p in POLICIES]) + " points.")
            continue
        parts = []
        for p in POLICIES:
            v, d = verdict[(g, p)], diff[(g, p)]
            if v == "higher":
                parts.append(f"{P[p]} {pts(d)} lower")
            elif v == "lower":
                parts.append(f"{P[p]} {pts(d)} higher")
            else:
                parts.append(f"{P[p]} a tie")
        lines.append(f"<b>{L[g]}</b> when sharing: {_list(parts)}.")
    if hurt and len(hurt) == len(diff):
        lines.append("Every workload loses under every policy, on every "
                     f"seed ({_seeds_txt(s)}).")
    controls = [
        '<table><thead><tr><th>Workload</th><th>Requests measured, alone / '
        'sharing</th><th>Its prompt tokens per second</th><th>Prompt tokens '
        'per second the cache takes, alone / sharing</th><th>Median time to '
        'the full reply under LRU, alone / sharing</th></tr></thead><tbody>'
    ]
    lat = {}
    for g in GROUPS:
        r = rows[g]
        lat[g] = (mean(r["lru"]["latency_alone_s"]),
                  mean(r["lru"]["latency_mixed_s"]))
        controls.append(
            f"<tr><td>{L[g]}</td><td>{mean(r['requests']):,.0f} / "
            f"{mean(r['mixed_requests']):,.0f}</td><td>{rate[g]:,.0f}</td>"
            f"<td>{rate[g]:,.0f} / {total:,.0f}</td>"
            f"<td>{secs(lat[g][0])} / {secs(lat[g][1])}</td></tr>")
    controls.append("</tbody></table>")
    known = [g for g in GROUPS if None not in lat[g]]
    biggest = max(known,
                  key=lambda g: abs(lat[g][0] - lat[g][1]),
                  default=None)
    speed = ""
    if biggest is not None:
        a, b = lat[biggest]
        speed = (
            " The reply times are estimated by the GPU stand-in; they move "
            "the closed-loop turns slightly. The largest change in median "
            f"reply time under LRU is {L[biggest]}'s: {secs(a)} alone "
            f"against {secs(b)} sharing.")
    control_note = (
        '<details class="more"><summary>Traffic check</summary>'
        '<div class="stack"><p class="prose">Over the measured '
        f"{minutes:.0f} minutes (mean of seeds {_seeds_txt(s)}), each "
        "workload sends the same requests and prompt tokens per second "
        "alone and sharing; that holds by construction, and the counts "
        f"confirm it. The cache's total load differs.{speed}</p>"
        f'<div class="tablewrap">{"".join(controls)}</div></div></details>')
    lru_hurt = [g for g in GROUPS if verdict[(g, "lru")] == "higher"]
    lru_helped = [g for g in GROUPS if verdict[(g, "lru")] == "lower"]
    lru_tie = [g for g in GROUPS if verdict[(g, "lru")] == "tie"]
    answer = []
    if lru_hurt:
        answer.append("Under LRU, sharing the cache costs " + _list([
            f"{L[g]} {pts(diff[(g, 'lru')])}"
            for g in sorted(lru_hurt, key=lambda g: diff[(g, "lru")])
        ]) + " of usable reuse against running alone with its own traffic "
                      "unchanged")
    if lru_helped:
        answer.append(
            "it helps " +
            _list([f"{L[g]} ({pts(diff[(g, 'lru')])})" for g in lru_helped]))
    if lru_tie:
        answer.append(
            _list([L[g]
                   for g in lru_tie]) + " " + _verb(lru_tie, "ties", "tie"))
    tail = ""
    if hurt and len(hurt) == len(diff):
        tail = (" LFU and FIFO lose too, for every workload, on every "
                "seed.")
    facts["q6_answer"] = "; ".join(answer) + "." + tail
    body = (f'<p class="prose">{intro}</p>' +
            figure("fig-q6", esc(claim), [badges(s)["engine"]], _ul(lines)) +
            control_note)
    return point(6, "Alone vs. sharing", "Does sharing the cache hurt each "
                 "workload?", body)


def method(s: Row) -> str:
    cfg = s["config"]
    mix = s["mix"]
    seeds = [str(x) for x in cfg["reported_seeds"]]
    pacing = {
        "sharegpt":
        "closed loop: the next turn is sent after the reply plus "
        "a pause copied from the Bailian chat trace",
        "agentbank":
        "closed loop, pauses copied from CC-Bench's tool times",
        "ccbench":
        "closed loop, its own recorded tool times",
        "bailian":
        "open loop: sent at the recorded times, whatever the "
        "server does",
    }
    run_min = cfg["stop_s"] / 60
    rows = []
    for g in GROUPS:
        tok = [
            sum(mix[x]["workloads"][w]["prompt_token_share"]
                for w in _members(g)) for x in seeds
        ]
        rows.append(f"<tr><td>{L[g]}</td><td>"
                    f"{_mix_mean(s, g, 'requests'):,.0f}</td>"
                    f"<td>{pct(sum(tok) / len(tok), 0)}</td>"
                    f'<td class="text">{pacing[g]}</td></tr>')
    table = ('<div class="tablewrap"><table><thead><tr><th>Workload</th>'
             f'<th>Requests per {run_min:.0f}-minute run</th><th>Share of '
             f'prompt tokens over the whole {run_min:.0f} minutes</th>'
             '<th class="text">When requests are sent</th></tr></thead>'
             '<tbody>' + "".join(rows) + "</tbody></table></div>")
    kept = [
        mix[x]["workloads"][w]["kept_session_fraction"] for x in seeds
        for w in BAILIAN
        if mix[x]["workloads"][w].get("kept_session_fraction") is not None
    ]
    others = sum(_mix_mean(s, g, "requests")
                 for g in TEXT_WORKLOADS) / len(TEXT_WORKLOADS)
    engine = [
        e for e in s["capacity_window"]["engine"] if e["replay"] is not None
    ]
    agree = max((abs(e["engine"] - e["replay"]) for e in engine), default=None)
    commit = s.get("provenance", {}).get("commit", "?")[:8]
    train = _list([str(x) for x in cfg["train_seeds"]])
    items = [
        "Engine runs: vLLM v0 with prefix caching, run by <code>vllm "
        "serve</code>. The model does not run; a stand-in executor returns "
        "each request's recorded answer and estimates each step's time, so "
        "the scheduler and the cache behave as vLLM does while timing is an "
        "estimate. Model Qwen2.5-0.5B-Instruct; cache "
        f"{cfg['capacity_tokens']:,} tokens in blocks of {cfg['block_size']}.",
        f"Runs: {run_min:.0f} minutes each, sessions starting throughout. "
        f"Numbers count the requests sent from minute {cfg['w0_s'] / 60:.0f} "
        f"to {run_min:.0f}, after the cache has filled. Seed k draws its own "
        f"sessions and uses the k-th {run_min:.0f}-minute stretch of each "
        f"Bailian trace. Seed {train} is kept for training.",
        "Each of Bailian's four traces is thinned to the request rate of the "
        "other workloads" +
        (f" (each keeps {pct(min(kept), 0)} to {pct(max(kept), 0)} of its "
         "sessions)" if kept else "") + ", so Bailian as a whole sends about "
        f"{_mix_mean(s, 'bailian', 'requests') / others:.0f} times as many "
        "requests as each other workload. The traces store only fingerprints "
        "of the text, so we generate stand-in tokens that repeat exactly "
        "where the fingerprints repeat: the reuse pattern is real, the words "
        "are not.",
        "Alone runs (points 5 and 6) use the mix's plan with every other "
        "workload's requests removed; nothing else changes.",
        "Point 3 uses the requests as planned, not a run: its signals do not "
        "depend on the eviction policy.",
    ]
    if agree is not None:
        items.append(f"The cache replay agrees with the engine within "
                     f"{pts(agree)} on all {len(engine)} engine runs it was "
                     "checked against.")
    items.append(
        "Reproduce: <code>python -m analysis.run_m3_task2 grid --grid phase1"
        "</code> and <code>--grid isolated</code>; <code>python -m "
        "analysis.tables</code> on each run; <code>python -m "
        "analysis.identify</code>; <code>python -m analysis.run_m3_task2 sim "
        "... --window 600 1800</code>; <code>python -m analysis.m3t2_study"
        "</code>; <code>python -m analysis.m3t2_report</code>. Summary built "
        f"at commit {commit}.")
    return (f'<section class="section" aria-labelledby="method-h">'
            f'<h2 id="method-h">How the numbers were made</h2>{table}'
            f'{_ul(items)}</section>')


def sections(s: Row, data: Row) -> List[str]:
    facts: Row = {}
    parts = [
        q1(s, facts),
        q2(s, facts),
        q3(s, facts),
        q4(s, data, facts),
        q5(s, facts),
        q6(s, facts),
    ]
    return [
        header(s),
        answers(facts),
        terms(s),
        part("Part A · Points 1 to 4", "All four workloads sharing one cache"),
        *parts[:4],
        part("Part B · Points 5 and 6",
             "Each workload alone, and against sharing"),
        *parts[4:],
        method(s),
    ]
