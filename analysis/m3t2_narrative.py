"""The results page's words (``analysis/m3t2_report.py``).

Every number below is read from ``summary.json``; the sentences around
them were written for the Phase 1 results and are checked against them by
the review agents before the page is published (spec 007 FR-011).
"""
from typing import Any, Dict, List, Optional

from analysis.m3t2_report import esc, figure, pct, pts, secs

Row = Dict[str, Any]
HUMAN = ("sharegpt", "bailian-chat", "bailian-thinking", "bailian-coder")
AGENTS = ("agentbank", "ccbench")


def _u(s: Row, policy: str, w: str) -> Optional[float]:
    v = s["headline"][policy][w]["usable"]
    return None if v is None else v["mean"]


def _strip(states: List[str]) -> str:
    return "".join(f'<i class="cell {x}"></i>' for x in states)


def header(s: Row, data: Row) -> str:
    cfg = s["config"]
    e = s.get("example") or {}
    commit = (s.get("provenance") or {}).get("commit", "")[:8]
    runs = (len(cfg["policies"]) + 1) * len(cfg["reported_seeds"])
    seeds = ", ".join(str(x) for x in cfg["reported_seeds"])
    strip = ""
    if e:
        n = len(e["ceiling"])
        rows = [("No eviction", e["ceiling"])
                ] + [(p.upper(), e[p]) for p in cfg["policies"] if p in e]
        cells = "".join(
            f'<div class="strip-row"><span class="strip-label">{esc(name)}'
            f'</span><span class="cells" role="img" aria-label="{esc(name)}: '
            f'{states.count("hit")} of {n} blocks from the cache">'
            f'{_strip(states)}</span></div>' for name, states in rows)
        strip = f"""
<div class="strip">
  <p class="prose">One real request from the run (seed {e['seed']}): turn
  {e['turn'] + 1} of a ShareGPT conversation, {e['prompt_tokens']:,} prompt
  tokens in {n} blocks of 16. The person sent it
  <span class="num">{secs(e.get('since_previous_s'))}</span> after the
  previous turn. Each square is one block; a filled square was read from the
  cache instead of being recomputed.</p>
  {cells}
  <div class="legend-inline"><span><i class="cell hit"></i> served from the
  cache</span><span><i class="cell miss"></i> recomputed</span><span><i
  class="cell cut"></i> cached, but behind a recomputed block, so
  unusable</span></div>
</div>"""
    return f"""
<header class="top">
  <span class="eyebrow">COMP 536 · Milestone 3 · Task 2 · Phase 1</span>
  <h1>Mixed-Workload Reuse Study</h1>
  <p class="prose">Seven kinds of LLM traffic share one prefix cache of
  131,072 tokens. We replayed them through vLLM with three eviction
  policies and measured, workload by workload, how soon cached blocks are
  needed again, how much of each prompt the cache can serve, and whether a
  server can tell the workloads apart. Each finding ends with what it means
  for the eviction policy we design next.</p>
  <div class="meta"><span>{runs} reported engine runs (seeds
  {seeds})</span><span>Qwen2.5-0.5B-Instruct on vLLM's CPU
  simulator</span><span>16-token blocks</span><span>branch
  milestone3_task2 · commit <span class="num">{esc(commit)}</span></span></div>
  {strip}
</header>"""


def answers(s: Row, data: Row) -> str:
    q = s["q1"]["intervals_all"]
    human = [q[w]["quantiles"]["p50"] for w in HUMAN]
    agents = [q[w]["quantiles"]["p50"] for w in AGENTS]
    keep = data["retention"]["lru"]["p50"]
    lfu_keep = data["retention"]["lfu"]["p50"]
    ceil, lru = _u(s, "ceiling", "all"), _u(s, "lru", "all")
    lfu, fifo = _u(s, "lfu", "all"), _u(s, "fifo", "all")
    ident = (s.get("identify") or {}).get("page", {}).get("answer", "")
    sig = (s.get("signals") or {}).get("page", {}).get("answer", "")
    return f"""
<section class="answers" aria-label="Short answers">
  <div class="answer"><span class="q">Question 1 · reuse interval</span>
  <h3>Agents return in a blink; people return in minutes</h3>
  <p>Agent loops send their next request a median
  <mark>{secs(min(agents))}–{secs(max(agents))}</mark> after the last answer,
  so their blocks are read again almost at once. People (ShareGPT and the
  three Bailian chat traces) return after a median of
  <mark>{secs(min(human))}–{secs(max(human))}</mark>. At this cache size LRU
  keeps a freed block for about {secs(keep)} and LFU for about
  {secs(lfu_keep)}, so the agents are served and most human returns are
  lost.</p></div>
  <div class="answer"><span class="q">Question 2 · reuse portion</span>
  <h3>The cache could serve {pct(ceil)} of prompt tokens; LRU serves
  {pct(lru)}</h3>
  <p>A cache that never evicts would serve {pct(ceil)} of all prompt tokens.
  LRU serves {pct(lru)}, LFU {pct(lfu)} and FIFO {pct(fifo)}. Agents keep
  most of their reuse ({pct(_u(s, "lru", "ccbench"))} of a possible
  {pct(_u(s, "ceiling", "ccbench"))} for CC-Bench under LRU); ShareGPT
  keeps {pct(_u(s, "lru", "sharegpt"))} of a possible
  {pct(_u(s, "ceiling", "sharegpt"))}.</p></div>
  <div class="answer">
  <span class="q">Question 3 · telling workloads apart</span>
  <h3>A cheap prompt fingerprint names most requests</h3>
  <p>{ident}</p></div>
  <div class="answer"><span class="q">For the design</span>
  <h3>Keep blocks by their chance of coming back, not by age alone</h3>
  <p>{sig}</p></div>
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
  The <b>idle age</b> is the part of it the block spends waiting unused,
  which is what an eviction policy sees.</dd>
  <dt>Session</dt><dd>One conversation, one agent task or one production
  user thread: the requests that build on each other.</dd>
  <dt>Measured / simulator</dt><dd>Measured numbers come from real
  <code>vllm serve</code> runs of three seeds (2, 3, 4). Simulator numbers
  come from our fast replay of vLLM's own cache code and only guide the
  design; they are marked.</dd>
</dl></section>"""


def mix(s: Row, data: Row) -> str:
    m = data["mix"]
    return f"""
<section class="section" aria-labelledby="mix"><h2 id="mix">What we
replayed</h2>
<p class="prose">Each run replays 30 minutes of traffic at about 3
requests per second, one seventh of the requests from each workload, and
measures the last 20 minutes (the first 10 warm the cache). Requests are
equal, tokens are not: CC-Bench agent prompts are long, so it brings
{pct(m["ccbench"]["tokens"])} of all prompt tokens.</p>
{figure("fig-mix", "Who brings the work", ["input:inputs to the study"],
        "Equal request shares hide very unequal cache demand. A policy that "
        "treats every request alike will let the long agent prompts set the "
        "pace of the whole cache.",
        note="Pauses between turns: ShareGPT borrows real gaps from the "
        "Bailian chat trace and AgentBank borrows CC-Bench's recorded tool "
        "times, because their datasets have no timestamps. Their timing is "
        "an input to the study; only the Bailian traces' and CC-Bench's "
        "timing is measured from real use. The Bailian traces are sampled "
        "down to fit the mix, which thins their cross-user reuse.")}
</section>"""


def headline(s: Row, data: Row) -> str:
    tiles = "".join(
        f'<div class="tile"><span class="label">{label}</span><span '
        f'class="value">{pct(_u(s, p, "all"))}</span><span class="sub">'
        f'{sub}</span></div>' for p, label, sub in [
            ("ceiling", "No eviction", "what the traffic allows"),
            ("lru", "LRU", "vLLM's default"),
            ("lfu", "LFU",
             pts(s["headline"]["lfu"]["all"]["vs_lru"]["mean"]) + " vs LRU"),
            ("fifo", "FIFO",
             pts(s["headline"]["fifo"]["all"]["vs_lru"]["mean"]) + " vs LRU"),
        ])
    return f"""
<section class="section" aria-labelledby="headline"><h2 id="headline">How
much each policy serves</h2>
<p class="prose">Share of prompt tokens served from the cache, averaged over
the three reported seeds. The grey bar on each row is what LRU leaves on the
table compared with a cache that never evicts.</p>
<div class="tiles">{tiles}</div>
{figure("fig-headline", "LRU is the best of the three, and still far from "
        "the ceiling", ["measured:measured, 3 seeds"],
        "The gap is not spread evenly. Agents lose little; the human-paced "
        "workloads lose most of their reuse. The new policy has to win "
        "back human returns without giving up the agents' reuse.")}
</section>"""


def q1(s: Row, data: Row) -> str:
    keep = data["retention"]["lru"]
    return f"""
<section class="section" aria-labelledby="q1"><h2 id="q1">Question 1: how
soon do blocks come back?</h2>
{figure("fig-comeback", "Workloads differ in return time by about a "
        "thousand times", ["measured:measured, no-eviction runs"],
        f"LRU keeps a freed block for {secs(keep['p10'])} to "
        f"{secs(keep['p90'])} (middle 80%) at this cache size. Blocks of "
        "agent loops come back well inside that window; most human returns "
        "come after it. A policy could keep human-paced blocks longer only "
        "by evicting something else, so it needs to know which blocks will "
        "come back at all: some workloads leave most blocks unused forever.",
        note="Each line is the share of freed blocks that were needed again "
        "within the time on the axis, in runs whose cache never fills. A "
        "line that stays low never comes back: those blocks are dead weight "
        "in any cache. The labels mark how long each policy kept an idle "
        "block before evicting it (median).")}
{figure("fig-hitinterval", "LRU serves every reuse under about 10 seconds "
        "and none after a minute", ["measured:measured, 3 seeds"],
        "LRU is blind to the workload: every line falls at the same place. "
        "LFU drops young blocks first, so it loses even short reuses of "
        "fresh conversation history. FIFO behaves like LRU on time but "
        "evicts long-lived shared prompts.",
        note="For each reuse in the policy's own run, whether the block was "
        "still cached, by how long it had been since its previous use.")}
{figure("fig-distance", "Most human-paced reuses are out of reach for any "
        "LRU cache of this size", ["measured:measured, no-eviction runs"],
        "Reuse distance is how many other blocks were touched in between. "
        "Below the cache's 8,192 blocks, LRU keeps the block; above, no "
        "amount of LRU tuning helps. Winning these back means keeping "
        "fewer, better-chosen blocks: being selective, not just bigger.")}
</section>"""


def q2(s: Row, data: Row) -> str:
    return f"""
<section class="section" aria-labelledby="q2"><h2 id="q2">Question 2: how
much of each prompt is reused?</h2>
{figure("fig-sources", "Reuse comes from a conversation's own history, "
        "except in production API traffic", ["measured:measured, "
        "no-eviction runs"],
        "Conversation history dominates every workload except the Bailian "
        "API, whose reuse is shared system prompts across users, and the "
        "thinking and coder traces, which mix both. Shared prompts are few "
        "and hot; history is large and private to one session. They want "
        "different treatment.")}
{figure("fig-lost", "Where the lost reuse goes", ["measured:measured, "
        "3 seeds"],
        "Almost all loss is eviction. FIFO also cuts off blocks that are "
        "still cached but sit behind an evicted one, which vLLM cannot use: "
        "a policy must never evict a block while its continuation stays "
        "cached.",
        note="Blocks that the no-eviction run reused, split into reused, "
        "evicted, and cached-but-unusable under each policy.")}
{figure("fig-position", "Early blocks are reused most",
        ["measured:measured, no-eviction runs"],
        "The first blocks (system prompts, task instructions) are shared "
        "widely. Deeper blocks belong to one session. Evicting from the "
        "deep end first keeps more reuse per block.")}
{figure("fig-density", "Some workloads fill the idle cache without using "
        "it", ["measured:measured, LRU, 3 seeds"],
        "Under LRU, the human-paced traces hold most of the idle cache but "
        "earn few of its reuses, because their blocks wait out LRU's window "
        "and are then evicted unused. That cache time is what a better "
        "policy can redistribute.")}
{figure("fig-evictions", "Who evicts whom", ["measured:measured, 3 seeds"],
        "The long agent prompts drive most evictions. A policy can protect "
        "a block only by evicting another, so any class rule shifts this "
        "matrix; it must be checked per workload, not only in total.")}
</section>"""


def q3(s: Row, data: Row) -> str:
    ident = (s.get("identify") or {}).get("page", {})
    sig = (s.get("signals") or {}).get("page", {})
    return f"""
<section class="section" aria-labelledby="q3"><h2 id="q3">Question 3: can a
server tell which workload a request belongs to?</h2>
<p class="prose">{ident.get("intro", "")}</p>
{figure("fig-ladder", "A prompt fingerprint names most requests; the rest "
        "need a fallback", ["measured:plans of seeds 2–4, trained on "
        "seed 1"], ident.get("takeaway", ""))}
{figure("fig-confusion", "Which workloads get confused", ["measured:plans "
        "of seeds 2–4"], ident.get("confusion_takeaway", ""))}
{figure("fig-signals", "What predicts that an idle block will come back",
        ["measured:no-eviction runs, trained on seed 1"],
        sig.get("takeaway", ""), note=sig.get("note", ""))}
</section>"""


def paper(s: Row, data: Row) -> str:
    card = (s.get("paper") or {}).get("page", {})
    rows = card.get("rows", [])
    body = "".join(
        f"<tr><td>{esc(r['claim'])}</td><td>{esc(r['paper'])}</td>"
        f"<td>{esc(r['raw'])}</td><td>{esc(r['mix'])}</td>"
        f'<td><span class="chip {r["chip"]}">{esc(r["verdict"])}</span>'
        f"</td></tr>" for r in rows)
    return f"""
<section class="section" aria-labelledby="paper"><h2 id="paper">Do the
paper's findings hold here?</h2>
<p class="prose">{card.get("intro", "")}</p>
<div class="tablewrap"><table class="signals"><thead><tr><th>Claim</th>
<th>Paper</th><th>Raw Bailian traces (full rate)</th><th>Our mix</th>
<th>Verdict</th></tr></thead><tbody>{body}</tbody></table></div>
<p class="takeaway"><b>For the design</b>{card.get("takeaway", "")}</p>
</section>"""


def headroom(s: Row, data: Row) -> str:
    b = data.get("beladySplit") or {}
    return f"""
<section class="section" aria-labelledby="headroom"><h2 id="headroom">How
much room is there?</h2>
<p class="prose">Belady's rule evicts the block needed again furthest in the
future. It needs to know the future, so no real server can run it, but it
is the best any policy could do at a given cache size. These numbers come
from our simulator, which replays vLLM's own cache code and matched the
engine runs closely (see Checks).</p>
{figure("fig-capacity", "A smarter policy is worth about as much as a "
        "bigger cache", ["sim:simulator"],
        f"At the fixed cache, Belady serves {pct(b.get('belady_all'))} "
        f"against LRU's {pct(b.get('lru_all'))}. Every policy converges "
        "once the cache holds the working set, so the policy matters most "
        "exactly where memory is scarce.")}
{figure("fig-belady", "Where the headroom is", ["sim:simulator"],
        "The room to gain is in the human-paced workloads. Belady maximises "
        "the total, so its split per workload is one good allocation, not "
        "a per-workload limit.")}
</section>"""


def signals(s: Row, data: Row) -> str:
    rows = (s.get("design") or {}).get("signals", [])
    body = "".join(
        f"<tr><td>{esc(r['signal'])}</td><td>{esc(r['evidence'])}</td>"
        f"<td>{esc(r['rule'])}</td><td>{esc(r['label'])}</td>"
        f"<td>{esc(r['risk'])}</td></tr>" for r in rows)
    return f"""
<section class="section" aria-labelledby="signals"><h2 id="signals">Signals
for the policy design</h2>
<p class="prose">What the measurements above suggest a workload-aware policy
should use, the evidence for each, and what could go wrong.</p>
<div class="tablewrap"><table class="signals"><thead><tr><th>Signal</th>
<th>Evidence</th><th>Rule it suggests</th><th>Needs the workload
label?</th><th>Risk</th></tr></thead><tbody>{body}</tbody></table></div>
</section>"""


def checks(s: Row, data: Row) -> str:
    items = (s.get("design") or {}).get("checks", [])
    lis = "".join(f"<li>{x}</li>" for x in items)
    return f"""
<section class="section" aria-labelledby="checks"><h2 id="checks">Checks
and limits</h2>
<ul class="prose stack">{lis}</ul>
</section>"""


def method(s: Row, data: Row) -> str:
    items = (s.get("design") or {}).get("method", [])
    lis = "".join(f"<li>{x}</li>" for x in items)
    return f"""
<section class="section" aria-labelledby="method"><details><summary><h2
id="method" style="display:inline">How it was measured</h2></summary>
<ul class="prose stack">{lis}</ul></details></section>"""


def sections(s: Row, data: Row) -> Dict[str, str]:
    return {
        "header": header(s, data),
        "answers": answers(s, data),
        "glossary": glossary(s, data),
        "mix": mix(s, data),
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
