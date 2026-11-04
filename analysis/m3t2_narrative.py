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


def _came_back(data: Row, w: str, t: float) -> float:
    g = data["comeBack"]["grid"]
    return data["comeBack"]["series"][w][g.index(t)]


# Header and answers --------------------------------------------------------


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
            f'{states.count("hit")} of {n} blocks served from the cache">'
            f'{_strip(states)}</span></div>' for name, states in rows)
        kept = e["lru"].count("hit") if "lru" in e else 0
        strip = f"""
<div class="strip">
  <p class="prose">One real request from the runs: turn {e['turn'] + 1} of
  a ShareGPT conversation, {e['prompt_tokens']:,} prompt tokens in {n}
  blocks of 16 tokens. The person sent it {secs(e.get('since_previous_s'))}
  after the previous answer. A cache that never evicts would serve
  {e['ceiling'].count('hit')} of its {n} blocks; every policy we tested
  kept {kept}, the system prompt all conversations share.</p>
  {cells}
  <div class="legend-inline"><span><i class="cell hit"></i>served from the
  cache</span><span><i class="cell miss"></i>recomputed</span></div>
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
  simulator</span><span>16-token blocks</span><span>branch milestone3_task2
  · commit <span class="num">{esc(commit)}</span></span></div>
  {strip}
</header>"""


def answers(s: Row, data: Row) -> str:
    h_lo, h_hi = _range(s, HUMAN)
    a_lo, a_hi = _range(s, AGENTS)
    keep = data["retention"]["lru"]["p50"]
    lfu_keep = data["retention"]["lfu"]["p50"]
    ceil, lru = _u(s, "ceiling", "all"), _u(s, "lru", "all")
    lfu, fifo = _u(s, "lfu", "all"), _u(s, "fifo", "all")
    lfu_v = _vs(s, "lfu")
    lfu_word = ("ties with LRU" if _tie(lfu_v) else
                ("beats LRU" if lfu_v["mean"] > 0 else "trails LRU"))
    idn = (data.get("identify") or {}).get("numbers") or {}
    auc = (data.get("signals") or {}).get("auc", {})
    q3 = ""
    if idn:
        q3 = (f"The first two blocks of a prompt, a fingerprint of its "
              f"system prompt or task, name {pct(idn['fingerprint'])} of new "
              f"requests without a single mistake. A small decision tree on "
              f"prompt length, turn number and timing names the rest, for "
              f"{pct(idn['accuracy'])} overall. No labels are needed.")
    design = ""
    if auc:
        extra = auc["label_free+oracle"]["60"] - auc["label_free"]["60"]
        design = (f"How long a block has sat idle, the only thing LRU looks "
                  f"at, predicts its return with an AUC of "
                  f"{auc['idle_age']['60']:.2f} (0.5 is a coin flip). Adding "
                  f"its past hits gives {auc['+hits_before']['60']:.2f}, and "
                  f"all label-free signals {auc['label_free']['60']:.2f}. The "
                  f"true workload label adds only {extra:.2f}.")
    return f"""
<section class="answers" aria-label="Short answers">
  <div class="answer"><span class="q">Question 1 · reuse interval</span>
  <h3>Agents return in a blink; people return in minutes</h3>
  <p>Agent loops send their next request a median of
  <mark>{secs(a_lo)} to {secs(a_hi)}</mark> after the last answer. People
  come back to their own conversation after a median of
  <mark>{secs(h_lo)} to {secs(h_hi)}</mark>. At this cache size LRU keeps a
  freed block for about {secs(keep)} and LFU for about {secs(lfu_keep)}, so
  the agents are served and most human returns find their history
  gone.</p></div>
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
  <h3>A cheap prompt fingerprint names most requests</h3>
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
  <dt>AUC</dt><dd>How well a signal ranks the blocks that will come back
  above those that will not: 0.5 is a coin flip, 1.0 is perfect.</dd>
  <dt>Measured / simulator</dt><dd>Measured numbers come from real
  <code>vllm serve</code> runs of three seeds (2, 3, 4). Simulator numbers
  come from our fast replay of vLLM's own cache code and only guide the
  design; they are marked.</dd>
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


def headline(s: Row, data: Row) -> str:
    lfu_v, fifo_v = _vs(s, "lfu"), _vs(s, "fifo")
    tiles = "".join(
        f'<div class="tile"><span class="label">{label}</span><span '
        f'class="value">{pct(_u(s, p, "all"))}</span><span class="sub">'
        f'{sub}</span></div>' for p, label, sub in [
            ("ceiling", "No eviction", "what the traffic allows"),
            ("lru", "LRU", "vLLM's default"),
            ("lfu", "LFU", pts(lfu_v["mean"]) + " vs LRU" +
             (", a tie" if _tie(lfu_v) else "")),
            ("fifo", "FIFO", pts(fifo_v["mean"]) + " vs LRU"),
        ])

    def word(v: Row) -> str:
        return ("ties with"
                if _tie(v) else "beats" if v["mean"] > 0 else "trails")

    claim = (f"LFU {word(lfu_v)} LRU, FIFO {word(fifo_v)} it, and all three "
             "are far from the ceiling")
    lab, sg = _labels(s), "sharegpt"
    lead = ("Overall ties hide large swings per workload"
            if _tie(lfu_v) else "The totals hide large swings per workload")
    return f"""
<section class="section" aria-labelledby="headline"><h2 id="headline">How
much each policy serves</h2>
<p class="prose">Share of prompt tokens served from the cache, averaged over
the three reported seeds; the thin line through each mark spans the seeds.
The grey bar is what LRU leaves on the table compared with a cache that
never evicts.</p>
<div class="tiles">{tiles}</div>
{figure("fig-headline", claim, ["measured:measured, 3 seeds"],
        f"{lead}: LFU serves "
        f"{lab[sg]} {pct(_u(s, 'lfu', sg))} against LRU's "
        f"{pct(_u(s, 'lru', sg))}. Agents lose little under any policy; "
        "human-paced workloads lose most of their reuse. The new policy has "
        "to win back human returns without giving up the agents' reuse, and "
        "it must be judged per workload.")}
</section>"""


def q1(s: Row, data: Row) -> str:
    keep = data["retention"]["lru"]
    lab = _labels(s)
    W = data["workloads"]
    t = 300.0
    seen = [w for w in W if _came_back(data, w, t) is not None]
    low = sorted([w for w in seen if _came_back(data, w, t) < 0.5],
                 key=lambda w: _came_back(data, w, t))
    low_txt = ("Many blocks never come back: within 5 minutes only " +
               ", ".join(f"{lab[w]} {pct(_came_back(data, w, t))}"
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
        "Reuse distance counts the other blocks used in between. When it "
        "is below the cache's 8,192 blocks, LRU keeps the block; above it, "
        f"LRU cannot. {far_txt}Winning them back means keeping fewer, "
        "better-chosen blocks.")}
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
        "treatment.")}
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
        "Evicting a prompt's deepest blocks first, as vLLM's tie-break "
        "already does, suits the first kind.")}
{figure("fig-density", "Some workloads fill the idle cache without using "
        "it", ["measured:measured, LRU, 3 seeds"],
        f"Under LRU, {lab[heavy[0]]} and {lab[heavy[1]]} hold {pct(pool)} of "
        f"the idle cache's time but earn {pct(rev)} of its reuses. CC-Bench "
        f"earns {pct(d['ccbench']['revive'])} from "
        f"{pct(d['ccbench']['pool'])}. That idle time is what a better "
        "policy can hand to blocks that will be used.")}
{figure("fig-evictions", "Who evicts whom", ["measured:measured, 3 seeds"],
        f"{lab[W[driver[0]]]} and {lab[W[driver[1]]]} cause most evictions, "
        f"and {pct(cross)} of all evictions remove another workload's block. "
        "Protecting one workload always costs another, so every rule must "
        "be checked per workload, not only in total.",
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
    auc, base = sig.get("auc", {}), sig.get("base", {})
    base_txt = ""
    if base:
        b0, b30 = base["0"], base["30"]
        base_txt = (f"Of blocks just freed, {pct(b0['ccbench'])} of "
                    f"CC-Bench's and {pct(b0['bailian-thinking'])} of Bailian "
                    f"thinking's come back within a minute; after 30 idle "
                    f"seconds, {pct(b30['ccbench'])} and "
                    f"{pct(b30['bailian-thinking'])}. Scores are pooled over "
                    "blocks idle for 0, 5, 30 and 120 seconds.")
    sig_take = ""
    if auc:
        sig_take = (f"Past hits are the strongest single signal "
                    f"({auc['+hits_before']['60']:.2f} within a minute), "
                    f"stronger than the true workload label "
                    f"({auc['+oracle_workload']['60']:.2f}). A policy can "
                    "rank blocks with label-free signals and keep almost all "
                    "of what the label offers. LFU fails despite using the "
                    "best signal because it applies it to blocks that have "
                    "had no chance to be reused yet.")
    return f"""
<section class="section" aria-labelledby="q3"><h2 id="q3">Question 3: can a
server tell which workload a request belongs to?</h2>
<p class="prose">A server sees only tokens and timing. We gave a classifier
only what a server could know when a request arrives, trained it on seed 1
and tested it on seeds 2 to 4. It never reads the token values themselves,
only whether blocks were seen before, how long prompts are and how they
grow.</p>
{figure("fig-ladder", "A prompt fingerprint names most requests; a small "
        "tree names the rest", ["measured:plans of seeds 2–4, trained on "
        "seed 1"],
        f"The fingerprint resolves {pct(idn['fingerprint'])} of requests "
        f"with no errors. With the tree, accuracy is {pct(idn['accuracy'])} "
        f"({pct(idn['first'])} for a session's first request, "
        f"{pct(idn['later'])} for later ones). The inferred parent request "
        f"is the true previous turn for at least {pct(idn['parent'])} of "
        "later turns, so a policy can follow a session without any label.")}
{figure("fig-confusion", "Which workloads get confused", ["measured:plans "
        "of seeds 2–4"],
        f"The mistakes are mostly first requests with short prompts: "
        f"{confused}. A policy that falls back to a neutral default when "
        "unsure loses little.")}
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
B.</p>
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
    return f"""
<section class="section" aria-labelledby="headroom"><h2 id="headroom">How
much room is there?</h2>
<p class="prose">Belady's rule evicts the block needed again furthest in
the future. It needs to know the future, so no real server can run it, but
it is the best any policy could do at a given cache size. These numbers come
from our simulator, which replays vLLM's own cache code and matched the
engine runs closely (see Checks).</p>
{figure("fig-capacity", "Policy matters most where memory is scarce",
        ["sim:simulator"],
        f"At the fixed cache, Belady serves {pct(b['belady_all'])} against "
        f"LRU's {pct(b['lru_all'])}: {pts(gain)} are there for a better "
        "policy to win. Every policy converges once the cache holds the "
        "working set.")}
{figure("fig-belady", "Where the headroom is", ["sim:simulator"],
        "The room to gain is in the human-paced workloads. Belady maximises "
        "the total, so its split across workloads is one good allocation, "
        "not a limit for each workload.")}
</section>"""


def signals(s: Row, data: Row) -> str:
    lab = _labels(s)
    h_lo, h_hi = _range(s, HUMAN)
    a_lo, a_hi = _range(s, AGENTS)
    keep = data["retention"]["lru"]["p50"]
    d = data["density"]["lru"]
    fr = s["policy_views"]["lfu"]["frequency"]["all"]
    auc = (data.get("signals") or {}).get("auc", {})
    idn = (data.get("identify") or {}).get("numbers") or {}
    src = s["q2"]["sources"]
    label_row = ""
    if idn and auc:
        extra = auc["label_free+oracle"]["60"] - auc["label_free"]["60"]
        label_row = (f"A fingerprint plus a small tree names "
                     f"{pct(idn['accuracy'])} of requests without labels; the "
                     f"true label adds {extra:.2f} AUC over label-free "
                     "signals.")
    hb = auc.get("+hits_before", {}).get("60")
    rows = [
        ("How soon a session returns",
         f"Agents return in {secs(a_lo)} to {secs(a_hi)} (median); people in "
         f"{secs(h_lo)} to {secs(h_hi)}. LRU keeps idle blocks about "
         f"{secs(keep)}.",
         "Give a session's history a retention time that fits its loop: "
         "short for agent loops, minutes for people, ended when the session "
         "stops.",
         "No: the time since the session's previous turn is visible to the "
         "server.",
         "Holding human history uses space the agents could have used."),
        ("Past hits", f"Blocks freed with 0, 1 and 2+ past hits came back "
         f"{pct(fr['0']['revived_share'])}, {pct(fr['1']['revived_share'])} "
         f"and {pct(fr['2+']['revived_share'])} of the time under LFU" +
         (f"; AUC {hb:.2f}." if hb else "."),
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
         "A rare long return, such as a user back after an hour, is lost."),
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
         "cached (spec 007 FR-005).", "No.",
         "Restricts which blocks a smarter score can pick."),
        ("Who the request is", label_row,
         "Run label-free, with oracle labels only as the upper bound to "
         "compare against (spec 007 label modes).", "No, by design.",
         "A session's first request is the least certain."),
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
            "Simulator against engine: on the same plans, the simulator's "
            "usable reuse is within "
            f"{100 * val['max_diff']:.1f} points of the engine runs overall.")
    sens = s.get("sensitivity") or {}
    noise = sens.get("noise")
    if noise:
        items.append(
            "Run-to-run noise: replaying seed 2 under LRU a second time moved "
            f"usable reuse by {pts(noise['all']['diff'])} overall.")
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
            "Pacing: with Phase 0's fixed 5 s pause for ShareGPT instead of "
            "real human gaps, LRU serves ShareGPT "
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
