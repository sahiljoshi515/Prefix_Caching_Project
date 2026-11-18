# Milestone 3, Task 1: eviction policies across workloads

Scope and definitions: [spec 006](../../specs/006-m3-workload-eviction/spec.md).

## Setup

```bash
python -m analysis.run_experiment --config examples/bailian_full.json \
  --out results/bailian-full-cap65536
python -m analysis.plot_experiment --out results/bailian-full-cap65536 \
  --figures report/milestone3/figures/task1/bailian-full
```

## Results

| Policy | Block hit rate | Overall usable token reuse | Mean request usable fraction |     Evictions | Evictions / 1,000 lookups |
| ------ | -------------: | -------------------------: | ---------------------------: | ------------: | ------------------------: |
| LRU    |         13.31% |                     13.27% |                       31.52% |     6,512,519 |                  1,053.49 |
| LFU    |     **15.04%** |                 **14.99%** |                   **35.68%** | **6,405,574** |              **1,036.19** |
| FIFO   |         12.01% |                     11.68% |                       28.17% |     6,593,086 |                  1,066.53 |

![Policy comparison](figures/task1/bailian-full/fig1_policy_comparison.png)

**Figure 1.** LFU achieves higher hit rate/usable reuse. Eviction rates can
exceed 1,000/1,000 lookups because generated response blocks also consume cache space.

Since overall usable token reuse is the primary ranking factor, LFU makes 14,879,904
input tokens reusable, versus 13,168,784 for LRU and 11,590,176 for FIFO. It has an
advantage of 1.72 percentage points over LRU and 3.32 percentage points over FIFO.

## Reuse profile

![Reuse distributions](figures/task1/bailian-full/fig2_reuse_profiles.png)

**Figure 2.** CDFs of per-request usable fraction (43,039 completed requests), hits
per distinct block (3,770,871 blocks), and reuse intervals (3,568,792 observations)
for each policy.

LFU has 958 zero reuse requests (2.23%), LRU has 3,727 (8.66%) and FIFO has 6,884 (15.99%).
While the mean request reuse is 35.68%, the percentage of token reuse is 14.99%. This is
due to short prompts skewing the percentage up due to them having fewer tokens and reusable
tokens taking up a higher percentage of these prompts. Therefore, this does not necessarily
translate to the same percentage of all prompt tokens being reusable.

Under all three policies LFU, LRU and FIFO, the percentage of block hits of one conversation
that was used in a different conversation are 99.61%, 99.66% and 99.67% respectively. This
differs from the ShareGPT multi-turn analysis, but due to Qwen-Bailian using anonymized hashes,
we do not know the exact content of these matches as they could be instruction prompts shared
between different conversations, explaining the high cross conversation block hit rate.

LFU gets 929,994 hits from 3,317 distinct reused blocks, while LRU gets 823,049 hits from 4,301
blocks. This suggests that LFU keeps the blocks that are repeatedly accessed rather than the most
variety.

About 99.9% of distinct observed blocks have zero hits. This is partly due to Qwen-Bailian only
containing response length rather than their anonymized hash tokens, so our adapter can only create
random and reproducible tokens to fill in the gap. This matters because in a real conversation the
next prompt could contain the response's tokens, reducing the percentage of zero hit blocks. This
percentage therefore describes our synthetic replay rather than the original production workload.

FIFO has 18,096 hits after a miss in 1,259 requests (2.93% of completed requests). These hits contain
289,536 tokens that are not part of a reusable prefix due to FIFO's evicting the oldest blocks and causing
prefix fragmentation. LRU and LFU have no such hits.

Reuse intervals are very similar across all three policies. The median is about 87 seconds and the median gap
is 540 intervening requests.

## Limitations

- Generating synthetic response blocks cannot reproduce all original response-to-next-prompt reuse within conversations
  or cross conversation since our randomly generated response blocks cannot be matched with the correct hash of the next
  input. This affects hit attribution and zero-hit profiles.

## AgentBank

1,000 trajectories, a seeded sample across all 19 subsets, replayed
multi-turn: each agent step is one request with the full history, and the
environment's observations are the user turns
([adapter rules](../../client_simulator/README.md#agentbank-and-cc-bench-agent-trajectories)).
AgentBank records neither session start times nor pauses, so it uses
Milestone 2's ShareGPT pacing: 1 conversation/s (Poisson) and a 5 s think
time. All other settings are the same as Bailian's.

```bash
python -m client_simulator.process_raw_traces --dataset agentbank \
  --raw data/raw/agentbank --seed 0 \
  --output data/processed/agentbank.responses.jsonl \
  --client-output data/processed/agentbank.conversations.jsonl
python -m analysis.run_experiment --config examples/agentbank_1000.json \
  --out results/agentbank-1000-cap65536
python -m analysis.plot_experiment --out results/agentbank-1000-cap65536 \
  --figures report/milestone3/figures/task1/agentbank-1000
```

### Results

| Policy | Block hit rate | Overall usable token reuse | Mean request usable fraction | Evictions | Evictions / 1,000 lookups |
| ------ | -------------: | -------------------------: | ---------------------------: | --------: | ------------------------: |
| LRU    |     **90.22%** |                 **89.47%** |                   **72.92%** | **36,002** |                **127.53** |
| LFU    |         18.64% |                     18.49% |                       21.88% |   238,086 |                    843.34 |
| FIFO   |         86.85% |                     55.12% |                       60.53% |    45,539 |                    161.31 |

![Policy comparison](figures/task1/agentbank-1000/fig1_policy_comparison.png)

**Figure 3.** LRU has the highest hit rate and usable reuse. FIFO's hit
rate is close to LRU's, but its usable reuse is much lower.

All 4,992 requests completed under every policy; none was too long for the
32,768-token context. LRU makes 4,075,376 of 4,554,839 input tokens
reusable, versus 2,510,832 for FIFO and 842,000 for LFU: an advantage of
34.35 percentage points over FIFO and 70.98 over LFU.

### Reuse profile

![Reuse distributions](figures/task1/agentbank-1000/fig2_reuse_profiles.png)

**Figure 4.** CDFs of per-request usable fraction (4,992 completed
requests), hits per distinct block (39,806 blocks), and reuse intervals
(about 255,000 observations) for each policy.

AgentBank fits in the cache. Sessions are short (median 3 replayed steps,
10 s) and prompts are small (median 460 tokens, p90 2,402), so about 20
sessions with about 18,000 tokens of history are active at once (at most
39,130), well under the 65,536-token cache. Reuse is almost all a
conversation's next step: the median reuse interval is 5.03 s, the think
time, with 25 requests in between, and only 3.0% of LRU's hits come from
another conversation. Those cross-conversation hits are shared task
instructions within a subset, which is why even a conversation's first
request reuses 24.5% of its prompt on average under LRU.

LRU keeps each active conversation's history until its next step. FIFO
evicts the oldest blocks, which are the start of each history, so later
blocks still hit but cannot be used: FIFO has 88,247 hits after a miss in
869 requests (17.41% of completed requests), 1,411,952 tokens that are not
part of a reusable prefix. LRU and LFU have no such hits.

LFU's collapse is cache pollution by stale blocks, the same collapse as in
[Milestone 2 Task 3](../milestone2/task3-cache-tuning.md). Plain LFU never
ages its counts, so blocks of conversations that have already finished
keep the few hits they collected. Replaying each run's event log, blocks of
finished conversations hold on average 89% of the cache under LFU after
the first quarter of the run, against 60% under LRU and 61% under FIFO. A
running conversation's newest blocks have no hits yet, so LFU evicts them
first: 85% of its evictions (202,155 of 238,086) removed a block of a
conversation that was still running, against 21% for FIFO and none for
LRU, which only ever evicted blocks of finished conversations.

These conclusions hold at this load: a 65,536-token cache, 1 conversation/s
and a 5 s think time, about 4,000 prompt tokens/s. In the
[Milestone 3 Task 2](https://github.com/sahiljoshi515/Prefix_Caching_Project/pull/16)
runs (numbers from PR #16's `task2-mixed-workload-policy.md`: a
131,072-token cache, recorded pauses, seeds 2–4), AgentBank alone never evicts, so LRU,
LFU and FIFO all reach 87.3%; in the mix with the other workloads LRU
(76.2%) and LFU (77.6%) tie and FIFO trails (72.1%).

## CC-Bench

All 370 Claude Code sessions (74 tasks, 5 models each), split into 495
conversations: each main thread, sub-agent and post-`/compact` context is
its own conversation. A resumed session logs its earlier history again and
continues; the copies are not replayed, and the new turns extend the
conversation they continue. Tool calls and results are text in the
history. Settings are the same as Bailian's.

**Pacing.** Each turn waits the pause recorded before it in the log, from
the end of the previous answer to the next message: mostly the tool run,
sometimes the person (median 0.125 s, mean 12 s, p99 232 s). Pauses are
capped at 10 minutes; 36 replayed pauses were longer (the longest
recorded pause is 15.8 hours, a session picked up again the next day). The sessions' start times are
when the benchmark ran each task over two months, not a load, so sessions
arrive at about 1 per minute (Poisson, 0.0166667/s). That is our estimate
of the CC-Bench rate in the
[Milestone 3 Task 2](task2-mixed-workload-policy.md) plans, whose sampler
picks about 31 CC-Bench sessions for each 30-minute plan (24 to 45 over 20
draws of its selection; their manifests are not in the repository).

```bash
python -m client_simulator.process_raw_traces --dataset ccbench \
  --raw data/raw/ccbench/train.parquet --seed 0 \
  --output data/processed/ccbench.responses.jsonl \
  --client-output data/processed/ccbench.conversations.jsonl
# About 9 hours per policy: README.md section 6 runs the three side by side.
python -m analysis.run_experiment --config examples/ccbench_full.json \
  --out results/ccbench-full-cap65536
python -m analysis.plot_experiment --out results/ccbench-full-cap65536 \
  --figures report/milestone3/figures/task1/ccbench-full
```

The reported run's frozen settings also carry `"think_time": 5`, the
fallback for a turn without a recorded pause; every replayed CC-Bench
turn has one, so it was never used, and `examples/ccbench_full.json` now
leaves it out.

### Results

| Policy | Block hit rate | Overall usable token reuse | Mean request usable fraction | Evictions | Evictions / 1,000 lookups |
| ------ | -------------: | -------------------------: | ---------------------------: | --------: | ------------------------: |
| LRU    |     **88.89%** |                 **88.84%** |                   **83.44%** | **1,237,281** |            **128.18** |
| LFU    |         80.77% |                     80.72% |                       75.41% | 2,021,607 |                    209.44 |
| FIFO   |         87.76% |                     85.25% |                       80.95% | 1,346,902 |                    139.54 |

![Policy comparison](figures/task1/ccbench-full/fig1_policy_comparison.png)

**Figure 5.** LRU has the highest hit rate and usable reuse, FIFO is
3.6 points behind and LFU 8.1 points behind; LFU evicts 63% more blocks
than LRU.

11,484 requests completed under every policy. 171 turns did not fit in the
32,768-token context, and since a skipped turn ends its conversation,
6,066 later turns were not sent: 35.2% of CC-Bench's 17,721 turns are not
replayed. There were no failures and no preemptions. LRU makes 137,281,168
of 154,524,769 input tokens reusable, versus 131,736,176 for FIFO and
124,731,904 for LFU.

### Reuse profile

![Reuse distributions](figures/task1/ccbench-full/fig2_reuse_profiles.png)

**Figure 6.** CDFs of per-request usable fraction (11,484 completed
requests), hits per distinct block (615,602 blocks), and reuse intervals
(about 9.2 million observations) for each policy.

Reuse is almost all a session's own history. Sessions are long (median 18
replayed turns, 95 s; p90 855 s) and prompts are large (median 12,619
tokens, p90 27,422), but at this rate only about 6 sessions are active at
once (at most 12), with a median of 77,786 tokens of history (at most
238,991), 1.2 times the cache. Most turns follow their tool result within
a second: the median reuse interval is 0.32 s with no other request in
between (p90 26 s and 9 requests). Under every policy 0.4% to 0.9% of hits
come from another conversation, and 5.2% (LRU, FIFO) to 6.2% (LFU) of
distinct blocks are never hit.

The working set is a little larger than the cache, so every policy evicts
some blocks of sessions that are still running, mostly sessions waiting
out a long pause. LRU evicts the fewest: 622,606 such blocks (50.3% of its
evictions), against 731,961 for FIFO (54.3%) and 1,407,395 for LFU
(69.6%). This is consistent with AgentBank's stale blocks at a smaller
scale: under LFU, blocks with a few old hits outlive a running session's
newest blocks, which have none yet. FIFO evicts the start of each history first,
so later blocks still hit but cannot be used: it has 236,941 hits after a
miss in 479 requests (4.2% of completed requests), 3,791,056 tokens that
are not part of a reusable prefix. LRU and LFU have no such hits.

The [Milestone 3 Task 2](task2-mixed-workload-policy.md) runs (a
131,072-token cache, seeds 2–4) find the same ranking for CC-Bench alone,
95.4% (LRU), 92.9% (FIFO) and 86.8% (LFU), a few points above ours with
twice the cache, and 85.2%, 78.3% and 82.4% in the mix.

### Limitations

- The arrival rate is our estimate of Task 2's, not measured: CC-Bench has
  no realistic session start times. The recorded pauses are measured, but
  they are what the original tools took on the benchmark's machines (and,
  in a few sessions, a person).
- Claude Code's system prompt and tool definitions (about 12,000 tokens per
  session) are not in the logs, so they are not replayed. They would be
  shared by every session and add cross-conversation reuse.
- 35.2% of CC-Bench's turns, the later turns of the longest sessions, are
  not replayed, so its real working set is larger than measured.
- Sub-agents start as independent conversations on the Poisson schedule,
  not at their time inside the parent session.
- AgentBank's source has a few identical trajectories under different ids
  (13 of the 4,992 replayed requests, 0.3%); they are replayed as the
  dataset has them.
- Responses are replayed from the recordings; AgentBank's agent outputs and
  CC-Bench's model outputs are not generated by Qwen.

## Best policy per workload

| Workload | Best policy | Overall usable token reuse (best / LRU / LFU / FIFO) | Why |
| -------- | ----------- | ---------------------------------------------------- | --- |
| Qwen Bailian | LFU | 14.99% / 13.27% / 14.99% / 11.68% | Recorded traffic shares prefixes across conversations, and LFU keeps the most-shared blocks |
| AgentBank | LRU | 89.47% / 89.47% / 18.49% / 55.12% | Short sessions fit in the cache; LRU keeps each history until its next step |
| CC-Bench | LRU | 88.84% / 88.84% / 80.72% / 85.25% | Most turns follow their tool result within a second; LRU keeps the running sessions' histories, LFU keeps old blocks instead and FIFO evicts the start of each history |

The policy that wins depends on where reuse comes from. When it is a
conversation's own next turn and the active histories fit (AgentBank),
recency is the right signal and LFU is the worst choice. CC-Bench's
reuse has the same source, so LRU wins there too, by less, because its
histories are a little larger than the cache. When it is prefixes shared
across conversations (Bailian), frequency wins. These rankings are for a
65,536-token cache. With Task 2's larger cache, AgentBank's policies tie
when it runs alone and LRU and LFU tie in the mix, while LRU leads for
CC-Bench.
