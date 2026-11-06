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
