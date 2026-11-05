"""The results page's words with the new policies, TTDF, RACE-RP and
Sampled LRFU (``analysis/m3t2_narrative.py``): they join points 1, 2 and
4, get a point 7, and leave the page as it was when the summary does not
hold them."""
import re

import pytest

from analysis import m3t2_narrative as nv
from analysis import m3t2_report as rp
from tests.analysis.test_m3t2_report_policies import (ALL_NEW, CAPS, FIXED,
                                                      NEW, coarse, curve,
                                                      summary, tuning)

pytestmark = pytest.mark.m3


def text(html):
    """The words of an HTML fragment, tags dropped, spaces collapsed."""
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()


def page(s):
    return nv.sections(s, rp.build_data(s))


def baseline():
    return summary(engine=(), replay=(), tune=False, sketch=False)


def table_rows(html):
    """Each table body row's cells, as text (header rows dropped)."""
    rows = [[text(c) for c in re.findall(r"<td[^>]*>(.*?)</td>", r)]
            for r in re.findall(r"<tr>(.*?)</tr>", html)]
    return [r for r in rows if r]


def run(caps_verdicts):
    return nv._vs_lru_runs(CAPS, caps_verdicts)


# --- The page's frame: header, answers, terms, eyebrows, method -------------


def test_header_counts_seven_questions_and_names_the_new_policies():
    t = text(nv.header(summary()))
    assert "answers seven questions" in t
    assert "two new policies, TTDF and RACE-RP, which read no workload " \
        "labels" in t
    t = text(nv.header(baseline()))
    assert "answers six questions" in t
    assert "TTDF" not in t and "RACE-RP" not in t


def test_answers_add_a_seventh_card_only_with_its_answer():
    facts = {f"q{n}_answer": f"answer {n}" for n in range(1, 7)}
    six = nv.answers(facts)
    assert "The six answers" in six and six.count('class="answer"') == 6
    assert "#point-7" not in six
    facts["q7_answer"] = ""  # an empty answer adds no card
    assert nv.answers(facts) == six
    facts["q7_answer"] = "answer 7"
    seven = nv.answers(facts)
    assert "The seven answers" in seven
    assert seven.count('class="answer"') == 7
    # Without the names q7 sets, the title names both policies.
    assert ('<span class="q">7 · TTDF and RACE-RP</span><p>answer 7</p>'
            '<a href="#point-7">See point 7</a>') in seven
    facts["q7_names"] = "RACE-RP"
    assert '<span class="q">7 · RACE-RP</span>' in nv.answers(facts)


def test_terms_define_the_new_policies_before_belady():
    html = nv.terms(summary())
    terms = re.findall(r"<dt>(.*?)</dt>", html)
    i = terms.index("LRU, LFU, FIFO")
    assert terms[i + 1:i + 4] == ["TTDF", "RACE-RP", "Belady"]
    t = text(html)
    assert "Two-timescale decayed frequency" in t
    assert "RACE-based reuse probability" in t
    assert "TTDF" not in re.findall(r"<dt>(.*?)</dt>", nv.terms(baseline()))
    assert "RACE-RP" not in nv.terms(baseline())


def test_point_eyebrow_has_no_total():
    html = nv.point(7, "Mixed workload", "Q?", "<p>b</p>")
    assert '<span class="eyebrow">Point 7 · Mixed workload</span>' in html
    assert 'id="point-7"' in html and "of 6" not in html


def test_method_describes_the_new_policies_before_reproduce():
    items = re.findall(r"<li>(.*?)</li>", nv.method(summary()))
    first = next(i for i, x in enumerate(items) if "reuse_evictors.py" in x)
    assert items[-1].startswith("Reproduce")
    assert first == len(items) - 6  # five new items, then Reproduce
    base = nv.method(baseline())
    assert "reuse_evictors.py" not in base and "RACE-RP" not in base


def test_new_method_items_follow_the_summary():
    items = [text(x) for x in nv._new_method(summary())]
    assert len(items) == 5
    assert items[0].startswith("TTDF and RACE-RP ( vllm/core/reuse_evictors"
                               ".py ) are vLLM evictors beside LRU, LFU and "
                               "FIFO")
    assert "sampled_lrfu.py" not in items[0]
    assert "VLLM_EVICTION_POLICY" in items[0]
    assert "The cache tells them each time" in items[0]
    assert items[1].startswith("TTDF and RACE-RP stamp every use")
    assert "and evict only idle blocks" in items[1]
    assert "3,639,507 observations" in items[2]
    assert "(39% were)" in items[2]
    assert items[3].startswith("Settings were tuned on seed 1 only")
    assert "TTDF's short and long time constants" in items[3]
    assert "RACE-RP's tables (16 or 64) and bits (1 to 8)" in items[3]
    assert "Sampled LRFU" not in items[3]  # not tuned here
    assert "Seeds 2, 3 and 4 were run once" in items[3]
    # No engine runs at other sizes: no check runs named.
    assert items[4] == (
        "Engine runs of TTDF and RACE-RP: 128K on seeds 2, 3 and 4 (points "
        "1, 2 and 7). The numbers behind every chart are also in CSV files "
        "under report/milestone3/data/task2 .")
    # No sketch, no tuning, no engine runs: the two policy items only.
    s = summary(sketch=False, tune=False)
    s["new_policies"]["engine"] = []
    items = [text(x) for x in nv._new_method(s)]
    assert len(items) == 2 and "VLLM_EVICTION_POLICY" in items[0]
    s = summary(sketch=False)
    items = [text(x) for x in nv._new_method(s)]
    assert len(items) == 4
    assert not any("sketch was learned" in x for x in items)


def engine_run(policy, seed, capacity):
    return {
        "seed": seed,
        "policy": policy,
        "capacity": capacity,
        "engine": 0.5,
        "replay": 0.5
    }


def engine_item(s):
    return text(nv._new_method(s)[-1])


def test_new_method_engine_item_names_the_policies_and_check_runs():
    s = summary(engine=("race_rp", ))
    assert engine_item(s).startswith("Engine runs of RACE-RP: 128K on seeds "
                                     "2, 3 and 4 (points 1, 2 and 7).")
    # One seed at two other sizes.
    s = summary()
    s["capacity_window"]["engine"] += [
        engine_run("ttdf", 2, 393216),
        engine_run("ttdf", 2, 32768),
        engine_run("race_rp", 2, 32768),  # the same seed and size
    ]
    assert engine_item(s).startswith(
        "Engine runs of TTDF and RACE-RP: 128K on seeds 2, 3 and 4 (points "
        "1, 2 and 7), and seed 2 at 32K and 384K to check the replay. The "
        "numbers behind")
    # Grouped by seed; the baselines' runs at other sizes do not count.
    s["capacity_window"]["engine"] += [
        engine_run("race_rp", 3, 65536),
        engine_run("lru", 4, 262144),
    ]
    assert ("(points 1, 2 and 7), and seed 2 at 32K and 384K; seed 3 at 64K "
            "to check the replay.") in engine_item(s)
    # New policies without engine runs: their runs elsewhere do not count.
    s["new_policies"]["engine"] = ["race_rp"]
    assert "and seed 2 at 32K; seed 3 at 64K to check" in engine_item(s)
    # No replay window at all: no check runs.
    del s["capacity_window"]
    assert "to check the replay" not in engine_item(s)


# --- Point 1 ----------------------------------------------------------------


def test_q1_adds_table_rows_and_a_clause_for_the_new_policies():
    facts = {}
    html = nv.q1(summary(), facts)
    rows = table_rows(html)
    assert [r[0] for r in rows] == ["LRU", "LFU", "FIFO", "TTDF", "RACE-RP"]
    assert rows[3] == ["TTDF", "97.7%", "50.0%", "10.0%", "0.0%"]
    assert rows[4] == ["RACE-RP", "95.0%", "40.0%", "30.0%", "20.0%"]
    t = text(html)
    assert "between FIFO, LRU and LFU (and TTDF and RACE-RP)?" in t
    assert "TTDF ranks a block by its recent uses" in t
    new = ("Of the two new policies, TTDF serves 97.7% of the quick reuses "
           "and 5% of those that waited over 30 s and RACE-RP serves 95.0% "
           "of the quick reuses and 25% of those that waited over 30 s.")
    assert new in t
    assert facts["q1_answer"].endswith(new)
    assert facts["lru_cut"] == 30


def test_q1_new_clause_without_an_lru_cutoff():
    # LRU serves some reuse at every wait: no "waited over" part.
    s = summary()
    bins = curve([1.0, 0.99, 0.95, 0.30, 0.05, 0.02])
    s["mixed_intervals"]["lru"] = {"curve": bins, "by_bin": coarse(bins)}
    facts = {}
    nv.q1(s, facts)
    assert facts["lru_cut"] is None
    assert facts["q1_answer"].endswith(
        "Of the two new policies, TTDF serves 97.7% of the quick reuses and "
        "RACE-RP serves 95.0% of the quick reuses.")


def test_q1_without_new_policies_keeps_the_three_rows():
    facts = {}
    html = nv.q1(baseline(), facts)
    assert [r[0] for r in table_rows(html)] == ["LRU", "LFU", "FIFO"]
    t = text(html)
    assert "between FIFO, LRU and LFU?" in t
    assert "TTDF" not in t and "new policies" not in facts["q1_answer"]
    assert "even one used a moment ago.</p>" in html  # no trailing space


def test_q1_one_new_policy():
    facts = {}
    html = nv.q1(summary(engine=("race_rp", )), facts)
    assert [r[0] for r in table_rows(html)][-1] == "RACE-RP"
    assert "TTDF serves" not in facts["q1_answer"]
    assert facts["q1_answer"].endswith(
        "Of the new policies, RACE-RP serves 95.0% of the quick reuses and "
        "25% of those that waited over 30 s.")
    assert "of the two new" not in text(html)
    assert "between FIFO, LRU and LFU (and RACE-RP)?" in text(html)


# --- Point 2 ----------------------------------------------------------------


def test_q2_lists_the_new_policies_against_lru():
    facts = {}
    t = text(nv.q2(summary(), facts))
    assert ("LRU serves 56.0% of the mix's prompt tokens from the cache, LFU "
            "54.0%, FIFO 50.0%, TTDF 57.0% and RACE-RP 55.5%, out of 78.0% "
            "possible") in t
    assert ("Over the whole mix, TTDF beats LRU, LFU and FIFO are lower than "
            "LRU and RACE-RP ties with it, on every seed (2, 3 and 4).") in t
    assert ("TTDF beats LRU on ShareGPT and AgentBank, is lower than LRU on "
            "Bailian and ties with it on CC-Bench") in t
    assert "RACE-RP ties with it on ShareGPT, AgentBank, CC-Bench and " \
        "Bailian" in t
    assert "between FIFO, LRU and LFU (and TTDF and RACE-RP)?" in t
    assert facts["q2_answer"].startswith(
        "Of the mix's prompt tokens, LRU serves 56.0% from the cache, LFU "
        "54.0%, FIFO 50.0%, TTDF 57.0% and RACE-RP 55.5%, against 78.0% with "
        "no eviction.")


def test_q2_without_new_policies():
    facts = {}
    t = text(nv.q2(baseline(), facts))
    assert "from the cache, LFU 54.0% and FIFO 50.0%, out of 78.0%" in t
    assert "between FIFO, LRU and LFU?" in t
    assert "TTDF" not in t and "RACE-RP" not in facts["q2_answer"]


def test_q2_one_new_policy():
    facts = {}
    t = text(nv.q2(summary(engine=("ttdf", )), facts))
    assert "between FIFO, LRU and LFU (and TTDF)?" in t
    assert "RACE-RP" not in t
    assert "LFU 54.0%, FIFO 50.0% and TTDF 57.0%, out of 78.0%" in t


# --- Point 4 ----------------------------------------------------------------


@pytest.mark.parametrize("verdicts, words", [
    (["b"] * 4, "trails LRU at every size"),
    (["a"] * 4, "beats LRU at every size"),
    (["tie"] * 4, "ties with LRU at every size"),
    (["a", "a", "b", "tie"
      ], "beats LRU at 32K to 64K, trails it at 128K and ties at 256K"),
    (["tie", "b", "b", "tie"
      ], "trails LRU at 64K to 128K and ties at 32K and 256K"),
    (["tie", "a", "a", "a"], "beats LRU at 64K to 256K and ties at 32K"),
    (["a", "b", "a", "b"
      ], "beats LRU at 32K and 128K and trails it at 64K and 256K"),
])
def test_vs_lru_runs(verdicts, words):
    assert run(verdicts) == words


def test_q4_adds_the_new_policies_against_lru():
    s = summary()
    facts = {}
    t = text(nv.q4(s, rp.build_data(s), facts))
    assert ("The new policies at 128K: TTDF 58.0% and RACE-RP 58.0%. Against "
            "LRU, TTDF beats LRU at 32K to 64K, trails it at 128K and ties at "
            "256K; RACE-RP trails LRU at every size.") in t
    assert facts["q4_new"] == (
        "TTDF beats LRU at 32K to 64K, trails it at 128K and ties at 256K; "
        "RACE-RP trails LRU at every size")
    assert facts["q4_answer"].endswith(" " + facts["q4_new"] + ".")


def test_q4_new_line_follows_the_drawn_curves():
    # Engine runs for both, a replay curve for RACE-RP only.
    s = summary(replay=("race_rp", ))
    facts = {}
    t = text(nv.q4(s, rp.build_data(s), facts))
    assert "The new policies at 128K: RACE-RP 58.0%." in t
    assert facts["q4_new"] == "RACE-RP trails LRU at every size"
    # No new curves: no line, and the answer is as before.
    facts = {}
    s = baseline()
    t = text(nv.q4(s, rp.build_data(s), facts))
    assert "new policies" not in t and "q4_new" not in facts
    assert facts["q4_answer"].endswith("Belady with 128K (72%) beats LRU "
                                       "with 256K.")


# --- Point 7 ----------------------------------------------------------------


def best_cv():
    """32K and 64K: TTDF a point above LRU on every seed. 128K: LRU a
    point above RACE-RP on average, but one seed disagrees. 256K: RACE-RP
    0.2 points above LRU."""
    return {
        "lru": {
            "hit_rate": [0.50, 0.51, 0.60, 0.65],
            "seeds": [[0.49, 0.50, 0.51], [0.50, 0.51, 0.52],
                      [0.58, 0.60, 0.62], [0.65] * 3]
        },
        "ttdf": {
            "hit_rate": [0.52, 0.53, 0.40, 0.40],
            "seeds": [[0.51, 0.52, 0.53], [0.52, 0.53, 0.54], [0.40] * 3,
                      [0.40] * 3]
        },
        "race_rp": {
            "hit_rate": [0.30, 0.30, 0.59, 0.652],
            "seeds": [[0.30] * 3, [0.30] * 3, [0.59] * 3, [0.652] * 3]
        },
    }


def test_best_by_size_uses_the_tie_rule_and_merges_sizes():
    assert nv._best_by_size(CAPS, best_cv(), [
        "lru", "ttdf", "race_rp"
    ]) == ("TTDF at 32K to 64K; LRU and RACE-RP (a tie) at 128K; RACE-RP and "
           "LRU (a tie) at 256K")
    # One size, one winner.
    assert nv._best_by_size(
        [32768], {
            "lru": {
                "hit_rate": [0.6],
                "seeds": [[0.6, 0.6]]
            },
            "fifo": {
                "hit_rate": [0.5],
                "seeds": [[0.5, 0.5]]
            },
        }, ["fifo", "lru"]) == "LRU at 32K"


def test_per_s_formats_rates():
    assert nv._per_s(None) == "–"
    assert nv._per_s(7.04) == "7.0"
    assert nv._per_s(465.3) == "465"
    assert nv._per_s(1234.4) == "1,234"


def test_overhead_table_fills_gaps_with_dashes():
    rows = table_rows(
        nv._overhead_table(summary(), ["lru", "lfu", "ttdf", "race_rp"],
                           FIXED))
    assert rows == [
        ["LRU", "1.5 µs", "–", "–", "–"],
        ["LFU", "–", "–", "–", "–"],
        ["TTDF", "10.0 µs", "12.3 µs", "250", "1,234 KiB"],
        # The engine has no count of blocks scored: the replay's.
        ["RACE-RP", "25.0 µs", "30.0 µs", "300", "2,048 KiB"],
    ]
    # A None engine value falls back to the replay's too.
    s = summary()
    eng = s["new_policies"]["overhead"]["engine"]["ttdf"]
    eng["candidates_per_scan"] = eng["metadata_kib"] = None
    assert table_rows(nv._overhead_table(s, ["ttdf"], FIXED)) == \
        [["TTDF", "10.0 µs", "12.3 µs", "240", "1,000 KiB"]]
    html = nv._overhead_table(summary(), ["ttdf"], 32768)
    assert "(32K)" in html
    assert table_rows(html) == [["TTDF", "–", "12.3 µs", "250", "1,234 KiB"]]


def test_tuning_block_shows_the_top_three_and_the_specified_setting():
    s = summary()
    html = nv._tuning_block(s, s["capacity_window"], list(NEW))
    t = text(html)
    assert "Tuning on seed 1" in t
    assert "at 32K, 64K, 128K, 256K and 384K" in t
    assert "Seed-1 usable reuse, mean of 5 sizes" in t
    assert table_rows(html) == [
        ["TTDF", "tau 1 s / 120 s, w 0.7 / 0.3", "53.0%", "picked"],
        ["TTDF", "tau 5 s / 600 s, w 0.7 / 0.3", "52.0%", "–"],
        ["TTDF", "tau 30 s / 120 s, w 0.7 / 0.3", "51.0%", "–"],
        ["TTDF", "tau 5 s / 120 s, w 0.7 / 0.3", "50.0%", "as specified"],
        ["RACE-RP", "L 16, P 6", "60.0%", "as specified, picked"],
        ["RACE-RP", "L 64, P 6", "59.0%", "–"],
        ["RACE-RP", "L 16, P 4", "58.0%", "–"],
    ]
    assert ("On seeds 2, 3 and 4 (run once, after the choice), TTDF with the "
            "seed-1 pick (tau 1 s / 120 s, w 0.7 / 0.3) is higher at 32K to "
            "64K and tied at 128K to 256K than as specified, by at most 1.0 "
            "point.") in t
    assert ("For RACE-RP the specified setting was also the best on seed 1, "
            "so it is the one reported.") in t


def test_tuning_block_held_out_lower_and_missing():
    s = summary()
    tuned = [0.43, 0.53, 0.57, 0.652]  # a point lower at 128K
    s["capacity_window"]["curve"]["ttdf_tuned"] = {
        "hit_rate": tuned,
        "seeds": [[v - 0.01, v, v + 0.01] for v in tuned]
    }
    t = text(nv._tuning_block(s, s["capacity_window"], ["ttdf"]))
    assert ("is higher at 32K to 64K, lower at 128K and tied at 256K than as "
            "specified, by at most 1.0 point.") in t
    # No complete held-out curve: no comparison, and no claim either way.
    s["capacity_window"]["curve"]["ttdf_tuned"]["hit_rate"][0] = None
    t = text(nv._tuning_block(s, s["capacity_window"], ["ttdf"]))
    assert "On seeds" not in t and "was also the best" not in t
    del s["capacity_window"]["curve"]["ttdf_tuned"]
    assert "On seeds" not in text(
        nv._tuning_block(s, s["capacity_window"], ["ttdf"]))


def test_tuning_block_specified_in_the_top_three_is_not_repeated():
    s = summary()
    tune = tuning()
    cands = tune["candidates"]["ttdf"]
    cands[1], cands[3] = cands[3], cands[1]  # specified ranks second
    s["new_policies"]["tuning"] = tune
    rows = table_rows(nv._tuning_block(s, s["capacity_window"], ["ttdf"]))
    assert [r[3] for r in rows] == ["picked", "as specified", "–"]


def test_tuning_block_is_empty_without_tuning():
    s = summary(tune=False)
    assert nv._tuning_block(s, s["capacity_window"], list(NEW)) == ""


def test_q7_is_empty_without_its_data():
    s = baseline()
    facts = {}
    assert nv.q7(s, rp.build_data(s), facts) == ""
    assert nv.q7(s, {}, facts) == "" and "q7_answer" not in facts


def test_q7_with_engine_runs():
    s = summary()
    facts = {}
    html = nv.q7(s, rp.build_data(s), facts)
    t = text(html)
    assert '<span class="eyebrow">Point 7 · Mixed workload</span>' in html
    assert ("Do TTDF and RACE-RP keep more of the mix than LRU, LFU and "
            "FIFO?") in t
    assert html.count('id="fig-q7-workloads"') == 1
    assert html.count('id="fig-q7-evictions"') == 1
    claim = (
        "Over the whole mix (cache replay), TTDF beats LRU at 32K to 64K, "
        "trails it at 128K and ties at 256K; RACE-RP trails LRU at every "
        "size")
    assert claim in t
    assert (
        "ShareGPT : TTDF beats LRU at 32K to 64K, trails it at 128K and "
        "ties at 256K; RACE-RP trails LRU at every size. At 128K: LRU 30%, "
        "LFU 29%, FIFO 25%, TTDF 28%, RACE-RP 28%.") in t
    best = ("TTDF at 32K to 64K; LRU at 128K; TTDF and LRU (a tie) at 256K")
    assert f"Highest usable reuse over the whole mix, by size: {best}." in t
    # Evictions (engine runs).
    assert ("Under every policy most evicted blocks are Bailian's; TTDF "
            "evicts the fewest in all (465 a second)") in t
    assert ("AgentBank: LRU 7.0/s (30% needed again), LFU 7.2/s (30% needed "
            "again), FIFO 7.3/s (30% needed again), TTDF 6.3/s (30% needed "
            "again), RACE-RP 6.6/s.") in t
    assert ("TTDF: 57.0% of the mix's prompt tokens, against LRU's 56.0% "
            "(higher than LRU on every seed); by workload, ShareGPT 17% (LRU "
            "16%)") in t
    assert ("RACE-RP: 55.5% of the mix's prompt tokens, against LRU's 56.0% "
            "(a tie with LRU by the rule)") in t
    # Cost and tuning.
    assert "What the new policies cost" in t
    assert "Blocks scored per scan" in t
    assert "Tuning on seed 1" in t
    assert facts["q7_answer"] == (
        f"{claim}. On the engine at 128K: TTDF 57.0% and RACE-RP 55.5%, LRU "
        f"56.0%. Best by size: {best}.")
    assert facts["q7_names"] == "TTDF and RACE-RP"


def test_q7_with_one_new_policy():
    s = summary(engine=("race_rp", ), replay=("race_rp", ))
    facts = {}
    t = text(nv.q7(s, rp.build_data(s), facts))
    assert ("Does RACE-RP keep more of the mix than LRU, LFU and FIFO?") in t
    assert "TTDF" not in facts["q7_answer"]
    assert facts["q7_answer"].startswith(
        "Over the whole mix (cache replay), RACE-RP trails LRU at every size."
        " On the engine at 128K: RACE-RP 55.5%, LRU 56.0%.")
    assert facts["q7_names"] == "RACE-RP"


def test_one_new_policy_page_names_only_it():
    secs = page(summary(engine=("race_rp", ), replay=("race_rp", )))
    whole = "\n".join(secs)
    assert '<span class="q">7 · RACE-RP</span>' in secs[1]
    assert whole.count("(and RACE-RP)?") == 2  # points 1 and 2
    assert "(and TTDF" not in whole
    assert "Does RACE-RP keep more of the mix" in text(whole)


def test_q7_evictions_claim_when_workloads_differ_and_lower_verdicts():
    s = summary()
    s["new_policies"]["evictions"]["ttdf"]["ccbench"]["per_s"] = 1000.0
    s["mixed_groups"]["all"]["race_rp"]["verdict"] = "lower"
    t = text(nv.q7(s, rp.build_data(s), {}))
    assert "TTDF evicts the fewest blocks (465 a second)" in t
    assert "Under every policy" not in t
    assert "(lower than LRU on every seed)" in t


def test_q7_without_engine_runs_has_no_evictions_chart():
    s = summary(engine=())
    facts = {}
    html = nv.q7(s, rp.build_data(s), facts)
    assert 'id="fig-q7-workloads"' in html
    assert "fig-q7-evictions" not in html
    assert "On the engine" not in facts["q7_answer"]
    assert facts["q7_answer"].startswith("Over the whole mix (cache replay), "
                                         "TTDF beats LRU")
    assert "Tuning on seed 1" in text(html)


# --- The whole page ---------------------------------------------------------


def test_sections_add_part_c_and_point_7_before_the_method():
    secs = page(summary())
    assert len(secs) == 14
    assert "Part C · Point 7" in secs[-3]
    assert 'id="point-7"' in secs[-2]
    assert 'aria-labelledby="method-h"' in secs[-1]
    assert "The seven answers" in secs[1]
    assert "7 · TTDF and RACE-RP" in secs[1]
    whole = "\n".join(secs)
    assert len(re.findall(r'class="eyebrow">Point \d · ', whole)) == 7
    assert " of 6 · " not in whole and " of 7 · " not in whole


def test_sections_without_new_policies_are_the_six_point_page():
    secs = page(baseline())
    assert len(secs) == 12
    whole = "\n".join(secs)
    assert "Part C" not in whole and 'id="point-7"' not in whole
    assert "The six answers" in secs[1]
    assert secs[1].count('class="answer"') == 6
    assert "answers six questions" in text(secs[0])
    assert "TTDF" not in whole and "RACE-RP" not in whole
    assert "(and TTDF" not in whole
    eyebrows = re.findall(r'class="eyebrow">(Point [^<]*)<', whole)
    assert [e.split(" · ")[0] for e in eyebrows] == \
        [f"Point {n}" for n in range(1, 7)]
    assert " of 6 · " not in whole


@pytest.mark.parametrize("kw", [{}, {
    "engine": ()
}, {
    "replay": ()
}, {
    "engine": (),
    "replay": ()
}],
                         ids=["both", "replay-only", "engine-only", "none"])
def test_header_question_count_matches_the_answers(kw):
    secs = page(summary(**kw))
    cards = secs[1].count('class="answer"')
    word = {6: "six", 7: "seven"}[cards]
    assert f"answers {word} questions" in text(secs[0])
    assert ("Part C · Point 7" in "".join(secs)) == (cards == 7)


@pytest.mark.parametrize("kw, point7, new", [
    ({}, True, True),
    ({
        "engine": ()
    }, True, True),
    ({
        "replay": ()
    }, False, True),
    ({
        "engine": (),
        "replay": ()
    }, False, False),
],
                         ids=["both", "replay-only", "engine-only", "none"])
def test_point7_and_new_policy_gates(kw, point7, new):
    s = summary(**kw)
    assert nv._has_point7(s) is point7
    assert nv._has_new(s) is new


def test_replay_only_page_shows_and_defines_the_new_policies():
    secs = page(summary(engine=()))
    whole = "\n".join(secs)
    head = text(secs[0])
    assert "answers seven questions" in head
    assert "two new policies, TTDF and RACE-RP" in head
    assert "The seven answers" in secs[1]
    assert 'id="point-7"' in whole and "Part C · Point 7" in whole
    assert "<dt>TTDF</dt>" in whole and "<dt>RACE-RP</dt>" in whole
    method = text(secs[-1])
    assert "reuse_evictors.py" in method
    assert "Settings were tuned on seed 1 only" in method
    assert "RACE-RP's sketch was learned" in method
    assert "Engine runs of" not in method  # none ran on the engine
    # Points 1 and 2 stay on the baselines.
    assert "(and TTDF and RACE-RP)" not in whole


def test_engine_only_page_has_six_points_but_names_the_new_policies():
    secs = page(summary(replay=()))
    whole = "\n".join(secs)
    head = text(secs[0])
    assert "answers six questions" in head
    assert "two new policies, TTDF and RACE-RP" in head
    assert "The six answers" in secs[1]
    assert secs[1].count('class="answer"') == 6
    assert "Part C" not in whole and 'id="point-7"' not in whole
    assert "<dt>TTDF</dt>" in whole and "<dt>RACE-RP</dt>" in whole
    assert "Engine runs of TTDF and RACE-RP" in text(secs[-1])
    # The engine views of points 1 and 2 hold them.
    assert whole.count("(and TTDF and RACE-RP)?") == 2


# --- Sampled LRFU, and one, two or three new policies -----------------------

ONLY_LRFU = {"engine": ("sampled_lrfu", ), "replay": ("sampled_lrfu", )}
ONLY_RACE = {"engine": ("race_rp", ), "replay": ("race_rp", )}
PAIR = {}
THREE = {"engine": ALL_NEW, "replay": ALL_NEW}
STATES = [ONLY_LRFU, ONLY_RACE, PAIR, THREE]
STATE_IDS = ["lrfu", "race_rp", "ttdf+race_rp", "three"]


def present(kw):
    s = summary(**kw)
    return s, nv._new_names(s)


@pytest.mark.parametrize(
    "kw, names", [
        ({
            "engine": ("sampled_lrfu", ),
            "replay": ()
        }, ["sampled_lrfu"]),
        ({
            "engine": (),
            "replay": ("sampled_lrfu", )
        }, ["sampled_lrfu"]),
        ({
            "engine": ("sampled_lrfu", ),
            "replay": ("ttdf", )
        }, ["ttdf", "sampled_lrfu"]),
        ({
            "engine": ("sampled_lrfu", "race_rp"),
            "replay": ()
        }, ["race_rp", "sampled_lrfu"]),
        (THREE, list(ALL_NEW)),
        ({
            "engine": (),
            "replay": ()
        }, []),
    ],
    ids=["engine", "replay", "union", "order", "three", "none"])
def test_new_names_are_the_engine_views_and_point_7(kw, names):
    s = summary(**kw)
    assert nv._new_names(s) == names
    assert nv._has_new(s) is bool(names)


@pytest.mark.parametrize("kw, sentence", [
    (ONLY_LRFU, "we also tried one new policy, Sampled LRFU, which reads no "
     "workload labels."),
    (PAIR, "we also tried two new policies, TTDF and RACE-RP, which read no "
     "workload labels."),
    (THREE, "we also tried three new policies, TTDF, RACE-RP and Sampled "
     "LRFU, which read no workload labels."),
],
                         ids=["one", "two", "three"])
def test_header_counts_and_names_the_new_policies(kw, sentence):
    t = text(nv.header(summary(**kw)))
    assert sentence in t
    assert "answers seven questions" in t


@pytest.mark.parametrize("kw", STATES, ids=STATE_IDS)
def test_terms_have_one_entry_per_new_policy(kw):
    s, names = present(kw)
    html = nv.terms(s)
    entries = dict(re.findall(r"<dt>(.*?)</dt><dd>(.*?)</dd>", html))
    terms = list(entries)
    i = terms.index("LRU, LFU, FIFO")
    labels = [rp.POLICY_LABELS[p] for p in names]
    assert terms[i + 1:i + 2 + len(names)] == [*labels, "Belady"]
    for p in names:
        assert entries[rp.POLICY_LABELS[p]] == nv.TERM[p]
    assert ("Sampled least recently / frequently used" in html) is \
        ("sampled_lrfu" in names)


@pytest.mark.parametrize("kw", STATES, ids=STATE_IDS)
def test_q1_explains_each_new_policy_present(kw):
    s, names = present(kw)
    facts = {}
    html = nv.q1(s, facts)
    for p in rp.NEW_POLICIES:
        assert (nv.WHY[p] in html) is (p in names)
    labels = [rp.POLICY_LABELS[p] for p in names]
    assert [r[0] for r in table_rows(html)][3:] == labels
    assert f"between FIFO, LRU and LFU (and {nv._list(labels)})?" in \
        text(html)
    lead = {
        1: "Of the new policies, ",
        2: "Of the two new policies, ",
        3: "Of the three new policies, "
    }[len(names)]
    assert lead + labels[0] + " serves" in facts["q1_answer"]


def test_q1_three_new_policies():
    facts = {}
    html = nv.q1(summary(**THREE), facts)
    assert table_rows(html)[-1] == \
        ["Sampled LRFU", "96.3%", "35.0%", "0.0%", "0.0%"]
    assert facts["q1_answer"].endswith(
        "Of the three new policies, TTDF serves 97.7% of the quick reuses "
        "and 5% of those that waited over 30 s, RACE-RP serves 95.0% of the "
        "quick reuses and 25% of those that waited over 30 s and Sampled "
        "LRFU serves 96.3% of the quick reuses and 0% of those that waited "
        "over 30 s.")
    assert "under a second of this mix's traffic" in text(html)


LEAF_ONLY = ("only {} out a block that no running request uses and that "
             "has no cached block after it in a prompt")
ANY_BLOCK = ("Sampled LRFU, as LRU, LFU and FIFO do, may throw out any block "
             "no running request uses, even one in the middle of a prompt")

ONE_POLICY = "A policy written for this mix, reading no workload labels."


@pytest.mark.parametrize("kw, count, leaf", [
    (ONLY_LRFU, ONE_POLICY, None),
    (ONLY_RACE, ONE_POLICY, "RACE-RP " + LEAF_ONLY.format("throws")),
    (PAIR, "Two policies written for this mix, none reading workload labels.",
     "TTDF and RACE-RP " + LEAF_ONLY.format("throw")),
    (THREE, "Three policies written for this mix, none reading workload "
     "labels.", "TTDF and RACE-RP " + LEAF_ONLY.format("throw")),
],
                         ids=STATE_IDS)
def test_q7_intro_counts_describes_and_says_what_each_may_evict(
        kw, count, leaf):
    s, names = present(kw)
    html = nv.q7(s, rp.build_data(s), {})
    t = text(html)
    assert count in t
    assert ("none reading" in t) is (len(names) > 1)
    for p in rp.NEW_POLICIES:
        assert (nv.ABOUT[p] in html) is (p in names)
    if leaf is None:
        assert "only throw" not in t
    else:
        assert leaf in t
    assert (ANY_BLOCK in t) is ("sampled_lrfu" in names)
    labels = nv._list([rp.POLICY_LABELS[p] for p in names])
    verb = "Does" if len(names) == 1 else "Do"
    assert f"{verb} {labels} keep more of the mix than LRU, LFU and FIFO?" \
        in t


SCAN = ("every evictable block when the first eviction of a scheduling step "
        "needs it")
SAMPLE = "Sampled LRFU scores only the 4 blocks it draws"


@pytest.mark.parametrize("kw, how", [
    (ONLY_LRFU, f"takes the top. {SAMPLE}. Times are per eviction"),
    (ONLY_RACE, f"takes the top. RACE-RP scores {SCAN}, then reuses that "
     "order for the rest of the step. Times are per eviction"),
    (PAIR, f"takes the top. TTDF and RACE-RP score {SCAN}, then reuse that "
     "order for the rest of the step. Times are per eviction"),
    (THREE, f"takes the top. TTDF and RACE-RP score {SCAN}, then reuse that "
     f"order for the rest of the step; {SAMPLE}. Times are per eviction"),
],
                         ids=STATE_IDS)
def test_q7_cost_paragraph_names_the_present_policies(kw, how):
    s, names = present(kw)
    t = text(nv.q7(s, rp.build_data(s), {}))
    assert how in t
    assert ("TTDF and RACE-RP score" in t) is \
        ({"ttdf", "race_rp"} <= set(names))
    assert (SAMPLE in t) is ("sampled_lrfu" in names)


def test_kib_switches_to_mib_at_1024():
    assert nv._kib(512) == "512 KiB"
    assert nv._kib(1023.4) == "1,023 KiB"
    assert nv._kib(1024) == "1.0 MiB"
    assert nv._kib(1536) == "1.5 MiB"
    assert nv._kib(2048000) == "2,000.0 MiB"


LRFU_MEMORY = ("Of Sampled LRFU's 1.5 MiB, its two numbers per cached block "
               "take 512 KiB; the rest (1.0 MiB) is mostly its record of "
               "which blocks each of the last 4,096 requests counted")
NEVER = ("No request was preempted in these runs, so that record never "
         "changed a count.")


def lrfu_summary(parts=None, preempted=None, views=True):
    """All three new policies; Sampled LRFU's engine metadata split into
    `parts` (KiB), and its preempted requests per seed."""
    s = summary(**THREE)
    eng = s["new_policies"]["overhead"]["engine"]["sampled_lrfu"]
    eng["metadata_kib"] = 1536.0
    if parts is not None:
        eng["metadata_kib_parts"] = parts
    if views:
        s["policy_views"] = {"sampled_lrfu": {}}
        if preempted is not None:
            s["policy_views"]["sampled_lrfu"]["preempted"] = preempted
    return s


PARTS = {"lrfu": 512.0, "bookkeeping": 1024.0}


@pytest.mark.parametrize("parts",
                         [None, {}, {
                             "lrfu": 512.0
                         }, {
                             "bookkeeping": 1024.0
                         }],
                         ids=["none", "empty", "no-bookkeeping", "no-lrfu"])
def test_lrfu_memory_needs_both_parts(parts):
    s = lrfu_summary(parts, [0, 0, 0])
    assert nv._lrfu_memory(s, ALL_NEW) == ""


def test_lrfu_memory_needs_sampled_lrfu_on_the_page():
    s = lrfu_summary(PARTS, [0, 0, 0])
    assert nv._lrfu_memory(s, NEW) == ""
    # Sampled LRFU on the page but without engine overhead: nothing.
    del s["new_policies"]["overhead"]["engine"]["sampled_lrfu"]
    assert nv._lrfu_memory(s, ALL_NEW) == ""


@pytest.mark.parametrize("preempted, views, never", [
    ([0, 0, 0], True, True),
    ([0, 2, 0], True, False),
    (None, True, False),
    (None, False, False),
],
                         ids=[
                             "none-preempted", "some-preempted",
                             "no-preempted", "no-policy-views"
                         ])
def test_lrfu_memory_and_the_preemption_sentence(preempted, views, never):
    s = lrfu_summary(PARTS, preempted, views)
    html = nv._lrfu_memory(s, ALL_NEW)
    assert html.startswith('<p class="prose">') and html.endswith("</p>")
    t = text(html)
    assert t.startswith(LRFU_MEMORY)
    assert ("which keeps a request that is computed again from counting a "
            "block twice.") in t
    assert (NEVER in t) is never


def test_q7_puts_the_lrfu_memory_after_the_overhead_table():
    s = lrfu_summary(PARTS, [0, 0, 0])
    t = text(nv.q7(s, rp.build_data(s), {}))
    assert LRFU_MEMORY in t and NEVER in t
    assert t.index("Policy metadata") < t.index(LRFU_MEMORY) < \
        t.index("Tuning on seed 1")
    # The default summary has no split of Sampled LRFU's metadata.
    s = summary(**THREE)
    assert "two numbers per cached block" not in text(
        nv.q7(s, rp.build_data(s), {}))


def test_q7_tuning_skips_policies_it_lacks():
    # Tuned before Sampled LRFU existed; point 7 shows all three.
    s = summary(**THREE)
    s["new_policies"]["tuning"] = tuning(NEW)
    rows = table_rows(nv._tuning_block(s, s["capacity_window"], ALL_NEW))
    assert {r[0] for r in rows} == {"TTDF", "RACE-RP"}
    secs = page(s)  # the whole page builds
    t = text(secs[-2])
    assert "Tuning on seed 1" in t
    assert "For RACE-RP the specified setting was also the best" in t
    assert "For Sampled LRFU" not in t
    assert "Sampled LRFU's half-life" not in text(secs[-1])


def test_q7_without_tuning_of_its_policies_has_no_tuning_block():
    # Point 7 shows TTDF and RACE-RP; only Sampled LRFU was tuned.
    s = summary()
    s["new_policies"]["tuning"] = tuning(("sampled_lrfu", ))
    assert nv._tuning_block(s, s["capacity_window"], list(NEW)) == ""
    secs = page(s)
    point7 = text(secs[-2])
    assert 'id="point-7"' in secs[-2]
    assert "Tuning on seed 1" not in point7
    assert "Each setting was replayed on seed 1 only" not in point7
    assert "Seed-1 usable reuse" not in point7


def test_q7_three_new_policies():
    s = summary(**THREE)
    facts = {}
    html = nv.q7(s, rp.build_data(s), facts)
    t = text(html)
    assert ("; Sampled LRFU beats LRU at 128K to 256K, trails it at 32K and "
            "ties at 64K") in t
    assert ("Highest usable reuse over the whole mix, by size: TTDF at 32K "
            "to 64K; Sampled LRFU at 128K to 256K.") in t
    assert "Sampled LRFU scores only the 4 blocks it draws" in t
    assert ["Sampled LRFU", "0.5 µs", "0.8 µs", "4", "64 KiB"] in \
        table_rows(html)
    assert [
        "Sampled LRFU", "half-life 512, 4 drawn", "62.0%",
        "as specified, picked"
    ] in table_rows(html)
    assert ("For Sampled LRFU the specified setting was also the best on "
            "seed 1") in t
    assert "Bailian: LRU 400/s" in t and "Sampled LRFU 392/s" in t
    assert facts["q7_names"] == "TTDF, RACE-RP and Sampled LRFU"
    assert "On the engine at 128K: TTDF 57.0%, RACE-RP 55.5% and Sampled " \
        "LRFU 56.5%, LRU 56.0%." in facts["q7_answer"]


def test_new_method_items_for_sampled_lrfu_alone():
    items = [text(x) for x in nv._new_method(summary(**ONLY_LRFU))]
    assert items[0].startswith("Sampled LRFU ( vllm/core/sampled_lrfu.py ) is "
                               "a vLLM evictor beside LRU, LFU and FIFO")
    assert "reuse_evictors.py" not in items[0]
    assert "The cache tells it each time" in items[0]
    # The summary holds RACE-RP's sketch, but RACE-RP is not on the page.
    assert not any("sketch was learned" in x for x in items)
    assert not any(x.startswith("TTDF and RACE-RP stamp") for x in items)
    assert sum(
        x.startswith("Sampled LRFU's clock counts block uses")
        for x in items) == 1
    tuned = next(x for x in items if x.startswith("Settings were tuned"))
    assert ("in the cache replay: Sampled LRFU's half-life (64 to 2,097,152 "
            "block uses) and sample size (4, 16 or 64). Seeds") in tuned
    assert "TTDF's" not in tuned and "RACE-RP's" not in tuned
    assert items[-1].startswith("Engine runs of Sampled LRFU: 128K")


def test_new_method_items_for_race_rp_alone():
    items = [text(x) for x in nv._new_method(summary(**ONLY_RACE))]
    assert items[0].startswith("RACE-RP ( vllm/core/reuse_evictors.py ) is a "
                               "vLLM evictor beside")
    assert "sampled_lrfu.py" not in items[0]
    assert "The cache tells it each time" in items[0]
    assert items[1].startswith("RACE-RP stamps every use and eviction")
    assert "and evicts only idle blocks" in items[1]
    assert not any("TTDF" in x for x in items)
    assert not any(x.startswith("Sampled LRFU's clock") for x in items)
    assert sum("RACE-RP's sketch was learned" in x for x in items) == 1


def test_new_method_tuning_lists_the_tuned_policies_only():
    s = summary(**THREE)
    items = [text(x) for x in nv._new_method(s)]
    assert items[0].startswith("TTDF, RACE-RP and Sampled LRFU ( vllm/core/"
                               "reuse_evictors.py , vllm/core/sampled_lrfu.py"
                               " ) are vLLM evictors")
    assert "The cache tells them each time" in items[0]
    assert sum("RACE-RP's sketch was learned" in x for x in items) == 1
    assert sum(x.startswith("TTDF and RACE-RP stamp") for x in items) == 1
    assert sum(x.startswith("Sampled LRFU's clock") for x in items) == 1
    s["new_policies"]["tuning"] = tuning(("ttdf", "sampled_lrfu"))
    tuned = next(
        text(x) for x in nv._new_method(s)
        if x.startswith("Settings were tuned"))
    assert ("tally) and Sampled LRFU's half-life (64 to 2,097,152 block uses)"
            " and sample size (4, 16 or 64).") in tuned
    assert "RACE-RP's" not in tuned
    # Tuning without candidates: no tuning item.
    s["new_policies"]["tuning"] = {"candidates": {}}
    assert not any(
        x.startswith("Settings were tuned") for x in nv._new_method(s))


def test_three_new_policy_page():
    secs = page(summary(**THREE))
    whole = "\n".join(secs)
    assert len(secs) == 14
    assert "answers seven questions" in text(secs[0])
    assert ('<span class="q">7 · TTDF, RACE-RP and Sampled LRFU</span>'
            in secs[1])
    assert whole.count("(and TTDF, RACE-RP and Sampled LRFU)?") == 2
    assert whole.count("<dt>Sampled LRFU</dt>") == 1
    assert "Sampled LRFU's clock counts block uses" in secs[-1]
