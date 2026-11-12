"""The results page's words with the new policies, TTDF, RACE-RP and
Sampled LRFU (``analysis/m3t2_narrative.py``): they join points 1, 2 and
4, get a point 7, and leave the page as it was when the summary does not
hold them."""
import re
from typing import Any, Dict

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


TIE_RULE = ("A difference counts only if all seeds put the two policies in "
            "the same order and the mean difference is at least 0.5 points; "
            "otherwise it is a tie.")


@pytest.mark.parametrize("kw, clean, loose", [
    ({
        "engine": (),
        "replay": ()
    }, "LRU and LFU", "FIFO"),
    ({}, "LRU, LFU, TTDF and RACE-RP", "FIFO"),
    ({
        "engine": ("sampled_lrfu", ),
        "replay": ("sampled_lrfu", )
    }, "LRU and LFU", "FIFO and Sampled LRFU"),
    ({
        "engine": ALL_NEW,
        "replay": ALL_NEW
    }, "LRU, LFU, TTDF and RACE-RP", "FIFO and Sampled LRFU"),
],
                         ids=["none", "pair", "lrfu", "three"])
def test_glossary_names_the_policies_on_the_page(kw, clean, loose):
    s = summary(**kw)
    entries = dict(re.findall(r"<dt>(.*?)</dt><dd>(.*?)</dd>", nv.terms(s)))
    assert entries["Usable prefix reuse"].endswith(
        f"Weighted by tokens. Under {clean} this equals vLLM's own prefix "
        f"cache hit rate; {loose} can leave cached blocks behind a missing "
        "one, which vLLM's figure counts and this page does not.")
    assert entries["Usable prefix reuse"].endswith(nv._hit_rate_note(s))
    assert entries["Seeds and ties"].endswith(TIE_RULE)


def test_point_eyebrow_has_no_total():
    html = nv.point(7, "Mixed workload", "Q?", "<p>b</p>")
    assert '<span class="eyebrow">Point 7 · Mixed workload</span>' in html
    assert 'id="point-7"' in html and "of 6" not in html


def test_method_describes_the_new_policies_before_reproduce():
    items = re.findall(r"<li>(.*?)</li>", nv.method(summary()))
    first = next(i for i, x in enumerate(items) if "reuse_evictors.py" in x)
    assert items[-1].startswith("Reproduce")
    assert first == len(items) - 7  # six new items, then Reproduce
    base = nv.method(baseline())
    assert "reuse_evictors.py" not in base and "RACE-RP" not in base


def test_new_method_items_follow_the_summary():
    items = [text(x) for x in nv._new_method(summary())]
    assert len(items) == 6
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
    assert "RACE-RP's tables (16 or 64) and bits (1, 2, 3, 4, 6 or 8)" in \
        items[3]
    assert "Sampled LRFU" not in items[3]  # not tuned here
    assert "Seeds 2, 3 and 4 were run once" in items[3]
    assert items[4] == (
        "Seed 1, which trained RACE-RP and tuned all three, is another "
        "30-minute draw of the same four workloads, so seeds 2, 3 and 4 test "
        "new traffic of the same kind, not a different mix.")
    # No engine runs at other sizes: no check runs named.
    assert items[5] == (
        "Engine runs of TTDF and RACE-RP: 128K on seeds 2, 3 and 4 (points "
        "1, 2 and 7). The numbers behind every chart are also in CSV files "
        "under report/milestone3/data/task2 .")
    # No sketch, no tuning, no engine runs: the two policy items and the
    # note on seed 1.
    s = summary(sketch=False, tune=False)
    s["new_policies"]["engine"] = []
    items = [text(x) for x in nv._new_method(s)]
    assert len(items) == 3 and "VLLM_EVICTION_POLICY" in items[0]
    assert items[2].startswith("Seed 1, which trained RACE-RP")
    s = summary(sketch=False)
    items = [text(x) for x in nv._new_method(s)]
    assert len(items) == 5
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


def claim(html):
    """A finding's claim (its heading), as text."""
    match = re.search(r'<div class="claim"><h3>(.*?)</h3>', html)
    assert match is not None
    return text(match[1])


def test_q1_adds_table_rows_and_a_clause_for_the_new_policies():
    facts: Dict[str, Any] = {}
    html = nv.q1(summary(), facts)
    rows = table_rows(html)
    assert [r[0] for r in rows] == ["LRU", "LFU", "FIFO", "TTDF", "RACE-RP"]
    # The last column: the waits past LRU's cut-off (30 s) taken together.
    assert "<th>more than 30 s, together</th>" in html
    assert [r[-1] for r in rows] == ["0.0%", "15.0%", "0.0%", "5.0%", "25.0%"]
    assert rows[3] == ["TTDF", "97.7%", "50.0%", "10.0%", "0.0%", "5.0%"]
    assert rows[4] == ["RACE-RP", "95.0%", "40.0%", "30.0%", "20.0%", "25.0%"]
    # The claim and the answer span every mixed policy.
    assert claim(html) == (
        "LRU and FIFO serve almost no reuse that waited longer than 30 s; LFU "
        "still serves 15% of them, TTDF still serves 5% of them and RACE-RP "
        "still serves 25% of them")
    t = text(html)
    assert "between FIFO, LRU and LFU (and TTDF and RACE-RP)?" in t
    assert "TTDF ranks a block by its recent uses" in t
    new = ("Of the new policies, TTDF serves 97.7% of the quick reuses and "
           "RACE-RP 95.0%.")
    assert new not in t  # in the answer card only
    assert facts["q1_answer"].endswith(
        "LFU, TTDF and RACE-RP still serve 15%, 5% and 25% of those that "
        "waited over 30 s, LFU but only 94.0% of the quick ones. " + new)
    assert facts["lru_cut"] == 30


def test_q1_new_clause_without_an_lru_cutoff():
    # LRU serves some reuse at every wait: no column for the long waits.
    s = summary()
    bins = curve([1.0, 0.99, 0.95, 0.30, 0.05, 0.02])
    s["mixed_intervals"]["lru"] = {"curve": bins, "by_bin": coarse(bins)}
    facts: Dict[str, Any] = {}
    html = nv.q1(s, facts)
    assert facts["lru_cut"] is None
    assert "together" not in html
    assert {len(r) for r in table_rows(html)} == {5}
    assert claim(html) == "LRU serves some reuse at every wait"
    assert facts["q1_answer"].endswith(
        "Of the new policies, TTDF serves 97.7% of the quick reuses and "
        "RACE-RP 95.0%.")


def test_q1_without_new_policies_keeps_the_three_rows():
    facts: Dict[str, Any] = {}
    html = nv.q1(baseline(), facts)
    assert [r[0] for r in table_rows(html)] == ["LRU", "LFU", "FIFO"]
    t = text(html)
    assert "between FIFO, LRU and LFU?" in t
    assert "TTDF" not in t and "new policies" not in facts["q1_answer"]
    assert "even one used a moment ago.</p>" in html  # no trailing space


def test_q1_one_new_policy():
    facts: Dict[str, Any] = {}
    html = nv.q1(summary(engine=("race_rp", )), facts)
    assert [r[0] for r in table_rows(html)][-1] == "RACE-RP"
    assert "TTDF serves" not in facts["q1_answer"]
    assert facts["q1_answer"].endswith(
        "Of the new policies, RACE-RP serves 95.0% of the quick reuses.")
    assert claim(html).endswith(
        "; LFU still serves 15% of them and RACE-RP still serves 25% of them")
    assert "between FIFO, LRU and LFU (and RACE-RP)?" in text(html)


# --- Point 2 ----------------------------------------------------------------


def test_q2_lists_the_new_policies_against_lru():
    facts: Dict[str, Any] = {}
    t = text(nv.q2(summary(), facts))
    assert ("LRU serves 56.0% of the mix's prompt tokens from the cache, LFU "
            "54.0%, FIFO 50.0%, TTDF 57.0% and RACE-RP 55.5%, out of 78.0% "
            "possible") in t
    assert ("Over the whole mix, TTDF beats LRU; LFU and FIFO are lower than "
            "LRU; RACE-RP ties with it, on every seed (2, 3 and 4).") in t
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
    facts: Dict[str, Any] = {}
    t = text(nv.q2(baseline(), facts))
    assert "from the cache, LFU 54.0% and FIFO 50.0%, out of 78.0%" in t
    assert "between FIFO, LRU and LFU?" in t
    assert "TTDF" not in t and "RACE-RP" not in facts["q2_answer"]


def test_q2_one_new_policy():
    facts: Dict[str, Any] = {}
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
    facts: Dict[str, Any] = {}
    t = text(nv.q4(s, rp.build_data(s), facts))
    assert ("The new policies at 128K (cache replay): TTDF 58.0% and RACE-RP "
            "58.0%. Across sizes, TTDF is from 2.0 points below to 2.0 points "
            "above LRU; RACE-RP is 2.0 points below LRU.") in t
    assert facts["q4_new"] == (
        "TTDF beats LRU at 32K to 64K, trails it at 128K and ties at 256K "
        "(from 2.0 points below to 2.0 points above); RACE-RP trails LRU at "
        "every size (2.0 points below) (cache replay)")
    assert facts["q4_answer"].endswith(" " + facts["q4_new"] + ".")
    # LFU's tie at 256K (the same mean as LRU) says why it is a tie.
    assert ("LFU beats LRU at 32K to 64K, trails it at 128K and ties at "
            "256K, where the seeds disagree or the gap is under 0.5 points") \
        in facts["q4_answer"]


def gap_cv(diffs):
    return {
        "lru": {
            "hit_rate": [0.5] * len(diffs)
        },
        "x": {
            "hit_rate": [0.5 + d for d in diffs]
        },
    }


@pytest.mark.parametrize("diffs, words", [
    ([0.008, 0.011, 0.01], "0.8 to 1.1 points above"),
    ([0.02, 0.02], "2.0 points above"),
    ([0.0, 0.004], "0.0 to 0.4 points above"),
    ([-0.03, -0.012], "1.2 to 3.0 points below"),
    ([-0.01, 0.05], "from 1.0 point below to 5.0 points above"),
])
def test_gap_text(diffs, words):
    assert nv._gap_text(gap_cv(diffs), "x") == words


LAND = ("The engine runs land within {} of the cache replay of the same "
        "seeds (× beside each marker)")


def q4_text(s):
    return text(nv.q4(s, rp.build_data(s), {}))


def test_q4_engine_note_names_an_outlying_run():
    s = summary()
    # TTDF, seed 3, 64K: 2.0 points off; the others within 0.5 points.
    s["capacity_window"]["engine"].append({
        "seed": 3,
        "policy": "ttdf",
        "capacity": 65536,
        "engine": 0.54,
        "replay": 0.52
    })
    t = q4_text(s)
    assert (LAND.format("2.0 points") + ": the largest gap is TTDF at 64K "
            "(seed 3); the other runs are within 0.5 points.") in t


def test_q4_engine_note_gives_a_reason_for_a_sampled_lrfu_outlier():
    s = summary(**THREE)
    s["capacity_window"]["engine"].append({
        "seed": 3,
        "policy": "sampled_lrfu",
        "capacity": 65536,
        "engine": 0.54,
        "replay": 0.52
    })
    assert (LAND.format("2.0 points") + ": the largest gap is Sampled LRFU "
            "at 64K (seed 3; its random draws depend on the exact order of "
            "events, which differs between the engine and the replay); the "
            "other runs are within 0.5 points.") in q4_text(s)


def test_q4_engine_note_without_an_outlier():
    # The worst gap (0.5 points, LRU at 32K) is under half a point above
    # the next (0.2 points): no run named.
    t = q4_text(summary())
    assert LAND.format("0.5 points") + "." in t
    assert "the largest gap is" not in t
    # Two runs share the worst gap: none stands out.
    s = summary()
    s["capacity_window"]["engine"] += [{
        "seed": x,
        "policy": "lfu",
        "capacity": 65536,
        "engine": 0.53,
        "replay": 0.51
    } for x in (2, 3)]
    t = q4_text(s)
    assert LAND.format("2.0 points") + "." in t
    assert "the largest gap is" not in t


def test_q4_engine_note_with_a_single_engine_run():
    s = summary()
    s["capacity_window"]["engine"] = [{
        "seed": 2,
        "policy": "lru",
        "capacity": FIXED,
        "engine": 0.62,
        "replay": 0.60
    }]
    t = q4_text(s)
    assert LAND.format("2.0 points") + "." in t
    assert "the largest gap is" not in t
    # No engine run with a replay value: no note at all.
    s["capacity_window"]["engine"][0]["replay"] = None
    assert "The engine runs land within" not in q4_text(s)


def test_q4_new_line_follows_the_drawn_curves():
    # Engine runs for both, a replay curve for RACE-RP only.
    s = summary(replay=("race_rp", ))
    facts: Dict[str, Any] = {}
    t = text(nv.q4(s, rp.build_data(s), facts))
    assert "The new policies at 128K (cache replay): RACE-RP 58.0%." in t
    assert facts["q4_new"] == ("RACE-RP trails LRU at every size (2.0 points "
                               "below) (cache replay)")
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
        # Baselines: no engine timing or metadata, and a heap (no scan).
        ["LRU", "1.5 µs", "not measured", "none (a heap)", "not measured"],
        [
            "LFU", "not measured", "not measured", "none (a heap)",
            "not measured"
        ],
        ["TTDF", "10.0 µs", "12.3 µs", "250", "1.2 MiB"],
        # The engine has no count of blocks scored: the replay's.
        ["RACE-RP", "25.0 µs", "30.0 µs", "300", "2.0 MiB"],
    ]
    # A None engine value falls back to the replay's too.
    s = summary()
    eng = s["new_policies"]["overhead"]["engine"]["ttdf"]
    eng["candidates_per_scan"] = eng["metadata_kib"] = None
    assert table_rows(nv._overhead_table(s, ["ttdf"], FIXED)) == \
        [["TTDF", "10.0 µs", "12.3 µs", "240", "1,000 KiB"]]
    # A new policy with nothing measured: dashes.
    del s["new_policies"]["overhead"]["engine"]["ttdf"]
    del s["new_policies"]["overhead"]["replay"]["ttdf"]
    assert table_rows(nv._overhead_table(s, ["ttdf"], FIXED)) == \
        [["TTDF", "–", "–", "–", "–"]]
    html = nv._overhead_table(summary(), ["ttdf"], 32768)
    assert "(32K)" in html
    assert table_rows(html) == [["TTDF", "–", "12.3 µs", "250", "1.2 MiB"]]


TUNING_INTRO = (
    "Each setting was replayed on seed 1 only (cache replay), at 32K, 64K, "
    "128K, 256K and 384K; LRU averages 45.0% there. We picked the highest "
    "average and kept the specified setting only on an exact tie. The best "
    "three, and the specified setting:")


def test_tuning_block_shows_the_top_three_and_the_specified_setting():
    s = summary()
    html = nv._tuning_block(s, s["capacity_window"], list(NEW))
    t = text(html)
    assert "Tuning on seed 1" in t
    assert TUNING_INTRO in t
    assert "Seed-1 usable reuse, mean of 5 sizes" in t
    assert table_rows(html) == [
        [
            "TTDF", "time constants 1 s / 120 s, weights 0.7 / 0.3", "53.0%",
            "picked"
        ],
        [
            "TTDF", "time constants 5 s / 600 s, weights 0.7 / 0.3", "52.0%",
            "–"
        ],
        [
            "TTDF", "time constants 30 s / 120 s, weights 0.7 / 0.3", "51.0%",
            "–"
        ],
        [
            "TTDF", "time constants 5 s / 120 s, weights 0.7 / 0.3", "50.0%",
            "as specified"
        ],
        ["RACE-RP", "16 tables of 6 bits", "60.0%", "as specified, picked"],
        ["RACE-RP", "64 tables of 6 bits", "59.0%", "–"],
        ["RACE-RP", "16 tables of 4 bits", "58.0%", "–"],
    ]
    notes = re.findall(r"<li>(.*?)</li>", html)
    assert [text(n) for n in notes] == [
        "Then replayed once on seeds 2, 3 and 4 with no further changes "
        "(cache replay), TTDF with the seed-1 pick (time constants 1 s / "
        "120 s, weights 0.7 / 0.3) against as specified: higher at 32K to "
        "64K; tied at 128K to 256K (largest difference 1.0 point).",
        "For RACE-RP the specified setting was also the best on seed 1, so it "
        "is the one reported.",
        "With the seed-1 picks, at 128K on seeds 2, 3 and 4 (cache replay): "
        "TTDF 58.0%, against LRU's 60.0%.",
    ]
    # The pick's seed-1 lead (3 points) clears the tie rule: no note on it.
    assert "less than the 0.5 points" not in t


def test_tuning_block_notes_a_pick_inside_the_tie_rule():
    s = summary()
    pick = s["new_policies"]["tuning"]["selected"]["ttdf"]
    pick["mean"] = pick["spec_mean"] + 0.003
    t = text(nv._tuning_block(s, s["capacity_window"], ["ttdf"]))
    lead = ("TTDF's pick led the specified setting by 0.3 points on seed 1, "
            "less than the 0.5 points this page needs to call a difference.")
    assert lead in t
    # The held-out comparison still follows.
    assert t.index(lead) < t.index("Then replayed once on seeds 2, 3 and 4")
    # Without LRU's seed-1 values, the intro does not quote them.
    del s["new_policies"]["tuning"]["lru"]
    t = text(nv._tuning_block(s, s["capacity_window"], ["ttdf"]))
    assert "LRU averages" not in t
    assert "384K. We picked the highest average" in t


def test_tuning_block_held_out_lower_and_missing():
    s = summary()
    tuned = [0.43, 0.53, 0.57, 0.652]  # a point lower at 128K
    s["capacity_window"]["curve"]["ttdf_tuned"] = {
        "hit_rate": tuned,
        "seeds": [[v - 0.01, v, v + 0.01] for v in tuned]
    }
    t = text(nv._tuning_block(s, s["capacity_window"], ["ttdf"]))
    assert ("against as specified: higher at 32K to 64K; lower at 128K; tied "
            "at 256K (largest difference 1.0 point).") in t
    assert ("With the seed-1 picks, at 128K on seeds 2, 3 and 4 (cache "
            "replay): TTDF 57.0%, against LRU's 60.0%.") in t
    # No complete held-out curve: no comparison, and no claim either way.
    s["capacity_window"]["curve"]["ttdf_tuned"]["hit_rate"][0] = None
    t = text(nv._tuning_block(s, s["capacity_window"], ["ttdf"]))
    assert "Then replayed once" not in t and "was also the best" not in t
    assert "With the seed-1 picks" not in t
    del s["capacity_window"]["curve"]["ttdf_tuned"]
    t = text(nv._tuning_block(s, s["capacity_window"], ["ttdf"]))
    assert "Then replayed once" not in t and "With the seed-1 picks" not in t


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
    facts: Dict[str, Any] = {}
    assert nv.q7(s, rp.build_data(s), facts) == ""
    assert nv.q7(s, {}, facts) == "" and "q7_answer" not in facts


def verdict_items(html):
    """The bullets of point 7's verdict, as text."""
    block = html.split('The verdict</h3>')[1].split("</ul>")[0]
    return [text(x) for x in re.findall(r"<li>(.*?)</li>", block)]


def claims(html):
    return [
        text(x) for x in re.findall(r'<div class="claim"><h3>(.*?)</h3>', html)
    ]


BEST_PAIR = "TTDF at 32K to 64K; LRU at 128K; TTDF and LRU (a tie) at 256K"


def test_q7_with_engine_runs():
    # TTDF and RACE-RP ran; Sampled LRFU has no runs (its replay curve is
    # all None, as m3t2_study writes it).
    s = summary()
    assert None in s["capacity_window"]["curve"]["sampled_lrfu"]["hit_rate"]
    facts: Dict[str, Any] = {}
    html = nv.q7(s, rp.build_data(s), facts)
    t = text(html)
    assert '<span class="eyebrow">Point 7 · Mixed workload</span>' in html
    assert ("Do TTDF and RACE-RP keep more of the mix than LRU, LFU and "
            "FIFO?") in t
    assert html.count('id="fig-q7-workloads"') == 1
    assert html.count('id="fig-q7-evictions"') == 1
    assert "Every chart and number in this point uses the specified " \
        "settings; the settings tuned on seed 1 are compared at the end." in t
    assert verdict_items(html) == [
        "Best overall: TTDF. On the engine at 128K it serves 57.0% of the "
        "mix's prompt tokens from the cache, against LRU's 56.0% (higher on "
        "every seed, by 1.0 point). In the cache replay, the best real "
        f"policy at each size: {BEST_PAIR}. Belady, which knows the future, "
        "is still 12.0 points ahead at 128K (cache replay).",
        "TTDF is from 2.0 points below to 2.0 points above LRU across sizes "
        "(cache replay); against the best of LRU, LFU and FIFO it beats all "
        "three at 32K to 64K, ties with the best of them at 256K and trails "
        "the best of them at 128K. On the engine at 128K it keeps less than "
        "LRU of Bailian.",
        "RACE-RP is 2.0 points below LRU across sizes (cache replay); against "
        "the best of LRU, LFU and FIFO it is below the best of them at every "
        "size.",
        "Why RACE-RP does not help ShareGPT (a likely reason, which we did "
        "not test): half of ShareGPT's reuses wait more than 120 s (point "
        "2); by then a block's recent-use counts have faded, so these scores "
        "rank it with blocks that will not come back.",
        "Part of the lead of TTDF may come from evicting only a prompt's end "
        "rather than from the scores themselves; we did not separate the "
        "two.",
    ]
    assert claims(html) == [
        "At 128K inside the mix (cache replay), TTDF loses on all four; "
        "RACE-RP loses on all four",
        "Under every policy most evicted blocks are Bailian's; TTDF evicts "
        "the fewest (465 a second, on every seed)",
    ]
    assert ("Here a policy beats or trails LRU only when all three seeds put "
            "them in the same order and the mean difference is at least 0.5 "
            "points; otherwise it ties.") in t
    assert ("ShareGPT : TTDF beats LRU at 32K to 64K, trails it at 128K and "
            "ties at 256K; RACE-RP trails LRU at every size. At 128K (cache "
            "replay): LRU 30%, LFU 29%, FIFO 25%, TTDF 28%, RACE-RP 28%.") in t
    # Evictions: the whole mix first; "needed again" defined.
    assert ("Whole mix: LRU 517/s (30% needed again), LFU 533/s (30% needed "
            "again), FIFO 538/s (30% needed again), TTDF 465/s (30% needed "
            "again), RACE-RP 491/s (30% needed again).") in t
    assert t.index("Whole mix: LRU") < t.index("ShareGPT: LRU")
    assert ("AgentBank: LRU 7.0/s (30% needed again), LFU 7.2/s (30% needed "
            "again), FIFO 7.3/s (30% needed again), TTDF 6.3/s (30% needed "
            "again), RACE-RP 6.6/s.") in t
    assert "needed again is the share of those blocks that a later request " \
        "looked up" in t
    assert "Fewer evictions and fewer blocks needed again" not in t
    # The old per-policy engine paragraph is gone.
    assert "of the mix's prompt tokens, against LRU's 56.0% (higher than" \
        not in t
    assert ("At these eviction rates, choosing what to evict takes, of each "
            "second: LRU 0.8 ms (replay timing); TTDF 5.7 ms and RACE-RP "
            "14.7 ms (engine timing).") in t
    assert "Tuning on seed 1" in t
    assert facts["q7_answer"] == (
        "TTDF keeps the most: on the engine at 128K, 57.0% of the mix's "
        "prompt tokens against LRU's 56.0%. RACE-RP is 2.0 points below LRU "
        f"(cache replay). Best real policy by size (cache replay): "
        f"{BEST_PAIR}.")
    assert facts["q7_names"] == "TTDF and RACE-RP"


def test_q7_with_engine_runs_for_a_policy_not_yet_replayed():
    # Engine runs for all three, replay curves for TTDF and RACE-RP only.
    s = summary(engine=ALL_NEW, replay=NEW)
    facts: Dict[str, Any] = {}
    html = nv.q7(s, rp.build_data(s), facts)
    assert "Do TTDF and RACE-RP keep more" in text(html)
    # Sampled LRFU has no curve to rank: the replay ranking leaves it out.
    assert f"best real policy at each size: {BEST_PAIR}." in \
        verdict_items(html)[0]
    assert facts["q7_answer"].endswith(
        f"Best real policy by size (cache replay): {BEST_PAIR}.")
    # It keeps the least on the engine, but point 7 does not show it.
    assert "keeps the least" not in facts["q7_answer"]
    assert len(page(s)) == 14


def test_q7_with_one_new_policy():
    s = summary(engine=("race_rp", ), replay=("race_rp", ))
    facts: Dict[str, Any] = {}
    html = nv.q7(s, rp.build_data(s), facts)
    t = text(html)
    assert ("Does RACE-RP keep more of the mix than LRU, LFU and FIFO?") in t
    # LRU keeps the most on the engine: no "best overall", no lead to
    # explain.
    items = verdict_items(html)
    assert not any(x.startswith("Best overall") for x in items)
    assert not any(x.startswith("Part of the lead") for x in items)
    assert items[0].startswith("RACE-RP is 2.0 points below LRU across sizes")
    assert items[1].startswith("Why RACE-RP does not help ShareGPT")
    assert facts["q7_answer"] == (
        "RACE-RP is 2.0 points below LRU (cache replay). Best real policy by "
        "size (cache replay): LFU at 32K to 64K; LRU at 128K; LRU and LFU (a "
        "tie) at 256K.")
    assert facts["q7_names"] == "RACE-RP"


def test_one_new_policy_page_names_only_it():
    secs = page(summary(engine=("race_rp", ), replay=("race_rp", )))
    whole = "\n".join(secs)
    assert '<span class="q">7 · RACE-RP</span>' in secs[1]
    assert whole.count("(and RACE-RP)?") == 2  # points 1 and 2
    assert "(and TTDF" not in whole
    assert "Does RACE-RP keep more of the mix" in text(whole)
    assert "<h2>One new policy on the shared cache</h2>" in secs[-3]


def test_q7_evictions_claim_when_workloads_differ_or_seeds_cross():
    s = summary()
    s["new_policies"]["evictions"]["ttdf"]["ccbench"]["per_s"] = 1000.0
    # One LFU seed evicts fewer than TTDF's: not lowest "on every seed".
    s["new_policies"]["evictions"]["lfu"]["all"]["seeds"] = [460.0, 540, 540]
    assert claims(nv.q7(s, rp.build_data(s), {}))[1] == \
        "TTDF evicts the fewest (465 a second)"


def test_q7_evictions_link_fewer_evictions_to_fewer_needed_again():
    s = summary()
    s["new_policies"]["evictions"]["ttdf"]["all"]["regretted"] = 0.2
    t = text(nv.q7(s, rp.build_data(s), {}))
    assert ("Fewer evictions and fewer blocks needed again go with more "
            "reuse: TTDF is lowest on both.") in t
    assert "Sampled LRFU is the exception" not in t
    # With Sampled LRFU among the engine runs, its exception; a real 0.0
    # counts as the lowest share, and a missing one as none.
    s = summary(**THREE)
    ev = s["new_policies"]["evictions"]
    ev["ttdf"]["all"]["regretted"] = 0.0
    ev["race_rp"]["all"]["regretted"] = None
    t = text(nv.q7(s, rp.build_data(s), {}))
    assert ("TTDF is lowest on both. Sampled LRFU is the exception: it "
            "evicts no more than LRU yet keeps less, because of the stranded "
            "blocks above.") in t


def test_q7_without_engine_runs_has_no_evictions_chart():
    s = summary(engine=())
    facts: Dict[str, Any] = {}
    html = nv.q7(s, rp.build_data(s), facts)
    assert 'id="fig-q7-workloads"' in html
    assert "fig-q7-evictions" not in html
    items = verdict_items(html)
    assert [x.split(" is ")[0] for x in items] == ["TTDF", "RACE-RP"]
    assert not any("On the engine" in x for x in items)
    # The replay ranking includes the new policies' curves.
    assert facts["q7_answer"] == (
        "TTDF is from 2.0 points below to 2.0 points above LRU (cache "
        "replay). RACE-RP is 2.0 points below LRU (cache replay). Best real "
        f"policy by size (cache replay): {BEST_PAIR}.")
    assert "Tuning on seed 1" in text(html)


# --- The whole page ---------------------------------------------------------


def test_sections_add_part_c_and_point_7_before_the_method():
    secs = page(summary())
    assert len(secs) == 14
    assert "Part C · Point 7" in secs[-3]
    assert "<h2>Two new policies on the shared cache</h2>" in secs[-3]
    assert ("<h2>Each workload alone, and against sharing (LRU, LFU and FIFO "
            "only)</h2>") in "".join(secs)
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
    assert "Sampled LRFU" not in whole
    assert "(and TTDF" not in whole
    assert "<h2>Each workload alone, and against sharing</h2>" in whole
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
PAIR: Dict[str, Any] = {}
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
    facts: Dict[str, Any] = {}
    html = nv.q1(s, facts)
    for p in rp.NEW_POLICIES:
        assert (nv.WHY[p] in html) is (p in names)
    labels = [rp.POLICY_LABELS[p] for p in names]
    assert [r[0] for r in table_rows(html)][3:] == labels
    assert f"between FIFO, LRU and LFU (and {nv._list(labels)})?" in \
        text(html)
    assert f"Of the new policies, {labels[0]} serves" in facts["q1_answer"]
    assert "Of the new policies" not in html


def test_q1_three_new_policies():
    facts: Dict[str, Any] = {}
    html = nv.q1(summary(**THREE), facts)
    assert table_rows(html)[-1] == \
        ["Sampled LRFU", "96.3%", "35.0%", "0.0%", "0.0%", "0.0%"]
    assert claim(html).startswith(
        "LRU, FIFO and Sampled LRFU serve almost no reuse that waited longer "
        "than 30 s; LFU still serves 15% of them")
    assert facts["q1_answer"].endswith(
        "Of the new policies, TTDF serves 97.7% of the quick reuses, RACE-RP "
        "95.0% and Sampled LRFU 96.3%.")
    assert "the lowest-scoring of only 4 idle blocks drawn at random" in \
        text(html)


LEAF_ONLY = ("only {} out a block that no running request uses and that "
             "has no cached block after it in a prompt")
ANY_BLOCK = ("Like LRU, LFU and FIFO, Sampled LRFU may throw out any block no "
             "running request uses, even one in the middle of a prompt")
ONE_POLICY = "A policy written for this mix, reading no workload labels:"


@pytest.mark.parametrize("kw, count, leaf", [
    (ONLY_LRFU, ONE_POLICY, None),
    (ONLY_RACE, ONE_POLICY, "RACE-RP " + LEAF_ONLY.format("throws")),
    (PAIR, "Two policies written for this mix, none reading workload labels:",
     "TTDF and RACE-RP " + LEAF_ONLY.format("throw")),
    (THREE, "Three policies written for this mix, none reading workload "
     "labels:", "TTDF and RACE-RP " + LEAF_ONLY.format("throw")),
],
                         ids=STATE_IDS)
def test_q7_intro_counts_describes_and_says_what_each_may_evict(
        kw, count, leaf):
    s, names = present(kw)
    html = nv.q7(s, rp.build_data(s), {})
    t = text(html)
    assert count in t
    assert ("none reading" in t) is (len(names) > 1)
    # One list item per policy shown, describing it.
    intro = html.split("The verdict")[0]
    for p in rp.NEW_POLICIES:
        assert (f"<li>{nv.ABOUT[p]}</li>" in intro) is (p in names)
    if leaf is None:
        assert "only throw" not in t
    else:
        assert leaf in t
    assert (ANY_BLOCK in t) is ("sampled_lrfu" in names)
    assert ("Every chart and number in this point uses the specified "
            "settings") in t
    labels = nv._list([rp.POLICY_LABELS[p] for p in names])
    verb = "Does" if len(names) == 1 else "Do"
    assert f"{verb} {labels} keep more of the mix than LRU, LFU and FIFO?" \
        in t


SCAN = ("every evictable block (here: idle, with nothing cached after it in "
        "a prompt) when the first eviction of a scheduling step needs it")
SAMPLE = "Sampled LRFU scores only the 4 blocks it draws"
RACE = ("each RACE-RP score hashes five features into 16 tables, which is "
        "why it costs the most")
HEAP = ("LRU, LFU and FIFO keep their idle blocks in a priority queue (a "
        "heap) ordered by one key, so an eviction takes the top. ")


@pytest.mark.parametrize("kw, how", [
    (ONLY_LRFU, f"{SAMPLE}. Times are per eviction"),
    (ONLY_RACE, f"RACE-RP scores {SCAN}, then reuses that order for the rest "
     f"of the step; {RACE}. Times are per eviction"),
    (PAIR, f"TTDF and RACE-RP score {SCAN}, then reuse that order for the "
     f"rest of the step; {RACE}. Times are per eviction"),
    (THREE, f"TTDF and RACE-RP score {SCAN}, then reuse that order for the "
     f"rest of the step; {SAMPLE}; {RACE}. Times are per eviction"),
],
                         ids=STATE_IDS)
def test_q7_cost_paragraph_names_the_present_policies(kw, how):
    s, names = present(kw)
    t = text(nv.q7(s, rp.build_data(s), {}))
    assert HEAP + how in t
    assert ("TTDF and RACE-RP score" in t) is \
        ({"ttdf", "race_rp"} <= set(names))
    assert (SAMPLE in t) is ("sampled_lrfu" in names)
    assert ("each RACE-RP score hashes" in t) is ("race_rp" in names)
    assert ("Policy metadata is the extra memory a policy keeps, counting "
            "each stored number as 8 bytes, at its largest during the run; "
            "it was measured for the new policies only.") in t


@pytest.mark.parametrize("costs", [{
    "ttdf": 50.0
}, {
    "ttdf": None,
    "race_rp": None
}],
                         ids=["ttdf-dearest", "no-engine-times"])
def test_q7_cost_says_race_rp_costs_the_most_only_when_it_does(costs):
    s = summary()
    for p, us in costs.items():
        s["new_policies"]["overhead"]["engine"][p]["evict_us"] = us
    t = text(nv.q7(s, rp.build_data(s), {}))
    assert "each RACE-RP score hashes five features into 16 tables. Times" \
        in t
    assert "costs the most" not in t


def test_cost_scale_per_second():
    s = summary(**THREE)
    q = rp.build_data(s)["q7"]
    pols = q["policies"]
    assert text(nv._cost_scale(s, q, pols)) == (
        "At these eviction rates, choosing what to evict takes, of each "
        "second: LRU 0.8 ms (replay timing); TTDF 5.7 ms, RACE-RP 14.7 ms "
        "and Sampled LRFU 0.4 ms (engine timing).")
    # Only replay timing; only engine timing; no eviction rates at all.
    assert text(nv._cost_scale(s, q, [
        "lru", "lfu"
    ])) == ("At these eviction rates, choosing what to evict takes, of each "
            "second: LRU 0.8 ms (replay timing).")
    assert text(nv._cost_scale(
        s, q,
        ["ttdf"])).endswith("of each second: TTDF 5.7 ms (engine timing).")
    assert nv._cost_scale(s, {"evictions": {}}, pols) == ""
    assert nv._cost_scale(s, {}, pols) == ""


def test_kib_switches_to_mib_at_1024():
    assert nv._kib(512) == "512 KiB"
    assert nv._kib(1023.4) == "1,023 KiB"
    assert nv._kib(1024) == "1.0 MiB"
    assert nv._kib(1536) == "1.5 MiB"
    assert nv._kib(2048000) == "2,000.0 MiB"


LRFU_MEMORY = ("Of Sampled LRFU's 1.5 MiB, its two numbers per cached block "
               "take 512 KiB; the rest (1.0 MiB) is mostly its record of "
               "which blocks each of the last 4,096 requests counted")
NEVER = ("No request was preempted in the 128K runs, so there that record "
         "never changed a count.")


def lrfu_summary(parts=None, preempted=None, views=True):
    """All three new policies; Sampled LRFU's engine metadata split into
    `parts` (KiB), and its preempted requests per seed."""
    s = summary(**THREE)
    eng = s["new_policies"]["overhead"]["engine"]["sampled_lrfu"]
    eng["metadata_kib"] = 1536.0
    if parts is not None:
        eng["metadata_kib_parts"] = parts
    view = s["policy_views"]["sampled_lrfu"]
    if preempted is None:
        del view["preempted"]
    else:
        view["preempted"] = preempted
    if not views:
        del s["policy_views"]
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
    ([], True, False),
    (None, True, False),
    (None, False, False),
],
                         ids=[
                             "none-preempted", "some-preempted", "no-seeds",
                             "no-preempted", "no-policy-views"
                         ])
def test_lrfu_memory_and_the_preemption_sentence(preempted, views, never):
    s = lrfu_summary(PARTS, preempted, views)
    html = nv._lrfu_memory(s, ALL_NEW)
    assert html.startswith('<p class="prose">') and html.endswith("</p>")
    t = text(html)
    assert t.startswith(LRFU_MEMORY)
    assert ("counted. That record stops a preempted request (one the server "
            "pauses when the cache is full and computes again later) from "
            "counting a block twice.") in t
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


WHY_LRFU = (
    "Why Sampled LRFU loses (measured on the engine): of the reuse it loses "
    "against no eviction, 33% was still in the cache behind a missing "
    "block, which the server cannot use; 120–150 requests per seed hit such "
    "stranded blocks. It evicted about as much needed reuse as LRU (2,100 "
    "block uses against 2,100).")
CLEAN = ("LRU, LFU, TTDF and RACE-RP stranded none: TTDF and RACE-RP never "
         "evict a prompt's start before its end, and LRU and LFU never rank "
         "it lower, because it is used whenever the end is and ties go to "
         "the deeper block.")
RANDOM = (" Choosing among only 4 random blocks also means the victim is "
          "often a block used a moment ago.")


def test_q7_three_new_policies():
    s = summary(**THREE)
    facts: Dict[str, Any] = {}
    html = nv.q7(s, rp.build_data(s), facts)
    t = text(html)
    items = verdict_items(html)
    assert items[0].endswith(
        "In the cache replay, the best real policy at each size: TTDF at 32K "
        "to 64K; Sampled LRFU at 128K to 256K. Belady, which knows the "
        "future, is still 10.0 points ahead at 128K (cache replay).")
    assert items[3] == (
        "Sampled LRFU is from 1.0 point below to 5.0 points above LRU across "
        "sizes (cache replay); against the best of LRU, LFU and FIFO it "
        "beats all three at 128K to 256K and trails the best of them at 32K "
        "to 64K. On the engine at 128K it keeps less than LRU of ShareGPT, "
        "AgentBank, CC-Bench and Bailian.")
    # FIFO stranded blocks too, so it is not among the clean policies.
    assert items[4] == f"{WHY_LRFU} {CLEAN}{RANDOM}"
    assert claims(html)[0] == (
        "At 128K inside the mix (cache replay), TTDF loses on all four; "
        "RACE-RP loses on all four; Sampled LRFU gains on all four")
    assert "Sampled LRFU scores only the 4 blocks it draws" in t
    assert ["Sampled LRFU", "0.5 µs", "0.8 µs", "4", "64 KiB"] in \
        table_rows(html)
    assert [
        "Sampled LRFU", "half-life 512 uses, 4 drawn", "62.0%",
        "as specified, picked"
    ] in table_rows(html)
    assert ("For Sampled LRFU the specified setting was also the best on "
            "seed 1") in t
    assert "Bailian: LRU 400/s" in t and "Sampled LRFU 392/s" in t
    assert facts["q7_names"] == "TTDF, RACE-RP and Sampled LRFU"
    assert facts["q7_answer"] == (
        "TTDF keeps the most: on the engine at 128K, 57.0% of the mix's "
        "prompt tokens against LRU's 56.0%. RACE-RP is 2.0 points below LRU "
        "(cache replay). Sampled LRFU as specified keeps the least of all "
        "(45.0% on the engine), mostly because it cuts prompts in the "
        "middle. Best real policy by size (cache replay): TTDF at 32K to "
        "64K; Sampled LRFU at 128K to 256K.")


def lrfu_why(s):
    """The "Why Sampled LRFU loses" bullet, or None."""
    items = nv._verdict(s, s["capacity_window"], ALL_NEW)
    found = [text(x) for x in items if "Why Sampled LRFU loses" in x]
    return found[0] if found else None


def test_why_sampled_lrfu_loses_needs_a_loss_and_stranded_blocks():
    assert lrfu_why(summary(**THREE)) is not None
    # Not lower than LRU on the engine: nothing to explain.
    s = summary(**THREE)
    s["mixed_groups"]["all"]["sampled_lrfu"]["verdict"] = "tie"
    assert lrfu_why(s) is None
    # Nothing cut off behind a missing block.
    s = summary(**THREE)
    for ws in s["lost"]["sampled_lrfu"].values():
        for v in ws.values():
            v["cut_off"] = 0
    assert lrfu_why(s) is None
    # No count of stranded hits.
    s = summary(**THREE)
    del s["policy_views"]["sampled_lrfu"]["hits_after_miss"]
    assert lrfu_why(s) is None
    # No lost reuse for Sampled LRFU at all.
    s = summary(**THREE)
    del s["lost"]["sampled_lrfu"]
    assert lrfu_why(s) is None


def test_why_sampled_lrfu_loses_without_lru_and_with_fewer_clean():
    s = summary(**THREE)
    del s["lost"]["lru"]
    why = lrfu_why(s)
    assert "It evicted about as much" not in why
    # Only LRU and FIFO strand none: LRU's reason alone, verb agreeing.
    s = summary(**THREE)
    views = s["policy_views"]
    for p in ("lfu", "ttdf", "race_rp"):
        views[p]["hits_after_miss"] = [1, 0, 0]
    views["fifo"]["hits_after_miss"] = [0, 0, 0]
    assert ("LRU and FIFO stranded none: LRU never ranks it lower, because "
            "it is used whenever the end is and ties go to the deeper "
            "block.") in lrfu_why(s)
    # Only RACE-RP: its rule alone.
    for p in ("lru", "fifo"):
        views[p]["hits_after_miss"] = [1, 0, 0]
    views["race_rp"]["hits_after_miss"] = [0, 0, 0]
    assert ("RACE-RP stranded none: RACE-RP never evicts a prompt's start "
            "before its end.") in lrfu_why(s)
    # Only FIFO: named without a reason.
    views["race_rp"]["hits_after_miss"] = [2, 0, 0]
    views["fifo"]["hits_after_miss"] = [0, 0, 0]
    assert " FIFO stranded none. Choosing among" in lrfu_why(s)
    # None clean: no "stranded none" at all.
    views["fifo"]["hits_after_miss"] = [1, 1, 1]
    assert "stranded none" not in lrfu_why(s)
    # A policy without the count is not called clean.
    del views["fifo"]["hits_after_miss"]
    assert "stranded none" not in lrfu_why(s)


def test_why_sampled_lrfu_loses_names_the_tuned_pick():
    s = summary(**THREE)
    curve = s["capacity_window"]["curve"]
    tuned = [0.41, 0.52, 0.63, 0.71]  # above LRU at 32K to 256K
    curve["sampled_lrfu_tuned"] = {
        "hit_rate": tuned,
        "seeds": [[v - 0.01, v, v + 0.01] for v in tuned]
    }
    sel = s["new_policies"]["tuning"]["selected"]["sampled_lrfu"]
    sel["label"] = "half-life 64 uses, 4 drawn"
    assert ("The seed-1 pick (half-life 64 uses, 4 drawn) beats LRU at every "
            "size (cache replay).") in lrfu_why(s)
    # A tuned curve with a gap, or no pick label: no such sentence.
    curve["sampled_lrfu_tuned"]["hit_rate"][0] = None
    assert "The seed-1 pick" not in lrfu_why(s)
    curve["sampled_lrfu_tuned"]["hit_rate"][0] = 0.41
    s["new_policies"]["tuning"] = None
    assert "The seed-1 pick" not in lrfu_why(s)


def test_verdict_best_overall_and_its_seed_word():
    s = summary()
    s["mixed_groups"]["all"]["ttdf"]["verdict"] = "tie"
    items = nv._verdict(s, s["capacity_window"], NEW)
    assert "(a tie by the rule, by 1.0 point)" in text(items[0])
    # Without Belady's curve, no Belady sentence.
    del s["capacity_window"]["curve"]["belady"]
    first = text(nv._verdict(s, s["capacity_window"], NEW)[0])
    assert first.startswith("Best overall: TTDF.") and "Belady" not in first


def test_verdict_sharegpt_and_lead_lines_follow_the_engine_verdicts():
    s = summary()
    mg = s["mixed_groups"]
    mg["sharegpt"]["ttdf"]["verdict"] = "tie"
    mg["all"]["race_rp"]["verdict"] = "higher"
    items = [text(x) for x in nv._verdict(s, s["capacity_window"], NEW)]
    assert any(
        x.startswith("Why TTDF and RACE-RP do not help ShareGPT")
        for x in items)
    assert ("Part of the lead of TTDF and RACE-RP may come from evicting only "
            "a prompt's end") in items[-1]
    # Neither helps on ShareGPT's engine runs... or both do: no line.
    mg["sharegpt"]["ttdf"]["verdict"] = "higher"
    mg["sharegpt"]["race_rp"]["verdict"] = "higher"
    mg["all"]["ttdf"]["verdict"] = mg["all"]["race_rp"]["verdict"] = "tie"
    items = [text(x) for x in nv._verdict(s, s["capacity_window"], NEW)]
    assert not any("do not help ShareGPT" in x or "does not help" in x
                   for x in items)
    assert not any(x.startswith("Part of the lead") for x in items)


def test_q7_gains_claim_per_workload():
    s = summary()
    gs = s["capacity_window"]["group_seeds"]
    lru = gs["lru"]
    # At 128K (index 2): TTDF a point above LRU on ShareGPT, level on
    # Bailian, below elsewhere; RACE-RP level with LRU everywhere.
    gs["ttdf"]["sharegpt"][2] = [v + 0.01 for v in lru["sharegpt"][2]]
    gs["ttdf"]["bailian"][2] = list(lru["bailian"][2])
    for g in rp.GROUPS:
        gs["race_rp"][g][2] = list(lru[g][2])
    assert claims(nv.q7(s, rp.build_data(s), {}))[0] == (
        "At 128K inside the mix (cache replay), TTDF gains on ShareGPT and "
        "loses on AgentBank and CC-Bench; RACE-RP ties")


def test_lost_split_reads_all_or_sums_the_workloads():
    s = summary(**THREE)
    # Each seed's "all": 700 evicted, 350 cut off; over three seeds.
    assert nv._lost_split(s, "sampled_lrfu") == (2100, 1050)
    # Without "all", the workloads summed: the same totals.
    for ws in s["lost"]["sampled_lrfu"].values():
        del ws["all"]
    assert nv._lost_split(s, "sampled_lrfu") == (2100, 1050)
    assert nv._lost_split(s, "belady") is None
    del s["lost"]
    assert nv._lost_split(s, "lru") is None


def test_small_helpers():
    assert nv._span_ints([120, 150, 130]) == "120–150"
    assert nv._span_ints([1500, 1500]) == "1,500"
    assert nv._num(None) == float("-inf")
    assert nv._num(None, float("inf")) == float("inf")
    assert nv._num(0.0) == 0.0 and nv._num(0.25, 1.0) == 0.25


def best_base_cv(diffs):
    """LRU, LFU, FIFO and `x` over four sizes, `x` `diffs` above the best."""
    base = {
        "lru": [0.5, 0.5, 0.6, 0.6],
        "lfu": [0.52, 0.5, 0.55, 0.6],
        "fifo": [0.4, 0.4, 0.4, 0.4]
    }
    top = [max(v[k] for v in base.values()) for k in range(4)]
    cv = {
        p: {
            "hit_rate": v,
            "seeds": [[x] * 3 for x in v]
        }
        for p, v in base.items()
    }
    xs = [t + d for t, d in zip(top, diffs)]
    cv["x"] = {"hit_rate": xs, "seeds": [[x] * 3 for x in xs]}
    return cv


@pytest.mark.parametrize("diffs, words", [
    ([0.01] * 4, "beats all three at every size"),
    ([-0.01] * 4, "is below the best of them at every size"),
    ([0.01, 0.0, -0.02, 0.01], "beats all three at 32K and 256K, ties with "
     "the best of them at 64K and trails the best of them at 128K"),
    ([0.0, 0.001, 0.0, 0.0], "ties with the best of them at 32K to 256K"),
])
def test_vs_best_baseline(diffs, words):
    assert nv._vs_best_baseline(CAPS, best_base_cv(diffs), "x") == words


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
    assert "<h2>Three new policies on the shared cache</h2>" in secs[-3]
    assert "answers seven questions" in text(secs[0])
    assert ('<span class="q">7 · TTDF, RACE-RP and Sampled LRFU</span>'
            in secs[1])
    assert whole.count("(and TTDF, RACE-RP and Sampled LRFU)?") == 2
    assert whole.count("<dt>Sampled LRFU</dt>") == 1
    assert "Sampled LRFU's clock counts block uses" in secs[-1]
