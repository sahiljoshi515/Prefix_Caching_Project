"""The results page's computed wording (spec 007 SC-003): every judgement
follows the numbers, including the cases the real data does not show."""
import pytest

from analysis import m3t2_narrative as nv

pytestmark = pytest.mark.m3


def bins(shares, offered=5000, edges=(0, 1, 10, 100, 1000)):
    bounds = list(edges) + [None]
    return [{
        "lo": lo,
        "hi": hi,
        "offered": offered,
        "share": x,
        "served": int(x * offered)
    } for lo, hi, x in zip(bounds[:-1], bounds[1:], shares)]


def test_cutoff_is_none_when_a_policy_keeps_serving():
    # Served shares fall below 1% from the 10-100 s bin on: the cutoff is
    # that bin's lower edge, 10 s.
    assert nv._cutoff(bins([1.0, 0.9, 0.005, 0.0, 0.0])) == 10
    # Some reuse at every wait: no cutoff (the page then says so).
    assert nv._cutoff(bins([1.0, 0.9, 0.5, 0.2, 0.1])) is None
    # A bin with too few reuses on offer is not drawn, so it does not count.
    sparse = bins([1.0, 0.9, 0.005, 0.0, 0.5])
    sparse[-1]["offered"] = 10
    assert nv._cutoff(sparse) == 10
    assert nv._served_beyond(bins([1.0, 0.5, 0.2, 0.0, 0.0]), 10) == \
        pytest.approx(0.2 / 3)


def test_verdicts_against_lru_read_three_ways():
    assert nv._against_lru({"lfu": "lower", "fifo": "lower"}) == \
        "LFU and FIFO are lower than LRU"
    assert nv._against_lru({"lfu": "higher", "fifo": "tie"}) == \
        "LFU beats LRU; FIFO ties with it"
    assert nv._against_lru({"lfu": "tie", "fifo": "tie"}) == \
        "LFU and FIFO tie with LRU"


def test_best_policy_follows_the_rule_not_the_largest_mean():

    def vs(lfu, fifo):
        return {"lfu": {"verdict": lfu}, "fifo": {"verdict": fifo}}

    assert nv._best_text(vs("tie", "tie")) == "all three tie"
    assert nv._best_text(vs("lower", "lower")) == "LRU is best"
    assert nv._best_text(vs("tie", "lower")) == "LRU and LFU tie for best"
    assert nv._best_text(vs("higher", "lower")) == "LFU beats LRU"


def test_span_and_lists():
    assert nv._span(0.042, 0.05) == "4.2 to 5.0 points"
    assert nv._span(0.05, 0.05) == "5.0 points"
    assert nv._list(["a"]) == "a" and nv._list(["a", "b", "c"]) == \
        "a, b and c"
