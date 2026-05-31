from mqa_scalesim.pe_timing import PETiming
from mqa_scalesim.softmax_ops import estimate_online_softmax_cost


def test_softmax_cost_is_positive():
    timing = PETiming()
    cost = estimate_online_softmax_cost(tokens=8, timing=timing)
    assert cost.total_cycles > 0
