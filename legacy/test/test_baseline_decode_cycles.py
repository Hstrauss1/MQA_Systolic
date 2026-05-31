from mqa_scalesim.baseline_decode import BaselineMQADecodeSimulator
from mqa_scalesim.workload import MQAWorkload


def test_baseline_simulate_creates_expected_stage_names():
    workload = MQAWorkload(
        mode='baseline_mqa_decode',
        sequence_length=16,
        batch_size=1,
        query_heads=8,
        kv_heads=1,
        head_dim=32,
    )
    sim = BaselineMQADecodeSimulator(workload)
    result = sim.simulate()
    assert [stage.name for stage in result.stages] == ['score_gemm', 'softmax_reduce', 'value_gemm', 'writeback']
