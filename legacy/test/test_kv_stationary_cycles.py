from mqa_scalesim.kv_stationary_decode import KVStationaryMQADecodeSimulator
from mqa_scalesim.workload import MQAWorkload


def test_kv_stationary_simulate_creates_expected_stage_names():
    workload = MQAWorkload(
        mode='kv_stationary_mqa_decode',
        sequence_length=16,
        batch_size=1,
        query_heads=8,
        kv_heads=1,
        head_dim=32,
    )
    sim = KVStationaryMQADecodeSimulator(workload)
    result = sim.simulate()
    assert [stage.name for stage in result.stages] == ['kv_preload', 'query_stream', 'online_softmax_accum', 'final_normalize', 'writeback']
