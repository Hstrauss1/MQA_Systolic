from mqa_scalesim.workload import MQAWorkload


def test_workload_validate_accepts_valid_input():
    workload = MQAWorkload(
        mode='baseline_mqa_decode',
        sequence_length=128,
        batch_size=1,
        query_heads=8,
        kv_heads=1,
        head_dim=64,
    )
    workload.validate()
    assert workload.kv_sharing_ratio == 8
