from mqa_scalesim.validation_bridge import validate_against_reference


def test_validation_bridge_reports_missing_keys():
    summary = validate_against_reference({'a': 1}, {'a': 1, 'b': 2})
    assert not summary.passed
    assert 'b' in summary.metadata['missing_keys']
