#!/usr/bin/env python3
"""
Focused Phase 2 validation for SCALE-Sim MQA control-plane routing.

This script validates:
1. Legacy config round-trip behavior remains intact.
2. Legacy GEMM topology parsing still works.
3. MQA config ingestion creates staged pseudo-layers.
4. Simulator parameter binding works for legacy and MQA paths.
5. New workload routes dispatch correctly to the Phase 1 MQA scaffold simulators.

It avoids modifying compute/memory backends and monkeypatches the MQA runner classes
for route-validation so Phase 2 can be tested independently.
"""

from __future__ import annotations

import json
import shutil
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class ValidationFailure(Exception):
    pass


def ensure_imports():
    try:
        from scalesim.scale_config import scale_config  # noqa: F401
        from scalesim.topology_utils import topologies  # noqa: F401
        from scalesim.simulator import simulator  # noqa: F401
    except ModuleNotFoundError as exc:
        missing = getattr(exc, 'name', 'unknown dependency')
        raise ValidationFailure(
            "Missing Python dependency: {}. Install repository dependencies first. "
            "Most commonly this means running: python3 -m pip install numpy".format(missing)
        ) from exc
    except Exception as exc:
        raise ValidationFailure("Import failure: {}".format(repr(exc))) from exc


ensure_imports()

from scalesim.scale_config import scale_config
from scalesim.topology_utils import topologies
from scalesim.simulator import simulator
import scalesim.simulator as sim_mod


ARTIFACT_DIR = REPO_ROOT / "phase2_validation_artifacts"


def clean_artifacts():
    if ARTIFACT_DIR.exists():
        shutil.rmtree(ARTIFACT_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


class ResultCollector:
    def __init__(self):
        self.results = []

    def record(self, name, passed, details=None, error=None):
        entry = {
            "name": name,
            "status": "PASS" if passed else "FAIL",
        }
        if details is not None:
            entry["details"] = details
        if error is not None:
            entry["error"] = error
        self.results.append(entry)

    def as_dict(self):
        total = len(self.results)
        passed = sum(1 for r in self.results if r["status"] == "PASS")
        failed = total - passed
        return {
            "summary": {
                "total": total,
                "passed": passed,
                "failed": failed,
                "artifact_dir": str(ARTIFACT_DIR),
            },
            "results": self.results,
        }


collector = ResultCollector()


def run_check(name, fn):
    try:
        details = fn()
        collector.record(name, True, details=details)
    except Exception as exc:
        collector.record(name, False, error={
            "message": str(exc),
            "type": exc.__class__.__name__,
            "traceback": traceback.format_exc(),
        })


def make_legacy_topology_csv(path: Path):
    path.write_text(
        "Layer, M, N, K,\n"
        "fc1,64,32,16,\n",
        encoding="utf-8",
    )
    return path


def build_mqa_conf(mode: str,
                   sequence_length: int = 128,
                   batch_size: int = 1,
                   query_heads: int = 8,
                   kv_heads: int = 2,
                   head_dim: int = 64,
                   decode_tokens: int = 1,
                   decode_step: int = 5):
    conf = scale_config()
    values = scale_config.get_default_conf_as_list()
    values[0] = "phase2_{}".format(mode)
    values[14] = mode
    values[15] = str(sequence_length)
    values[16] = str(batch_size)
    values[17] = str(query_heads)
    values[18] = str(kv_heads)
    values[19] = str(head_dim)
    values[20] = "int8"
    values[21] = str(decode_tokens)
    values[22] = str(decode_step)
    values[23] = "online"
    values[24] = "lookup"
    conf.update_from_list(values)
    return conf


def check_legacy_config_roundtrip():
    conf = scale_config()
    conf.force_valid()
    baseline_list = conf.get_conf_as_list()

    clone = scale_config()
    clone.update_from_list(baseline_list)

    if clone.get_workload_type() != "gemm":
        raise ValidationFailure("Legacy workload_type changed from gemm")
    if clone.is_mqa_workload():
        raise ValidationFailure("Legacy config incorrectly reports MQA workload")

    params = clone.get_mqa_params()
    if params["precision"] != "int8":
        raise ValidationFailure("Expected default precision=int8, got {}".format(params["precision"]))

    return {
        "conf_list_length": len(baseline_list),
        "workload_type": clone.get_workload_type(),
        "is_mqa": clone.is_mqa_workload(),
    }


def check_legacy_topology_path():
    topo_path = make_legacy_topology_csv(ARTIFACT_DIR / "legacy_topology.csv")
    topology = topologies()
    topology.load_arrays(str(topo_path), mnk_inputs=True)

    if topology.get_num_layers() != 1:
        raise ValidationFailure("Expected exactly 1 legacy layer")
    if topology.get_layer_name(0) != "fc1":
        raise ValidationFailure("Unexpected legacy layer name: {}".format(topology.get_layer_name(0)))
    if topology.get_workload_type() != "gemm":
        raise ValidationFailure("Legacy topology should remain gemm")

    return {
        "topology_file": str(topo_path),
        "layer_name": topology.get_layer_name(0),
        "num_layers": topology.get_num_layers(),
    }


def check_mqa_baseline_topology_staging():
    conf = build_mqa_conf("baseline_mqa_decode", sequence_length=128, batch_size=2, query_heads=8, kv_heads=2, head_dim=64)
    topology = topologies()
    topology.load_arrays(config_obj=conf)

    names = [topology.get_layer_name(i) for i in range(topology.get_num_layers())]
    expected = ["score_stage", "softmax_reduce", "value_stage", "writeback"]
    if names != expected:
        raise ValidationFailure("Unexpected staged layers: {}".format(names))

    return {
        "workload_type": topology.get_workload_type(),
        "stages": names,
        "mqa_metadata": topology.get_mqa_metadata(),
    }


def check_mqa_kv_topology_staging():
    conf = build_mqa_conf("kv_stationary_mqa_decode", sequence_length=256, batch_size=1, query_heads=16, kv_heads=4, head_dim=128, decode_tokens=2, decode_step=9)
    topology = topologies()
    topology.load_arrays(config_obj=conf)

    stages = topology.get_mqa_stage_metadata()
    if len(stages) != 4:
        raise ValidationFailure("Expected 4 pseudo-stages, got {}".format(len(stages)))
    if topology.get_workload_type() != "kv_stationary_mqa_decode":
        raise ValidationFailure("Incorrect workload type propagated into topology")

    return {
        "workload_type": topology.get_workload_type(),
        "stage_count": len(stages),
        "stage_names": [stage["name"] for stage in stages],
    }


def check_legacy_simulator_binding():
    conf = scale_config()
    conf.force_valid()
    conf.run_name = "phase2_legacy_binding"

    topo_path = make_legacy_topology_csv(ARTIFACT_DIR / "legacy_sim_bind_topology.csv")
    conf.topofile = str(topo_path)

    topology = topologies()
    topology.load_arrays(str(topo_path), mnk_inputs=True)

    sim = simulator()
    sim.set_params(
        config_obj=conf,
        topo_obj=topology,
        top_path=str(ARTIFACT_DIR / "reports"),
        verbosity=False,
        save_trace=False,
    )

    if sim.num_layers != 1:
        raise ValidationFailure("Expected simulator.num_layers == 1 for legacy workload")
    if sim.conf.get_workload_type() != "gemm":
        raise ValidationFailure("Legacy simulator binding changed workload type")

    return {
        "num_layers": sim.num_layers,
        "top_path": sim.top_path,
    }


def check_mqa_route_dispatch():
    original_workload = sim_mod.MQAWorkload
    original_baseline = sim_mod.BaselineMQADecodeSimulator
    original_kv = sim_mod.KVStationaryMQADecodeSimulator

    class DummyWorkload:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.validated = False

        def validate(self):
            self.validated = True
            return True

    class DummyBaseline:
        def __init__(self, workload):
            self.workload = workload

        def simulate(self):
            return {
                "route": "baseline",
                "mode": self.workload.kwargs["mode"],
                "validated": self.workload.validated,
            }

    class DummyKV:
        def __init__(self, workload):
            self.workload = workload

        def simulate(self):
            return {
                "route": "kv_stationary",
                "mode": self.workload.kwargs["mode"],
                "validated": self.workload.validated,
            }

    sim_mod.MQAWorkload = DummyWorkload
    sim_mod.BaselineMQADecodeSimulator = DummyBaseline
    sim_mod.KVStationaryMQADecodeSimulator = DummyKV

    try:
        route_results = {}
        for mode in ["baseline_mqa_decode", "kv_stationary_mqa_decode"]:
            conf = build_mqa_conf(mode, sequence_length=64, batch_size=1, query_heads=4, kv_heads=1, head_dim=32, decode_tokens=1, decode_step=3)
            topology = topologies()
            sim = simulator()
            sim.set_params(
                config_obj=conf,
                topo_obj=topology,
                top_path=str(ARTIFACT_DIR / "reports"),
                verbosity=False,
                save_trace=False,
            )
            sim.run()
            route_results[mode] = sim.mqa_result

        if route_results["baseline_mqa_decode"]["route"] != "baseline":
            raise ValidationFailure("baseline_mqa_decode did not route to baseline scaffold")
        if route_results["kv_stationary_mqa_decode"]["route"] != "kv_stationary":
            raise ValidationFailure("kv_stationary_mqa_decode did not route to KV scaffold")

        return route_results
    finally:
        sim_mod.MQAWorkload = original_workload
        sim_mod.BaselineMQADecodeSimulator = original_baseline
        sim_mod.KVStationaryMQADecodeSimulator = original_kv


def check_invalid_mqa_validation():
    conf = scale_config()
    values = scale_config.get_default_conf_as_list()
    values[14] = "baseline_mqa_decode"
    values[15] = "0"
    values[16] = "1"
    values[17] = "8"
    values[18] = "2"
    values[19] = "64"
    values[21] = "1"

    try:
        conf.update_from_list(values)
    except ValueError as exc:
        return {"caught": True, "message": str(exc)}

    raise ValidationFailure("Invalid MQA configuration should have raised ValueError")


def write_report(report_dict):
    report_path = ARTIFACT_DIR / "phase2_validation_report.json"
    report_path.write_text(json.dumps(report_dict, indent=2, sort_keys=True), encoding="utf-8")
    return report_path


def main():
    clean_artifacts()

    run_check("legacy_config_roundtrip", check_legacy_config_roundtrip)
    run_check("legacy_topology_path", check_legacy_topology_path)
    run_check("mqa_baseline_topology_staging", check_mqa_baseline_topology_staging)
    run_check("mqa_kv_topology_staging", check_mqa_kv_topology_staging)
    run_check("legacy_simulator_binding", check_legacy_simulator_binding)
    run_check("mqa_route_dispatch", check_mqa_route_dispatch)
    run_check("invalid_mqa_validation", check_invalid_mqa_validation)

    report = collector.as_dict()
    report_path = write_report(report)

    print(json.dumps(report, indent=2, sort_keys=True))
    print("\nValidation report written to: {}".format(report_path))

    failed = report["summary"]["failed"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
