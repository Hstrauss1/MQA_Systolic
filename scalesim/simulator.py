"""
This file contains the 'simulator' class that simulates the entire model using the class
'single_layer_sim' and generates the reports (.csv files).
"""

import os

from scalesim.scale_config import scale_config as cfg
from scalesim.topology_utils import topologies as topo
from scalesim.layout_utils import layouts as layout
from scalesim.single_layer_sim import single_layer_sim as layer_sim
from scalesim.linear_model.tpu import tpuv4_linear_model, tpuv5e_linear_model, tpuv6e_linear_model

try:
    from mqa_scalesim.workload import MQAWorkload
    from mqa_scalesim.baseline_decode import BaselineMQADecodeSimulator
    from mqa_scalesim.kv_stationary_decode import KVStationaryMQADecodeSimulator
except ImportError:
    MQAWorkload = None
    BaselineMQADecodeSimulator = None
    KVStationaryMQADecodeSimulator = None


class simulator:
    """
    Class which runs the simulations and manages generated data across various layers
    """
    #
    def __init__(self):
        self.conf = cfg()
        self.topo = topo()
        self.layout = layout()

        self.top_path = "./"
        self.verbose = True
        self.save_trace = True

        self.num_layers = 0

        self.single_layer_sim_object_list = []

        self.params_set_flag = False
        self.all_layer_run_done = False

        self.mqa_result = None
        self.mqa_workload = None

    def _build_mqa_workload(self):
        if MQAWorkload is None:
            raise ImportError("mqa_scalesim package is not available for MQA workload execution")

        params = self.conf.get_mqa_params()
        return MQAWorkload(
            mode=params['workload_type'],
            sequence_length=params['sequence_length'],
            batch_size=params['batch_size'],
            query_heads=params['query_heads'],
            kv_heads=params['kv_heads'],
            head_dim=params['head_dim'],
            precision=params['precision'],
            array_rows=self.conf.array_rows,
            array_cols=self.conf.array_cols,
            ifmap_sram_kb=self.conf.ifmap_sz_kb,
            filter_sram_kb=self.conf.filter_sz_kb,
            ofmap_sram_kb=self.conf.ofmap_sz_kb,
            bandwidth_mode='user' if self.conf.use_user_bandwidth else 'calc',
            dram_bandwidth=min(self.conf.bandwidths) if self.conf.bandwidths else None,
            decode_tokens=params['decode_tokens'],
            decode_step=params['decode_step'],
            softmax_variant=params['softmax_variant'],
            exp_variant=params['exp_variant'],
        )

    def _run_mqa_path(self):
        workload_type = self.conf.get_workload_type()
        self.mqa_workload = self._build_mqa_workload()
        self.mqa_workload.validate()

        if workload_type == 'baseline_mqa_decode':
            if BaselineMQADecodeSimulator is None:
                raise ImportError("BaselineMQADecodeSimulator is unavailable")
            runner = BaselineMQADecodeSimulator(self.mqa_workload)
        elif workload_type == 'kv_stationary_mqa_decode':
            if KVStationaryMQADecodeSimulator is None:
                raise ImportError("KVStationaryMQADecodeSimulator is unavailable")
            runner = KVStationaryMQADecodeSimulator(self.mqa_workload)
        else:
            raise ValueError("Unsupported MQA workload type '{}'".format(workload_type))

        if not os.path.isdir(self.top_path):
            os.mkdir(self.top_path)

        report_path = self.top_path + '/' + self.conf.get_run_name()
        if not os.path.isdir(report_path):
            os.mkdir(report_path)

        self.top_path = report_path
        self.mqa_result = runner.simulate()
        self.all_layer_run_done = True
        return self.mqa_result

    #
    def set_params(self,
                   config_obj=cfg(),
                   topo_obj=topo(),
                   layout_obj=layout(),
                   top_path='.',
                   verbosity=True,
                   save_trace=True
                   ):
        self.conf = config_obj
        self.topo = topo_obj
        self.layout = layout_obj
        self.top_path = top_path
        self.verbose = verbosity
        self.save_trace = save_trace

        if self.conf.is_mqa_workload():
            self.topo.load_arrays(config_obj=self.conf)
            self.num_layers = self.topo.get_num_layers()
        else:
            self.num_layers = self.topo.get_num_layers()

        self.params_set_flag = True

    #
    def run(self):
        if not self.params_set_flag:
            print('ERROR: simulator.run(): Parameters are not set')
            return

        if self.conf.get_workload_type() == 'gemm' or not self.conf.is_mqa_workload():
            self.run_legacy()
            return

        self._run_mqa_path()

    def run_legacy(self):
        if not os.path.isdir(self.top_path):
            os.mkdir(self.top_path)

        report_path = self.top_path + '/' + self.conf.get_run_name()
        if not os.path.isdir(report_path):
            os.mkdir(report_path)

        self.top_path = report_path

        if self.conf.get_layout_path() != '':
            self.layout.load_data(self.conf.get_layout_path())

        for layer_id in range(self.num_layers):
            layer_sim_obj = layer_sim()
            layer_sim_obj.set_params(
                layer_id=layer_id,
                config_obj=self.conf,
                topology_obj=self.topo,
                layout_obj=self.layout,
                verbose=self.verbose,
                save_trace=self.save_trace
            )
            layer_sim_obj.run()
            self.single_layer_sim_object_list.append(layer_sim_obj)

        if self.conf.get_time_linear_model() == 'TPUv4':
            linear_model = tpuv4_linear_model()
            linear_model.apply(self.single_layer_sim_object_list)
        elif self.conf.get_time_linear_model() == 'TPUv5e':
            linear_model = tpuv5e_linear_model()
            linear_model.apply(self.single_layer_sim_object_list)
        elif self.conf.get_time_linear_model() == 'TPUv6e':
            linear_model = tpuv6e_linear_model()
            linear_model.apply(self.single_layer_sim_object_list)

        self.generate_reports()
        self.all_layer_run_done = True

    def generate_reports(self):
        if not self.all_layer_run_done and len(self.single_layer_sim_object_list) == 0:
            return

        compute_report = open(self.top_path + '/COMPUTE_REPORT.csv', 'w')
        bandwidth_report = open(self.top_path + '/BANDWIDTH_REPORT.csv', 'w')
        detail_report = open(self.top_path + '/DETAILED_ACCESS_REPORT.csv', 'w')

        compute_report.write('LayerID, Total Cycles, Stall Cycles, Overall Util %, Mapping Efficiency %, Compute Util %\n')
        bandwidth_report.write('LayerID, Avg IFMAP SRAM BW, Avg FILTER SRAM BW, Avg OFMAP SRAM BW, Avg IFMAP DRAM BW, Avg FILTER DRAM BW, Avg OFMAP DRAM BW\n')
        detail_report.write('LayerID, IFMAP SRAM Start Cycle, IFMAP SRAM Stop Cycle, FILTER SRAM Start Cycle, FILTER SRAM Stop Cycle, OFMAP SRAM Start Cycle, OFMAP SRAM Stop Cycle, IFMAP DRAM Start Cycle, IFMAP DRAM Stop Cycle, FILTER DRAM Start Cycle, FILTER DRAM Stop Cycle, OFMAP DRAM Start Cycle, OFMAP DRAM Stop Cycle\n')

        for lid, sim_obj in enumerate(self.single_layer_sim_object_list):
            total_cycles = sim_obj.get_total_cycles()
            stall_cycles = sim_obj.get_stall_cycles()
            overall_util = sim_obj.get_overall_util_perc()
            mapping_eff = sim_obj.get_mapping_efficiency()
            compute_util = sim_obj.get_compute_utilization()
            compute_report.write('{},{},{},{},{},{}\n'.format(
                lid, total_cycles, stall_cycles, overall_util, mapping_eff, compute_util
            ))

            avg_ifmap_sram_bw = sim_obj.get_avg_ifmap_sram_bw()
            avg_filter_sram_bw = sim_obj.get_avg_filter_sram_bw()
            avg_ofmap_sram_bw = sim_obj.get_avg_ofmap_sram_bw()
            avg_ifmap_dram_bw = sim_obj.get_avg_ifmap_dram_bw()
            avg_filter_dram_bw = sim_obj.get_avg_filter_dram_bw()
            avg_ofmap_dram_bw = sim_obj.get_avg_ofmap_dram_bw()
            bandwidth_report.write('{},{},{},{},{},{},{}\n'.format(
                lid,
                avg_ifmap_sram_bw,
                avg_filter_sram_bw,
                avg_ofmap_sram_bw,
                avg_ifmap_dram_bw,
                avg_filter_dram_bw,
                avg_ofmap_dram_bw
            ))

            ifmap_sram_start, ifmap_sram_stop = sim_obj.get_ifmap_sram_start_stop_cycles()
            filter_sram_start, filter_sram_stop = sim_obj.get_filter_sram_start_stop_cycles()
            ofmap_sram_start, ofmap_sram_stop = sim_obj.get_ofmap_sram_start_stop_cycles()
            ifmap_dram_start, ifmap_dram_stop = sim_obj.get_ifmap_dram_start_stop_cycles()
            filter_dram_start, filter_dram_stop = sim_obj.get_filter_dram_start_stop_cycles()
            ofmap_dram_start, ofmap_dram_stop = sim_obj.get_ofmap_dram_start_stop_cycles()
            detail_report.write('{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(
                lid,
                ifmap_sram_start, ifmap_sram_stop,
                filter_sram_start, filter_sram_stop,
                ofmap_sram_start, ofmap_sram_stop,
                ifmap_dram_start, ifmap_dram_stop,
                filter_dram_start, filter_dram_stop,
                ofmap_dram_start, ofmap_dram_stop
            ))

        compute_report.close()
        bandwidth_report.close()
        detail_report.close()
