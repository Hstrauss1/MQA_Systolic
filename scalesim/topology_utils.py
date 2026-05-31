"""
This file contains the 'topologies' class that handles the topology files fed to SCALE_Sim tool.
"""

import math


class topologies(object):
    """
    Class which contains the methods to preprocess the data from topology file (.csv format) before
    doing compute simulation.
    """
    #
    def __init__(self):
        """
        __init__ method
        """
        self.current_topo_name = ""
        self.topo_file_name = ""
        self.topo_arrays = []
        self.spatio_temp_dim_arrays = []
        self.layers_calculated_hyperparams = []
        self.num_layers = 0
        self.topo_load_flag = False
        self.topo_calc_hyper_param_flag = False
        self.topo_calc_spatiotemp_params_flag = False
        self.df = ""
        self.current_toponame = ""
        self.layer_name = ""
        self.workload_type = 'gemm'
        self.mqa_metadata = {}
        self.mqa_stage_metadata = []

    def reset(self):
        print("All data reset")
        self.current_topo_name = ""
        self.topo_file_name = ""
        self.topo_load_flag = False
        self.topo_arrays = []
        self.num_layers = 0
        self.topo_calc_hyper_param_flag = False
        self.layers_calculated_hyperparams = []
        self.df = ""
        self.current_toponame = ""
        self.layer_name = ""
        self.spatio_temp_dim_arrays = []
        self.topo_calc_spatiotemp_params_flag = False
        self.workload_type = 'gemm'
        self.mqa_metadata = {}
        self.mqa_stage_metadata = []

    #
    def load_layer_params_from_list(self, layer_name, elems_list=[]):
        self.topo_file_name = ''
        self.current_toponame = ''
        self.layer_name = layer_name
        self.append_topo_arrays(layer_name, elems_list)

        self.num_layers += 1
        self.topo_load_flag = True

    #
    def load_arrays(self, topofile='', mnk_inputs=False, config_obj=None):
        if config_obj is not None and hasattr(config_obj, 'is_mqa_workload') and config_obj.is_mqa_workload():
            self.load_arrays_mqa(config_obj)
            return

        if mnk_inputs:
            self.load_arrays_gemm(topofile)
        else:
            self.load_arrays_conv(topofile)

    def load_arrays_mqa(self, config_obj):
        self.reset()
        self.workload_type = config_obj.get_workload_type()
        self.mqa_metadata = dict(config_obj.get_mqa_params())
        self.topo_file_name = ''
        self.current_topo_name = self.workload_type

        params = self.mqa_metadata
        seq_len = int(params['sequence_length'])
        batch_size = int(params['batch_size'])
        query_heads = int(params['query_heads'])
        kv_heads = int(params['kv_heads'])
        head_dim = int(params['head_dim'])
        decode_tokens = int(params['decode_tokens'])

        score_m = batch_size * query_heads * decode_tokens
        score_n = seq_len
        score_k = head_dim

        value_m = batch_size * query_heads * decode_tokens
        value_n = head_dim
        value_k = seq_len

        pseudo_layers = [
            {
                'name': 'score_stage',
                'mnk': (score_m, score_n, score_k),
                'meta': dict(params),
            },
            {
                'name': 'softmax_reduce',
                'mnk': (score_m, 1, score_n),
                'meta': dict(params),
            },
            {
                'name': 'value_stage',
                'mnk': (value_m, value_n, value_k),
                'meta': dict(params),
            },
            {
                'name': 'writeback',
                'mnk': (value_m, value_n, 1),
                'meta': dict(params),
            },
        ]

        self.mqa_stage_metadata = pseudo_layers

        for stage in pseudo_layers:
            layer_name = stage['name']
            m, n, k = stage['mnk']
            entries = [
                layer_name,
                int(m),
                int(k),
                1,
                int(k),
                1,
                int(n),
                1,
                1,
                1,
                1,
            ]
            self.append_topo_arrays(layer_name=layer_name, elems=entries)

        self.num_layers = len(self.topo_arrays)
        self.topo_load_flag = True
        self.topo_calc_hyperparams()

    #
    def load_arrays_gemm(self, topofile=''):
        self.topo_file_name = topofile.split('/')[-1]
        name_arr = self.topo_file_name.split('.')
        if len(name_arr) > 1:
            self.current_topo_name = self.topo_file_name.split('.')[-2]
        else:
            self.current_topo_name = self.topo_file_name

        f = open(topofile, 'r')
        first = True

        for row in f:
            row = row.strip()
            if first:
                first = False
                continue
            elif row == '':
                continue
            else:
                elems = row.split(',')[:-1]
                assert len(elems) > 3, 'There should be at least 4 entries per row'
                layer_name = elems[0].strip()
                m = elems[1].strip()
                n = elems[2].strip()
                k = elems[3].strip()
                if len(elems) < 5:
                    elems.append("1:1")
                sparsity_ratio = elems[4].strip().split(':')

                entries = [layer_name, m, k, 1, k, 1, n, 1, 1, sparsity_ratio[0], sparsity_ratio[1]]
                self.append_topo_arrays(layer_name=layer_name, elems=entries)

        self.num_layers = len(self.topo_arrays)
        self.topo_load_flag = True

    def load_arrays_conv(self, topofile=''):
        first = True
        self.topo_file_name = topofile.split('/')[-1]
        name_arr = self.topo_file_name.split('.')
        if len(name_arr) > 1:
            self.current_topo_name = self.topo_file_name.split('.')[-2]
        else:
            self.current_topo_name = self.topo_file_name

        f = open(topofile, 'r')
        for row in f:
            row = row.strip()
            if first or row == '':
                first = False
            else:
                elems = row.split(',')[:-1]
                elems = elems[0:8] + [elems[7]] + elems[8:]

                if len(elems) < 10:
                    elems = elems + ['1', '1']

                self.append_topo_arrays(layer_name=elems[0], elems=elems)

        self.num_layers = len(self.topo_arrays)
        self.topo_load_flag = True

    def append_topo_arrays(self, layer_name="", elems=None):
        if elems is None:
            elems = []
        entry = [layer_name]
        for e in elems[1:]:
            entry.append(int(e))
        self.current_toponame = layer_name
        self.current_topo_name = layer_name if self.current_topo_name == "" else self.current_topo_name

        self.topo_arrays.append(entry)
        self.topo_load_flag = True
        self.topo_calc_hyperparams()
        self.num_layers = len(self.topo_arrays)

    def topo_calc_hyperparams(self, topofilename=""):
        if not self.topo_load_flag and topofilename:
            self.load_arrays(topofilename)
        self.layers_calculated_hyperparams = []
        for array in self.topo_arrays:
            ifmap_h = array[1]
            ifmap_w = array[2]
            filt_h = array[3]
            filt_w = array[4]
            num_ch = array[5]
            num_filt = array[6]
            stride_h = array[7]
            stride_w = array[8]
            ofmap_h = int(math.ceil((ifmap_h - filt_h + stride_h) / stride_h))
            ofmap_w = int(math.ceil((ifmap_w - filt_w + stride_w) / stride_w))
            num_mac = ofmap_h * ofmap_w * filt_h * filt_w * num_ch * num_filt
            window_size = filt_h * filt_w * num_ch
            entry = [ofmap_h, ofmap_w, num_mac, window_size]
            self.layers_calculated_hyperparams.append(entry)
        self.topo_calc_hyper_param_flag = True

    #
    def calc_spatio_temporal_params(self, df='os', layer_id=0):
        s_row = -1
        s_col = -1
        t_time = -1
        if self.topo_calc_hyper_param_flag:
            num_filt = self.get_layer_num_filters(layer_id=layer_id)
            num_ofmap = self.get_layer_num_ofmap_px(layer_id=layer_id)
            num_ofmap = int(num_ofmap / num_filt)
            window_sz = self.get_layer_window_size(layer_id=layer_id)
            if df == 'os':
                s_row = num_ofmap
                s_col = num_filt
                t_time = window_sz
            elif df == 'ws':
                s_row = window_sz
                s_col = num_filt
                t_time = num_ofmap
            elif df == 'is':
                s_row = window_sz
                s_col = num_ofmap
                t_time = num_filt
        else:
            self.topo_calc_hyperparams(self.topo_file_name)
        return s_row, s_col, t_time

    #
    def set_spatio_temporal_params(self):
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams(self.topo_file_name)
        self.spatio_temp_dim_arrays = []
        for i in range(self.num_layers):
            this_layer_params_arr = []
            for df in ['os', 'ws', 'is']:
                sr, sc, tt = self.calc_spatio_temporal_params(df=df, layer_id=i)
                this_layer_params_arr.append([sr, sc, tt])
            self.spatio_temp_dim_arrays.append(this_layer_params_arr)
        self.topo_calc_spatiotemp_params_flag = True

    #
    def get_transformed_mnk_dimensions(self):
        if not self.topo_calc_hyper_param_flag:
            self.topo_calc_hyperparams(self.topo_file_name)

        mnk_dims_arr = []
        for i in range(self.num_layers):
            M = self.get_layer_num_ofmap_px(layer_id=i)
            N = self.get_layer_num_filters(layer_id=i)
            K = self.get_layer_window_size(layer_id=i)
            mnk_dims_arr.append([M, N, K])

        return mnk_dims_arr

    #
    def get_current_topo_name(self):
        current_topo_name = ""
        if self.topo_load_flag:
            current_topo_name = self.current_topo_name
        else:
            print('Error: get_current_topo_name(): Topo file not read')
        return current_topo_name

    #
    def get_num_layers(self):
        if not self.topo_load_flag:
            print("ERROR: topologies.get_num_layers: No array loaded")
            return
        return self.num_layers

    #
    def get_layer_ifmap_dims(self, layer_id=0):
        if not (self.topo_load_flag or self.num_layers - 1 < layer_id):
            print("ERROR: topologies.get_layer_ifmap_dims: Invalid layer id")

        layer_params = self.topo_arrays[layer_id]
        return layer_params[1:3]

    def get_layer_filter_dims(self, layer_id=0):
        layer_params = self.topo_arrays[layer_id]
        return layer_params[3:5]

    def get_layer_num_channels(self, layer_id=0):
        layer_params = self.topo_arrays[layer_id]
        return layer_params[5]

    def get_layer_num_filters(self, layer_id=0):
        layer_params = self.topo_arrays[layer_id]
        return layer_params[6]

    def get_layer_strides(self, layer_id=0):
        layer_params = self.topo_arrays[layer_id]
        return layer_params[7:9]

    def get_layer_name(self, layer_id=0):
        layer_params = self.topo_arrays[layer_id]
        return layer_params[0]

    def get_layer_num_ofmap_px(self, layer_id=0):
        return self.layers_calculated_hyperparams[layer_id][0] * self.layers_calculated_hyperparams[layer_id][1] * \
               self.get_layer_num_filters(layer_id=layer_id)

    def get_layer_window_size(self, layer_id=0):
        return self.layers_calculated_hyperparams[layer_id][3]

    def get_spatiotemporal_dims(self, layer_id=0, df='os'):
        if not self.topo_calc_spatiotemp_params_flag:
            self.set_spatio_temporal_params()

        df_idx = {'os': 0, 'ws': 1, 'is': 2}[df]
        return self.spatio_temp_dim_arrays[layer_id][df_idx]

    def get_workload_type(self):
        return self.workload_type

    def is_mqa_workload(self):
        return self.workload_type != 'gemm'

    def get_mqa_stage_metadata(self):
        return self.mqa_stage_metadata

    def get_mqa_metadata(self):
        return self.mqa_metadata
