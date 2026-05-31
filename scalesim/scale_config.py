"""
This file defines the 'scale_config' class responsible for all the configuration file related
activities such as parsing the config file, writing the parameters into a config file, updating the
parameters.
"""
import configparser as cp


class scale_config:
    """
    Class that handles the SCALE-Sim configuration files.
    """
    #
    def __init__(self):
        """
        __init__ method
        """
        self.run_name = "scale_run"
        # Anand: ISSUE #2. Patch
        self.use_user_bandwidth = False

        self.array_rows = 4
        self.array_cols = 4
        self.ifmap_sz_kb = 256
        self.filter_sz_kb = 256
        self.ofmap_sz_kb = 128
        self.df = 'ws'
        self.ifmap_offset = 0
        self.filter_offset = 10000000
        self.ofmap_offset = 20000000
        self.req_buf_sz_rd = 60
        self.req_buf_sz_wr = 60
        self.topofile = ""
        self.layoutfile = ""
        self.bandwidths = []
        self.valid_conf_flag = False
        self.num_bank = 1
        self.num_port = 2

        # Layout flags with default values
        self.using_ifmap_custom_layout = False
        self.ifmap_sram_bank_bandwidth = 10
        self.ifmap_sram_bank_num = 10
        self.ifmap_sram_bank_port = 2
        self.using_filter_custom_layout = False
        self.filter_sram_bank_bandwidth = 10
        self.filter_sram_bank_num = 10
        self.filter_sram_bank_port = 2

        self.valid_df_list = ['os', 'ws', 'is']

        self.sparsity_support = False
        self.sparsity_representation = ""
        self.sparsity_optimized_mapping = False
        self.sparsity_block_size = 4
        self.sparsity_rand_seed = 40

        # Sarbartha: Added ramulator based DRAM trace support
        self.use_ramulator_trace = False

        # Time linear model parameter
        self.time_linear_model = 'None'

        # Phase 2: MQA control-plane defaults
        self.workload_type = 'gemm'
        self.sequence_length = 0
        self.batch_size = 0
        self.query_heads = 0
        self.kv_heads = 0
        self.head_dim = 0
        self.precision = 'int8'
        self.decode_tokens = 1
        self.decode_step = None
        self.softmax_variant = 'online'
        self.exp_variant = 'lookup'

    #
    def _parse_optional_int(self, config, section, key, default=None):
        if config.has_section(section) and config.has_option(section, key):
            value = config.get(section, key).strip()
            if value == '' or value.lower() == 'none':
                return default
            return int(value)
        return default

    def _parse_optional_str(self, config, section, key, default=None):
        if config.has_section(section) and config.has_option(section, key):
            value = config.get(section, key).strip()
            if value == '':
                return default
            return value
        return default

    def _load_mqa_section(self, config):
        candidate_sections = ['MQA', 'mqa', 'run_presets', 'architecture_presets', 'general']
        parsed = {}

        for section in candidate_sections:
            if not config.has_section(section):
                continue

            workload_type = self._parse_optional_str(config, section, 'workload_type', None)
            if workload_type is not None:
                parsed['workload_type'] = workload_type

            sequence_length = self._parse_optional_int(config, section, 'sequence_length', None)
            if sequence_length is not None:
                parsed['sequence_length'] = sequence_length

            batch_size = self._parse_optional_int(config, section, 'batch_size', None)
            if batch_size is not None:
                parsed['batch_size'] = batch_size

            query_heads = self._parse_optional_int(config, section, 'query_heads', None)
            if query_heads is not None:
                parsed['query_heads'] = query_heads

            kv_heads = self._parse_optional_int(config, section, 'kv_heads', None)
            if kv_heads is not None:
                parsed['kv_heads'] = kv_heads

            head_dim = self._parse_optional_int(config, section, 'head_dim', None)
            if head_dim is not None:
                parsed['head_dim'] = head_dim

            precision = self._parse_optional_str(config, section, 'precision', None)
            if precision is not None:
                parsed['precision'] = precision

            decode_tokens = self._parse_optional_int(config, section, 'decode_tokens', None)
            if decode_tokens is not None:
                parsed['decode_tokens'] = decode_tokens

            decode_step = self._parse_optional_int(config, section, 'decode_step', None)
            if decode_step is not None:
                parsed['decode_step'] = decode_step

            softmax_variant = self._parse_optional_str(config, section, 'softmax_variant', None)
            if softmax_variant is not None:
                parsed['softmax_variant'] = softmax_variant

            exp_variant = self._parse_optional_str(config, section, 'exp_variant', None)
            if exp_variant is not None:
                parsed['exp_variant'] = exp_variant

        for key, value in parsed.items():
            setattr(self, key, value)

    def _validate_mqa_config(self):
        if not self.is_mqa_workload():
            return

        positive_fields = {
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'query_heads': self.query_heads,
            'kv_heads': self.kv_heads,
            'head_dim': self.head_dim,
            'decode_tokens': self.decode_tokens,
        }

        for field_name, field_value in positive_fields.items():
            if field_value is None or int(field_value) <= 0:
                raise ValueError(
                    "MQA workload requires '{}' to be strictly greater than zero".format(field_name)
                )

        if self.workload_type not in ['baseline_mqa_decode', 'kv_stationary_mqa_decode']:
            raise ValueError(
                "Unsupported MQA workload_type '{}'. Expected baseline_mqa_decode or "
                "kv_stationary_mqa_decode".format(self.workload_type)
            )

    #
    def read_conf_file(self, conf_file_in):
        """
        Method to read the configuration file and extract all the archietctural knobs.
        """

        me = 'scale_config.' + 'read_conf_file()'

        config = cp.ConfigParser()
        config.read(conf_file_in)

        section = 'general'
        self.run_name = config.get(section, 'run_name')

        section = 'run_presets'
        bw_mode_string = config.get(section, 'InterfaceBandwidth')
        if bw_mode_string == 'USER':
            self.use_user_bandwidth = True
        elif bw_mode_string == 'CALC':
            self.use_user_bandwidth = False
        else:
            message = 'ERROR: ' + me
            message += 'Use either USER or CALC in InterfaceBandwidth feild. Aborting!'
            return

        if config.has_option(section, 'UseRamulatorTrace'):
            ramulator_on = config.get(section, 'UseRamulatorTrace')
            if ramulator_on == 'True':
                self.use_ramulator_trace = True
            else:
                self.use_ramulator_trace = False

        if config.has_option(section, 'TimeLinearModel'):
            self.time_linear_model = config.get(section, 'TimeLinearModel')
            assert self.time_linear_model in ['None', 'TPUv4', 'TPUv5e', 'TPUv6e'], \
                "ERROR: Invalid time linear model '{}'. Must be one of: None, TPUv4, TPUv5e, TPUv6e".format(
                    self.time_linear_model
                )

        div_factor = 1

        section = 'architecture_presets'
        self.array_rows = int(config.get(section, 'ArrayHeight'))
        self.array_cols = int(config.get(section, 'ArrayWidth'))
        self.ifmap_sz_kb = int(config.get(section, 'ifmapsramszkB'))
        self.filter_sz_kb = int(config.get(section, 'filtersramszkB'))
        self.ofmap_sz_kb = int(config.get(section, 'ofmapsramszkB'))
        self.ifmap_offset = int(config.get(section, 'IfmapOffset'))
        self.filter_offset = int(config.get(section, 'FilterOffset'))
        self.ofmap_offset = int(config.get(section, 'OfmapOffset'))
        self.df = config.get(section, 'Dataflow')

        if config.has_option(section, 'ReadRequestBuffer'):
            self.req_buf_sz_rd = int(config.get(section, 'ReadRequestBuffer')) // div_factor
        if config.has_option(section, 'WriteRequestBuffer'):
            self.req_buf_sz_wr = int(config.get(section, 'WriteRequestBuffer')) // div_factor

        layout_section = 'layout'
        if config.has_section(layout_section):
            self.using_ifmap_custom_layout = config.getboolean(layout_section, 'IfmapCustomLayout')
            self.using_filter_custom_layout = config.getboolean(layout_section, 'FilterCustomLayout')
            self.ifmap_sram_bank_bandwidth = int(config.get(layout_section, 'IfmapSRAMBankBandwidth'))
            self.ifmap_sram_bank_num = int(config.get(layout_section, 'IfmapSRAMBankNum'))
            self.ifmap_sram_bank_port = int(config.get(layout_section, 'IfmapSRAMBankPort'))
            self.filter_sram_bank_bandwidth = int(config.get(layout_section, 'FilterSRAMBankBandwidth'))
            self.filter_sram_bank_num = int(config.get(layout_section, 'FilterSRAMBankNum'))
            self.filter_sram_bank_port = int(config.get(layout_section, 'FilterSRAMBankPort'))

        section = 'network_presets'
        if config.has_section(section):
            if config.has_option(section, 'TopologyCsvLoc'):
                self.topofile = config.get(section, 'TopologyCsvLoc').replace('"', '').strip()
            if config.has_option(section, 'LayoutCsvLoc'):
                self.layoutfile = config.get(section, 'LayoutCsvLoc').replace('"', '').strip()
            if config.has_option(section, 'Bandwidth'):
                bw_string = config.get(section, 'Bandwidth')
                if bw_string.strip():
                    self.bandwidths = [float(x) for x in bw_string.split(',') if str(x).strip()]

        sparsity_section = 'sparsity'
        if config.has_section(sparsity_section):
            if config.has_option(sparsity_section, 'Enable'):
                self.sparsity_support = config.getboolean(sparsity_section, 'Enable')
            if config.has_option(sparsity_section, 'Representation'):
                self.sparsity_representation = config.get(sparsity_section, 'Representation')
            if config.has_option(sparsity_section, 'OptimizeMapping'):
                self.sparsity_optimized_mapping = config.getboolean(sparsity_section, 'OptimizeMapping')
            if self.sparsity_optimized_mapping and config.has_option(sparsity_section, 'BlockSize'):
                self.sparsity_block_size = int(config.get(sparsity_section, 'BlockSize'))
                assert self.sparsity_block_size <= self.array_rows, "ERROR: Invalid block size"
            if config.has_option(sparsity_section, 'RandomNumberGeneratorSeed'):
                self.sparsity_rand_seed = int(config.get(sparsity_section, 'RandomNumberGeneratorSeed'))

        self._load_mqa_section(config)
        self._validate_mqa_config()

        self.valid_conf_flag = True

    #
    def update_from_list(self, conf_list):
        """
        Method to update the parameters through a configuration list.
        """
        if not len(conf_list) > 11:
            print("ERROR: scale_config.update_from_list: "
                  "Incompatible number of elements in the list")

        self.run_name = conf_list[0]
        self.array_rows = int(conf_list[1])
        self.array_cols = int(conf_list[2])
        self.ifmap_sz_kb = int(conf_list[3])
        self.filter_sz_kb = int(conf_list[4])
        self.ofmap_sz_kb = int(conf_list[5])
        self.ifmap_offset = int(conf_list[6])
        self.filter_offset = int(conf_list[7])
        self.ofmap_offset = int(conf_list[8])
        self.df = conf_list[9]
        bw_mode_string = str(conf_list[10])

        assert bw_mode_string in ['CALC', 'USER'], 'Invalid mode of operation'
        if bw_mode_string == "USER":
            assert not len(conf_list) < 12, 'The user bandwidth needs to be provided'
            self.bandwidths = conf_list[11]
            self.use_user_bandwidth = True
        elif bw_mode_string == 'CALC':
            self.use_user_bandwidth = False

        if len(conf_list) > 12:
            self.topofile = conf_list[12]
        if len(conf_list) > 13:
            self.layoutfile = conf_list[13]

        if len(conf_list) > 14:
            self.workload_type = str(conf_list[14])
        if len(conf_list) > 15:
            self.sequence_length = int(conf_list[15])
        if len(conf_list) > 16:
            self.batch_size = int(conf_list[16])
        if len(conf_list) > 17:
            self.query_heads = int(conf_list[17])
        if len(conf_list) > 18:
            self.kv_heads = int(conf_list[18])
        if len(conf_list) > 19:
            self.head_dim = int(conf_list[19])
        if len(conf_list) > 20:
            self.precision = str(conf_list[20])
        if len(conf_list) > 21:
            self.decode_tokens = int(conf_list[21])
        if len(conf_list) > 22:
            decode_step_val = conf_list[22]
            self.decode_step = None if decode_step_val in [None, '', 'None'] else int(decode_step_val)
        if len(conf_list) > 23:
            self.softmax_variant = str(conf_list[23])
        if len(conf_list) > 24:
            self.exp_variant = str(conf_list[24])

        self._validate_mqa_config()
        self.valid_conf_flag = True

    #
    def write_conf_file(self, conf_file_out):
        """
        Method to generate a configuration file.
        """
        if not self.valid_conf_flag:
            print('ERROR: scale_config.write_conf_file: No valid config loaded')
            return

        config = cp.ConfigParser()

        section = 'general'
        config.add_section(section)
        config.set(section, 'run_name', str(self.run_name))

        section = 'architecture_presets'
        config.add_section(section)
        config.set(section, 'ArrayHeight', str(self.array_rows))
        config.set(section, 'ArrayWidth', str(self.array_cols))

        config.set(section, 'ifmapsramszkB', str(self.ifmap_sz_kb))
        config.set(section, 'filtersramszkB', str(self.filter_sz_kb))
        config.set(section, 'ofmapsramszkB', str(self.ofmap_sz_kb))

        config.set(section, 'IfmapOffset', str(self.ifmap_offset))
        config.set(section, 'FilterOffset', str(self.filter_offset))
        config.set(section, 'OfmapOffset', str(self.ofmap_offset))

        config.set(section, 'Dataflow', str(self.df))
        config.set(section, 'Bandwidth', ','.join([str(x) for x in self.bandwidths]))

        section = 'network_presets'
        config.add_section(section)
        topofile = '"' + self.topofile + '"'
        config.set(section, 'TopologyCsvLoc', str(topofile))
        layoutfile = '"' + self.layoutfile + '"'
        config.set(section, 'LayoutCsvLoc', str(layoutfile))

        section = 'run_presets'
        config.add_section(section)
        bw_mode = 'USER' if self.use_user_bandwidth else 'CALC'
        config.set(section, 'InterfaceBandwidth', str(bw_mode))
        config.set(section, 'UseRamulatorTrace', str(self.use_ramulator_trace))
        config.set(section, 'TimeLinearModel', str(self.time_linear_model))

        section = 'MQA'
        config.add_section(section)
        config.set(section, 'workload_type', str(self.workload_type))
        config.set(section, 'sequence_length', str(self.sequence_length))
        config.set(section, 'batch_size', str(self.batch_size))
        config.set(section, 'query_heads', str(self.query_heads))
        config.set(section, 'kv_heads', str(self.kv_heads))
        config.set(section, 'head_dim', str(self.head_dim))
        config.set(section, 'precision', str(self.precision))
        config.set(section, 'decode_tokens', str(self.decode_tokens))
        config.set(section, 'decode_step', '' if self.decode_step is None else str(self.decode_step))
        config.set(section, 'softmax_variant', str(self.softmax_variant))
        config.set(section, 'exp_variant', str(self.exp_variant))

        with open(conf_file_out, 'w') as configfile:
            config.write(configfile)

    #
    def set_arr_dims(self, rows=1, cols=1):
        self.array_rows = rows
        self.array_cols = cols

    #
    def set_dataflow(self, dataflow='os'):
        self.df = dataflow

    #
    def set_buffer_sizes_kb(self, ifmap_size_kb=1, filter_size_kb=1, ofmap_size_kb=1):
        self.ifmap_sz_kb = ifmap_size_kb
        self.filter_sz_kb = filter_size_kb
        self.ofmap_sz_kb = ofmap_size_kb

    #
    def set_topology_file(self, topofile=''):
        self.topofile = topofile

    #
    def set_layout_file(self, layoutfile=''):
        self.layoutfile = layoutfile

    #
    def set_offsets(self, ifmap_offset=0, filter_offset=10000000, ofmap_offset=20000000):
        self.ifmap_offset = ifmap_offset
        self.filter_offset = filter_offset
        self.ofmap_offset = ofmap_offset

    def set_bandwidth(self, bandwidths=None):
        if bandwidths is None:
            bandwidths = []
        self.bandwidths = bandwidths

    def use_user_dram_bandwidth(self):
        if not self.valid_conf_flag:
            me = 'scale_config.' + 'use_user_dram_bandwidth()'
            message = 'ERROR: ' + me + ': Configuration is not valid'
            print(message)
            return

        return self.use_user_bandwidth

    #
    def get_conf_as_list(self):
        out_list = []

        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_conf_as_list: Configuration is not valid")
            return

        out_list.append(str(self.run_name))
        out_list.append(str(self.array_rows))
        out_list.append(str(self.array_cols))
        out_list.append(str(self.ifmap_sz_kb))
        out_list.append(str(self.filter_sz_kb))
        out_list.append(str(self.ofmap_sz_kb))
        out_list.append(str(self.ifmap_offset))
        out_list.append(str(self.filter_offset))
        out_list.append(str(self.ofmap_offset))
        out_list.append(str(self.df))
        out_list.append('USER' if self.use_user_bandwidth else 'CALC')
        out_list.append(self.bandwidths)
        out_list.append(str(self.topofile))
        out_list.append(str(self.layoutfile))
        out_list.append(str(self.workload_type))
        out_list.append(str(self.sequence_length))
        out_list.append(str(self.batch_size))
        out_list.append(str(self.query_heads))
        out_list.append(str(self.kv_heads))
        out_list.append(str(self.head_dim))
        out_list.append(str(self.precision))
        out_list.append(str(self.decode_tokens))
        out_list.append('' if self.decode_step is None else str(self.decode_step))
        out_list.append(str(self.softmax_variant))
        out_list.append(str(self.exp_variant))

        return out_list

    #
    def get_run_name(self):
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_run_name() : Config data is not valid")
            return
        return self.run_name

    #
    def get_topology_path(self):
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_topology_path() : Config data is not valid")
            return
        return self.topofile

    def get_layout_path(self):
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_layout_path() : Config data is not valid")
            return
        return self.layoutfile

    def get_dataflow(self):
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_dataflow() : Config data is not valid")
            return
        return self.df

    def get_array_dims(self):
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_array_dims() : Config data is not valid")
            return
        return self.array_rows, self.array_cols

    def get_mem_sizes(self):
        me = 'scale_config.' + 'get_mem_sizes()'
        if not self.valid_conf_flag:
            message = 'ERROR: ' + me
            message += 'Config is not valid. Not returning any values'
            return
        return self.ifmap_sz_kb, self.filter_sz_kb, self.ofmap_sz_kb

    #
    def get_offsets(self):
        if self.valid_conf_flag:
            return self.ifmap_offset, self.filter_offset, self.ofmap_offset

    def get_ramulator_trace(self):
        if self.valid_conf_flag:
            return self.use_ramulator_trace

    def get_req_buf_sz_rd(self):
        if self.valid_conf_flag:
            return self.req_buf_sz_rd

    def get_req_buf_sz_wr(self):
        if self.valid_conf_flag:
            return self.req_buf_sz_wr

    #
    def get_bandwidths_as_string(self):
        if self.valid_conf_flag:
            return ','.join([str(x) for x in self.bandwidths])

    def get_ifmap_sram_bandwidth(self):
        if self.valid_conf_flag:
            return self.ifmap_sram_bank_bandwidth

    def get_filter_sram_bandwidth(self):
        if self.valid_conf_flag:
            return self.filter_sram_bank_bandwidth

    def get_bandwidths_as_list(self):
        if self.valid_conf_flag:
            return self.bandwidths

    def get_num_bank(self):
        if self.valid_conf_flag:
            return self.num_bank

    def get_num_port(self):
        if self.valid_conf_flag:
            return self.num_port

    def get_min_dram_bandwidth(self):
        if not self.use_user_dram_bandwidth():
            me = 'scale_config.' + 'get_min_dram_bandwidth()'
            message = 'ERROR: ' + me + ': No user bandwidth provided'
            print(message)
        else:
            return min(self.bandwidths)

    def get_time_linear_model(self):
        if self.valid_conf_flag:
            return self.time_linear_model
        return "Default"

    def get_workload_type(self):
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_workload_type() : Config data is not valid")
            return
        return self.workload_type

    def get_mqa_params(self):
        if not self.valid_conf_flag:
            print("ERROR: scale_config.get_mqa_params() : Config data is not valid")
            return

        return {
            'workload_type': self.workload_type,
            'sequence_length': self.sequence_length,
            'batch_size': self.batch_size,
            'query_heads': self.query_heads,
            'kv_heads': self.kv_heads,
            'head_dim': self.head_dim,
            'precision': self.precision,
            'decode_tokens': self.decode_tokens,
            'decode_step': self.decode_step,
            'softmax_variant': self.softmax_variant,
            'exp_variant': self.exp_variant,
        }

    def is_mqa_workload(self):
        return str(self.workload_type) != 'gemm'

    def force_valid(self):
        self.valid_conf_flag = True

    @staticmethod
    def get_default_conf_as_list():
        dummy_obj = scale_config()
        dummy_obj.force_valid()
        out_list = dummy_obj.get_conf_as_list()
        return out_list
