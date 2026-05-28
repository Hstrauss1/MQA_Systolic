#!/bin/bash

python3 create_action_count.py --saved_folder /Users/hudsons/Code/ScaleSim/MQA_SCALE_SIM/rundir-accelergy/scalesim_output --run_name mqa_decode_64x64_os --arch_name systolic_array --SRAM_row_size 2 --DRAM_row_size 2 --config /Users/hudsons/Code/ScaleSim/MQA_SCALE_SIM/configs/mqa_accelergy.cfg

cp /Users/hudsons/Code/ScaleSim/MQA_SCALE_SIM/rundir-accelergy/scalesim_output/mqa_decode_64x64_os/action_count.yaml ./accelergy_input/action_count.yaml

mv /Users/hudsons/Code/ScaleSim/MQA_SCALE_SIM/rundir-accelergy/scalesim_output/mqa_decode_64x64_os  /Users/hudsons/Code/ScaleSim/MQA_SCALE_SIM/rundir-accelergy/output/scale_sim_output_mqa_decode_64x64_os

