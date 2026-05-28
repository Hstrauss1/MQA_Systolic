"""create_action_count.py — adapted for SCALE-Sim MQA output format.

Differences from upstream version:
  - REPEAT_CYCLE.csv may not exist → treat all repeat counts as 0
  - DETAILED_ACCESS_REPORT may have only 19 columns (no PE spad columns 19-24)
    → estimate PE spad counts from SRAM totals and array size
"""
import pandas as pd
import numpy as np
import os, sys
import yaml
from yaml import dump
from collections import OrderedDict
import argparse
import configparser as cp
import copy

# ── loaders ──────────────────────────────────────────────────────────────────

def load_detail_report_data(data_dir, run_name):
    csv_filename = os.path.join(data_dir, run_name, 'DETAILED_ACCESS_REPORT.csv')
    return pd.read_csv(csv_filename, sep=r'\s*,\s*', engine='python')

def load_repeat_report_data(data_dir, run_name):
    """Return None if the file doesn't exist (older SCALE-Sim versions)."""
    csv_filename = os.path.join(data_dir, run_name, 'REPEAT_CYCLE.csv')
    if not os.path.exists(csv_filename):
        return None
    return pd.read_csv(csv_filename, sep=r'\s*,\s*', engine='python')

def load_compute_report_data(data_dir, run_name):
    csv_filename = os.path.join(data_dir, run_name, 'COMPUTE_REPORT.csv')
    return pd.read_csv(csv_filename, sep=r'\s*,\s*', engine='python')

# ── yaml helpers ─────────────────────────────────────────────────────────────

def write_yaml_file(filepath, content):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if os.path.exists(filepath):
        os.remove(filepath)
    with open(filepath, 'a') as f:
        f.write(dump(content, default_flow_style=False))

def yaml_name_generator(name, sram, address_delta, data_delta, counts):
    if name == 'idle':
        return {'name': name, 'counts': counts}
    return {'name': name,
            'arguments': {'address_delta': address_delta, 'data_delta': data_delta},
            'counts': counts}

def yaml_name_generator_dram(name, counts):
    return {'name': name, 'counts': counts}

def yaml_name_generator_mac(name, counts):
    return {'name': name, 'counts': counts}

def yaml_generator(name, contents):
    return {'name': name, 'action_counts': contents}

# ── main ─────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("--saved_folder")
parser.add_argument("--run_name")
parser.add_argument("--arch_name")
parser.add_argument("--SRAM_row_size")
parser.add_argument("--DRAM_row_size")
parser.add_argument("--config")
parser.add_argument("--SRAM_repeat_check", default='True')
args = parser.parse_args()

saved_folder      = args.saved_folder
run_name          = args.run_name
arch_name         = args.arch_name
SRAM_row_size     = int(args.SRAM_row_size)
DRAM_row_size     = int(args.DRAM_row_size)
SRAM_repeat_check = args.SRAM_repeat_check == 'True'

# ── load SCALE-Sim outputs ───────────────────────────────────────────────────

data_dir = os.path.join(os.getcwd(), os.pardir, saved_folder)
detail_df  = load_detail_report_data(data_dir, run_name)
repeat_df  = load_repeat_report_data(data_dir, run_name)
compute_df = load_compute_report_data(data_dir, run_name)

# Collapse layers (sum across rows, drop last col which is empty/trailing comma)
detail_access = np.sum(detail_df.to_numpy()[:, :-1], axis=0)
if repeat_df is not None:
    repeat_access = np.sum(repeat_df.to_numpy()[:, :-1], axis=0)
    has_repeat = True
else:
    repeat_access = np.zeros(10, dtype=float)
    has_repeat = False

# Column indices in DETAILED_ACCESS_REPORT
SRAM_ifmap_start_cycle  = 1
SRAM_ifmap_stop_cycle   = 2
SRAM_ifmap_reads        = 3
SRAM_filter_start_cycle = 4
SRAM_filter_stop_cycle  = 5
SRAM_filter_reads       = 6
SRAM_ofmap_start_cycle  = 7
SRAM_ofmap_stop_cycle   = 8
SRAM_ofmap_writes       = 9
DRAM_ifmap_start_cycle  = 10
DRAM_ifmap_stop_cycle   = 11
DRAM_ifmap_reads        = 12
DRAM_filter_start_cycle = 13
DRAM_filter_stop_cycle  = 14
DRAM_filter_reads       = 15
DRAM_ofmap_start_cycle  = 16
DRAM_ofmap_stop_cycle   = 17
DRAM_ofmap_writes       = 18

# Optional PE spad columns (may not exist in older SCALE-Sim output)
IFMAP_Write_Count  = 19
IFMAP_Read_Count   = 20
Filter_Write_Count = 21
Filter_Read_Count  = 22
OFMAP_Write_Count  = 23
OFMAP_Read_Count   = 24

# Column indices in REPEAT_CYCLE (if present)
ifmap_sram_repeat  = 1
filter_sram_repeat = 2
ofmap_sram_repeat  = 3
ifmap_dram_repeat  = 4
filter_dram_repeat = 5
ofmap_dram_repeat  = 6

# ── read PE array size from config ───────────────────────────────────────────

config_path = os.path.join(os.getcwd(), os.pardir, args.config)
cfg = cp.ConfigParser()
cfg.read(config_path)
arrayheight   = int(cfg.get('architecture_presets', 'ArrayHeight'))
arraywidth    = int(cfg.get('architecture_presets', 'ArrayWidth'))
dataflow      = cfg.get('architecture_presets', 'Dataflow').strip()
PE_array_size = arrayheight * arraywidth

# ── SRAM action counts ────────────────────────────────────────────────────────

if SRAM_repeat_check and has_repeat:
    SRAM_ifmap_idle   = int((detail_access[SRAM_ifmap_stop_cycle]  - detail_access[SRAM_ifmap_start_cycle]  + 1) * arrayheight - detail_access[SRAM_ifmap_reads])
    SRAM_ifmap_random = int(detail_access[SRAM_ifmap_reads] - repeat_access[ifmap_sram_repeat])
    SRAM_ifmap_repeat = int(repeat_access[ifmap_sram_repeat])

    SRAM_filter_idle   = int((detail_access[SRAM_filter_stop_cycle] - detail_access[SRAM_filter_start_cycle] + 1) * arraywidth - detail_access[SRAM_filter_reads])
    SRAM_filter_random = int(detail_access[SRAM_filter_reads] - repeat_access[filter_sram_repeat])
    SRAM_filter_repeat = int(repeat_access[filter_sram_repeat])

    SRAM_ofmap_idle   = int((detail_access[SRAM_ofmap_stop_cycle]  - detail_access[SRAM_ofmap_start_cycle]  + 1) * arraywidth - detail_access[SRAM_ofmap_writes])
    SRAM_ofmap_random = int(detail_access[SRAM_ofmap_writes] - repeat_access[ofmap_sram_repeat])
    SRAM_ofmap_repeat = int(repeat_access[ofmap_sram_repeat])
else:
    SRAM_ifmap_idle   = max(0, int((detail_access[SRAM_ifmap_stop_cycle]  - detail_access[SRAM_ifmap_start_cycle]  + 1) * arrayheight - detail_access[SRAM_ifmap_reads]))
    SRAM_ifmap_random = int(detail_access[SRAM_ifmap_reads])
    SRAM_ifmap_repeat = 0

    SRAM_filter_idle   = max(0, int((detail_access[SRAM_filter_stop_cycle] - detail_access[SRAM_filter_start_cycle] + 1) * arraywidth - detail_access[SRAM_filter_reads]))
    SRAM_filter_random = int(detail_access[SRAM_filter_reads])
    SRAM_filter_repeat = 0

    SRAM_ofmap_idle   = max(0, int((detail_access[SRAM_ofmap_stop_cycle]  - detail_access[SRAM_ofmap_start_cycle]  + 1) * arraywidth - detail_access[SRAM_ofmap_writes]))
    SRAM_ofmap_random = int(detail_access[SRAM_ofmap_writes])
    SRAM_ofmap_repeat = 0

# ── DRAM action counts ────────────────────────────────────────────────────────

DRAM_ifmap_idle   = max(0, int((detail_access[DRAM_ifmap_stop_cycle]  - detail_access[DRAM_ifmap_start_cycle]  + 1) * DRAM_row_size - detail_access[DRAM_ifmap_reads]))
DRAM_ifmap_random = int(detail_access[DRAM_ifmap_reads])

DRAM_filter_idle   = max(0, int((detail_access[DRAM_filter_stop_cycle] - detail_access[DRAM_filter_start_cycle] + 1) * DRAM_row_size - detail_access[DRAM_filter_reads]))
DRAM_filter_random = int(detail_access[DRAM_filter_reads])

DRAM_ofmap_idle   = max(0, int((detail_access[DRAM_ofmap_stop_cycle]  - detail_access[DRAM_ofmap_start_cycle]  + 1) * DRAM_row_size - detail_access[DRAM_ofmap_writes]))
DRAM_ofmap_random = int(detail_access[DRAM_ofmap_writes])

# ── PE spad counts ────────────────────────────────────────────────────────────
# Use columns 19-24 if present; otherwise estimate from SRAM totals.

n_cols = len(detail_access)
if n_cols > OFMAP_Read_Count:
    PE_weights_spad_write = int(detail_access[Filter_Write_Count]) / PE_array_size
    PE_weights_spad_read  = int(detail_access[Filter_Read_Count])  / PE_array_size
    PE_ifmap_spad_write   = int(detail_access[IFMAP_Write_Count])  / PE_array_size
    PE_ifmap_spad_read    = int(detail_access[IFMAP_Read_Count])   / PE_array_size
    PE_psum_spad_write    = int(detail_access[OFMAP_Write_Count])  / PE_array_size
    PE_psum_spad_read     = int(detail_access[OFMAP_Read_Count])   / PE_array_size
else:
    # Estimate: each PE handles 1/PE_count share of global SRAM traffic
    PE_weights_spad_write = SRAM_filter_random / PE_array_size
    PE_weights_spad_read  = SRAM_filter_random / PE_array_size
    PE_ifmap_spad_write   = SRAM_ifmap_random  / PE_array_size
    PE_ifmap_spad_read    = SRAM_ifmap_random  / PE_array_size
    PE_psum_spad_write    = SRAM_ofmap_random  / PE_array_size
    PE_psum_spad_read     = SRAM_ofmap_random  / PE_array_size

# ── MAC counts ────────────────────────────────────────────────────────────────

layer_cycle   = compute_df['Total Cycles'].values.tolist()
PE_MAC_random = sum(layer_cycle)

# ── build action_counts YAML ─────────────────────────────────────────────────

action_counts = []

# DRAM
action_counts.append(yaml_generator(arch_name + '.ifmap_dram',
    [yaml_name_generator_dram('read', DRAM_ifmap_random),
     yaml_name_generator_dram('idle', DRAM_ifmap_idle)]))
action_counts.append(yaml_generator(arch_name + '.weights_dram',
    [yaml_name_generator_dram('read', DRAM_filter_random),
     yaml_name_generator_dram('idle', DRAM_filter_idle)]))
action_counts.append(yaml_generator(arch_name + '.psum_dram',
    [yaml_name_generator_dram('write', DRAM_ofmap_random),
     yaml_name_generator_dram('idle',  DRAM_ofmap_idle)]))

# GLB SRAMs
action_counts.append(yaml_generator(arch_name + '.ifmap_glb',
    [yaml_name_generator('read',  1, 1, 1, SRAM_ifmap_random),
     yaml_name_generator('read',  1, 0, 0, SRAM_ifmap_repeat),
     yaml_name_generator('idle',  1, 0, 0, SRAM_ifmap_idle)]))
action_counts.append(yaml_generator(arch_name + '.weights_glb',
    [yaml_name_generator('read',  1, 1, 1, SRAM_filter_random),
     yaml_name_generator('read',  1, 0, 0, SRAM_filter_repeat),
     yaml_name_generator('idle',  1, 0, 0, SRAM_filter_idle)]))
action_counts.append(yaml_generator(arch_name + '.psum_glb',
    [yaml_name_generator('update', 1, 1, 1, SRAM_ofmap_random),
     yaml_name_generator('update', 1, 0, 0, SRAM_ofmap_repeat),
     yaml_name_generator('idle',   1, 0, 0, SRAM_ofmap_idle)]))

# PE scratchpads and MACs
PE_weight_ac = [yaml_name_generator('write', 1, 1, 1, PE_weights_spad_write),
                yaml_name_generator('read',  1, 0, 0, PE_weights_spad_read)]
PE_ifmap_ac  = [yaml_name_generator('write', 1, 1, 1, PE_ifmap_spad_write),
                yaml_name_generator('read',  1, 0, 0, PE_ifmap_spad_read)]
PE_mac_ac    = [yaml_name_generator_mac('mac_random', PE_MAC_random)]

if dataflow == 'os':
    PE_psum_ac = [yaml_name_generator('write', 1, 0, 0, PE_psum_spad_write),
                  yaml_name_generator('read',  1, 0, 0, PE_psum_spad_read)]
else:
    PE_psum_ac = [yaml_name_generator('write', 1, 1, 1, PE_psum_spad_write),
                  yaml_name_generator('read',  1, 0, 0, PE_psum_spad_read)]

for n in range(PE_array_size):
    action_counts.append(yaml_generator(arch_name + f'.PE[{n}].weights_spad', copy.deepcopy(PE_weight_ac)))
for n in range(PE_array_size):
    action_counts.append(yaml_generator(arch_name + f'.PE[{n}].ifmap_spad',   copy.deepcopy(PE_ifmap_ac)))
for n in range(PE_array_size):
    action_counts.append(yaml_generator(arch_name + f'.PE[{n}].psum_spad',    copy.deepcopy(PE_psum_ac)))
for n in range(PE_array_size):
    action_counts.append(yaml_generator(arch_name + f'.PE[{n}].mac',          copy.deepcopy(PE_mac_ac)))

# ── write output ─────────────────────────────────────────────────────────────

output_path = os.path.join(os.path.join(os.getcwd(), os.pardir),
                            saved_folder, run_name, 'action_count.yaml')
result = {'action_counts': {'version': 0.3, 'local': action_counts}}
write_yaml_file(output_path, result)
print(f"Written: {output_path}")
print(f"  DRAM ifmap reads:   {DRAM_ifmap_random:,}")
print(f"  DRAM filter reads:  {DRAM_filter_random:,}")
print(f"  DRAM ofmap writes:  {DRAM_ofmap_random:,}")
print(f"  SRAM ifmap reads:   {SRAM_ifmap_random:,}")
print(f"  SRAM filter reads:  {SRAM_filter_random:,}")
print(f"  SRAM ofmap writes:  {SRAM_ofmap_random:,}")
print(f"  PE MAC count:       {PE_MAC_random:,}")
print(f"  PE array size:      {PE_array_size} ({arrayheight}x{arraywidth})")
