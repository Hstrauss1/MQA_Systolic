"""Parse Accelergy energy_estimation.yaml and print a grouped summary."""
import sys
import yaml

def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "output/accelergy_output_mqa_decode_64x64_os/energy_estimation.yaml"
    with open(path) as f:
        data = yaml.safe_load(f)

    components = data.get("energy_estimation", {}).get("components", [])

    groups = {"DRAM": 0.0, "GLB_SRAM": 0.0, "PE_spad": 0.0, "PE_mac": 0.0}
    for c in components:
        name = c["name"]
        energy = float(c.get("energy", 0) or 0)
        if "dram" in name.lower():
            groups["DRAM"] += energy
        elif "glb" in name.lower():
            groups["GLB_SRAM"] += energy
        elif ".mac" in name.lower():
            groups["PE_mac"] += energy
        elif "spad" in name.lower():
            groups["PE_spad"] += energy

    total = sum(groups.values())
    print(f"{'Component':<20} {'Energy (pJ)':>16} {'Share':>8}")
    print("-" * 48)
    for k, v in groups.items():
        pct = 100 * v / total if total else 0
        print(f"  {k:<18} {v:>16,.0f}   {pct:5.1f}%")
    print("-" * 48)
    print(f"  {'TOTAL':<18} {total:>16,.0f}   100.0%")
    print(f"\n  Total energy: {total/1e12:.6f} mJ  ({total/1e6:.3f} µJ)")

if __name__ == "__main__":
    main()
