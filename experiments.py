# Jacob Mitchell, Kyle Axtell
# CS 456 - Data Compression
# experiments.py
# 3/6/26

"""
Final Project Experiment: Huffman vs Huffman+OBST

Runs MANY experiments, with repeated runs, to produce data for final report

Outputs (in --outdir):
  - metrics.csv     (raw row per run per configuration)
  - summary.csv     (grouped mean/stdev)
  - *.png           (presentation charts)

How to run:
  python experiments.py --outdir results --runs 5
  python experiments.py --outdir results --runs 7 --exp1_size_kb 1024 --exp2_max_mb 16
  python experiments.py --outdir results --runs 5 --exp1_generators uniform256,zipf128,repetitive90,english_like

Notes:
"""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import matplotlib.pyplot as plt

# Our implementations
import huffman as huff
import obst as obst_mod


# Utilities

def now_ns() -> int:
    return time.perf_counter_ns()

def ns_to_ms(ns: int) -> float:
    return ns / 1_000_000.0

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def freq_table(data: bytes) -> Dict[int, int]:
    ft: Dict[int, int] = {}
    for b in data:
        ft[b] = ft.get(b, 0) + 1
    return ft

def _ensure_nonempty_code_map(code_map: Dict[int, str]) -> Dict[int, str]:
    # Edge case of file with one unique symbol -> Huffman code can be empty
    # Force it to 0 so that the encoding/decoding works
    if len(code_map) == 1:
        k = next(iter(code_map.keys()))
        if code_map[k] == "":
            code_map[k] = "0"
    return code_map


def pack_bits_from_codes(data: bytes, code_map: Dict[int, str]) -> Tuple[bytes, int]:
    """
    Converts Huffman codes into packed bytes
    Returns (packed_bytes, pad_bits) where pad_bits is number of 0 bits added at the end
    """
    out = bytearray()
    acc = 0
    acc_bits = 0

    for b in data:
        bits = code_map[b]
        for ch in bits:
            acc = (acc << 1) | (1 if ch == '1' else 0)
            acc_bits += 1
            if acc_bits == 8:
                out.append(acc & 0xFF)
                acc = 0
                acc_bits = 0

    pad_bits = 0
    if acc_bits != 0:
        pad_bits = 8 - acc_bits
        acc = acc << pad_bits
        out.append(acc & 0xFF)

    return bytes(out), pad_bits


def unpack_and_decode(packed: bytes, pad_bits: int, root: huff.HuffmanNode) -> bytes:
    """
    Decode packed bits using Huffman tree
    """
    total_bits = len(packed) * 8 - pad_bits
    decoded = bytearray()
    node = root
    bit_index = 0

    for byte in packed:
        for i in range(7, -1, -1):
            if bit_index >= total_bits:
                break
            bit = (byte >> i) & 1
            node = node.right if bit == 1 else node.left

            # Leaf
            if node.symbol is not None:
                decoded.append(node.symbol)
                node = root
            bit_index += 1

    return bytes(decoded)


def build_obst_for_codes(code_map: Dict[int, str], ft: Dict[int, int]) -> obst_mod.OBSTNode:
    keys = sorted(code_map.keys())
    total = sum(ft[k] for k in keys)
    probs = [(ft[k] / total) for k in keys]
    values = [code_map[k] for k in keys]
    return obst_mod.build_obst(keys, probs, values=values)


def pack_bits_with_obst(data: bytes, obst_root: obst_mod.OBSTNode) -> Tuple[bytes, int, int]:
    """
    Encode using OBST lookup for each symbol
    Returns packed bytes, pad bits, and total comparisons from searches
    """
    out = bytearray()
    acc = 0
    acc_bits = 0
    total_comparisons = 0

    for b in data:
        code, comps = obst_mod.obst_search(obst_root, b)
        total_comparisons += comps
        if code is None:
            raise ValueError(f"OBST lookup failed for symbol {b}")

        for ch in code:
            acc = (acc << 1) | (1 if ch == '1' else 0)
            acc_bits += 1
            if acc_bits == 8:
                out.append(acc & 0xFF)
                acc = 0
                acc_bits = 0

    pad_bits = 0
    if acc_bits != 0:
        pad_bits = 8 - acc_bits
        acc = acc << pad_bits
        out.append(acc & 0xFF)

    return bytes(out), pad_bits, total_comparisons


# Synthetic dataset generators

def gen_uniform(size: int, alphabet: int = 256, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    return bytes(rng.randrange(0, alphabet) for _ in range(size))

def gen_repetitive(size: int, dominant: int = ord('A'), dom_frac: float = 0.90, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    other_symbols = [i for i in range(256) if i != dominant]
    out = bytearray()
    for _ in range(size):
        if rng.random() < dom_frac:
            out.append(dominant)
        else:
            out.append(rng.choice(other_symbols))
    return bytes(out)

def gen_zipf_like(size: int, alphabet: int = 128, s: float = 1.2, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    weights = [1.0 / ((i + 1) ** s) for i in range(alphabet)]
    total = sum(weights)
    probs = [w / total for w in weights]
    # CDF
    cdf = []
    acc = 0.0
    for p in probs:
        acc += p
        cdf.append(acc)

    out = bytearray()
    for _ in range(size):
        r = rng.random()
        lo, hi = 0, len(cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if r <= cdf[mid]:
                hi = mid
            else:
                lo = mid + 1
        out.append(lo)
    return bytes(out)

def gen_english_like(size: int, seed: int = 0) -> bytes:
    rng = random.Random(seed)
    chars = (
        " etaoinshrdlcumwfgypbvkjxq"
        "ETAOINSHRDLCUMWFGYPBVKJXQ"
        "\n"
    )
    weights = []
    for ch in chars:
        if ch == ' ':
            weights.append(13.0)
        elif ch == '\n':
            weights.append(1.5)
        elif ch.lower() in "etaoinshrdlu":
            weights.append(6.0)
        elif ch.lower() in "cmfwgypbvk":
            weights.append(2.5)
        else:
            weights.append(1.2)

    total = sum(weights)
    probs = [w / total for w in weights]
    cdf = []
    acc = 0.0
    for p in probs:
        acc += p
        cdf.append(acc)

    out = bytearray()
    for _ in range(size):
        r = rng.random()
        lo, hi = 0, len(cdf) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if r <= cdf[mid]:
                hi = mid
            else:
                lo = mid + 1
        out.append(ord(chars[lo]))
    return bytes(out)

GENERATOR_REGISTRY: Dict[str, Callable[[int, int], bytes]] = {
    "uniform256": lambda size, seed: gen_uniform(size, alphabet=256, seed=seed),
    "uniform128": lambda size, seed: gen_uniform(size, alphabet=128, seed=seed),
    "zipf128": lambda size, seed: gen_zipf_like(size, alphabet=128, s=1.2, seed=seed),
    "zipf64": lambda size, seed: gen_zipf_like(size, alphabet=64, s=1.2, seed=seed),
    "repetitive90": lambda size, seed: gen_repetitive(size, dominant=ord('A'), dom_frac=0.90, seed=seed),
    "repetitive99": lambda size, seed: gen_repetitive(size, dominant=ord('A'), dom_frac=0.99, seed=seed),
    "english_like": lambda size, seed: gen_english_like(size, seed=seed),
}

def generate_dataset(name: str, size_bytes: int, seed: int) -> Tuple[str, bytes]:
    """
    Helper: if a dataset name is not recognized, we fall back to uniform256
    so the run does not fail completely or crahs
    """
    fn = GENERATOR_REGISTRY.get(name)
    if fn is None:
        # Fallback (explicit + safe)
        return f"{name}_fallback_uniform256", gen_uniform(size_bytes, alphabet=256, seed=seed)
    return name, fn(size_bytes, seed)




# Experiment runner

@dataclass
class MetricRow:
    exp_name: str
    dataset_name: str
    file_size_bytes: int
    run_id: int
    pipeline: str  # "huffman" or "huffman+obst"
    unique_symbols: int

    build_huffman_ms: float
    build_obst_ms: float
    encode_ms: float
    decode_ms: float
    total_ms: float

    compressed_bytes: int
    pad_bits: int
    compression_ratio: float

    lookup_comparisons_total: int
    lookup_comparisons_per_symbol: float
    correctness_ok: int  # 1 or 0


def run_one(data: bytes, pipeline: str) -> MetricRow:
    ft = freq_table(data)

    # Huffman build
    t0 = now_ns()
    root = huff.build_huffman_tree(ft)
    code_map = huff.generate_huffman_codes(root)
    code_map = _ensure_nonempty_code_map(code_map)
    t1 = now_ns()
    build_huffman_ms = ns_to_ms(t1 - t0)

    build_obst_ms = 0.0
    comparisons_total = 0

    if pipeline == "huffman":
        # encode
        t2 = now_ns()
        packed, pad_bits = pack_bits_from_codes(data, code_map)
        t3 = now_ns()
        encode_ms = ns_to_ms(t3 - t2)

        # decode
        t4 = now_ns()
        decoded = unpack_and_decode(packed, pad_bits, root)
        t5 = now_ns()
        decode_ms = ns_to_ms(t5 - t4)

    elif pipeline == "huffman+obst":
        # OBST build
        t2 = now_ns()
        obst_root = build_obst_for_codes(code_map, ft)
        t3 = now_ns()
        build_obst_ms = ns_to_ms(t3 - t2)

        # encode using OBST lookup
        t4 = now_ns()
        packed, pad_bits, comparisons_total = pack_bits_with_obst(data, obst_root)
        t5 = now_ns()
        encode_ms = ns_to_ms(t5 - t4)

        # decode using Huffman tree 
        t6 = now_ns()
        decoded = unpack_and_decode(packed, pad_bits, root)
        t7 = now_ns()
        decode_ms = ns_to_ms(t7 - t6)
    else:
        raise ValueError("pipeline must be 'huffman' or 'huffman+obst'")

    correctness_ok = 1 if decoded == data else 0
    comp_bytes = len(packed)
    ratio = comp_bytes / max(1, len(data))
    per_symbol = (comparisons_total / max(1, len(data))) if pipeline == "huffman+obst" else 0.0
    total_ms = build_huffman_ms + build_obst_ms + encode_ms + decode_ms

    return MetricRow(
        exp_name="",
        dataset_name="",
        file_size_bytes=len(data),
        run_id=0,
        pipeline=pipeline,
        unique_symbols=len(ft),
        build_huffman_ms=build_huffman_ms,
        build_obst_ms=build_obst_ms,
        encode_ms=encode_ms,
        decode_ms=decode_ms,
        total_ms=total_ms,
        compressed_bytes=comp_bytes,
        pad_bits=pad_bits,
        compression_ratio=ratio,
        lookup_comparisons_total=comparisons_total,
        lookup_comparisons_per_symbol=per_symbol,
        correctness_ok=correctness_ok
    )


def write_csv(path: Path, rows: List[MetricRow]) -> None:
    fields = list(MetricRow.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: getattr(r, k) for k in fields})


def group_summary(rows: List[MetricRow], out_path: Path) -> None:
    """
    Group by exp_name, dataset_name, file_size_bytes, pipeline and compute mean/stdev
    """
    key_to: Dict[Tuple[str, str, int, str], List[MetricRow]] = {}
    for r in rows:
        key = (r.exp_name, r.dataset_name, r.file_size_bytes, r.pipeline)
        key_to.setdefault(key, []).append(r)

    summary_fields = [
        "exp_name","dataset_name","file_size_bytes","pipeline","n_runs",
        "compression_ratio_mean","compression_ratio_stdev",
        "encode_ms_mean","encode_ms_stdev",
        "decode_ms_mean","decode_ms_stdev",
        "build_huffman_ms_mean","build_huffman_ms_stdev",
        "build_obst_ms_mean","build_obst_ms_stdev",
        "total_ms_mean","total_ms_stdev",
        "lookup_comparisons_per_symbol_mean","lookup_comparisons_per_symbol_stdev",
        "correctness_ok_rate"
    ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        for key, items in sorted(key_to.items()):
            exp_name, dataset_name, size_b, pipeline = key

            def mean_stdev(vals: List[float]) -> Tuple[float, float]:
                if len(vals) == 1:
                    return vals[0], 0.0
                return statistics.mean(vals), statistics.stdev(vals)

            cr_m, cr_s = mean_stdev([x.compression_ratio for x in items])
            en_m, en_s = mean_stdev([x.encode_ms for x in items])
            de_m, de_s = mean_stdev([x.decode_ms for x in items])
            bh_m, bh_s = mean_stdev([x.build_huffman_ms for x in items])
            bo_m, bo_s = mean_stdev([x.build_obst_ms for x in items])
            tt_m, tt_s = mean_stdev([x.total_ms for x in items])
            lc_m, lc_s = mean_stdev([x.lookup_comparisons_per_symbol for x in items])
            ok_rate = sum(x.correctness_ok for x in items) / len(items)

            w.writerow({
                "exp_name": exp_name,
                "dataset_name": dataset_name,
                "file_size_bytes": size_b,
                "pipeline": pipeline,
                "n_runs": len(items),
                "compression_ratio_mean": cr_m,
                "compression_ratio_stdev": cr_s,
                "encode_ms_mean": en_m,
                "encode_ms_stdev": en_s,
                "decode_ms_mean": de_m,
                "decode_ms_stdev": de_s,
                "build_huffman_ms_mean": bh_m,
                "build_huffman_ms_stdev": bh_s,
                "build_obst_ms_mean": bo_m,
                "build_obst_ms_stdev": bo_s,
                "total_ms_mean": tt_m,
                "total_ms_stdev": tt_s,
                "lookup_comparisons_per_symbol_mean": lc_m,
                "lookup_comparisons_per_symbol_stdev": lc_s,
                "correctness_ok_rate": ok_rate,
            })



# Plotting

def plot_experiment_1(rows: List[MetricRow], outdir: Path) -> None:
    exp_rows = [r for r in rows if r.exp_name == "exp1_distribution"]
    if not exp_rows:
        return

    datasets = sorted(set(r.dataset_name for r in exp_rows))
    pipelines = ["huffman", "huffman+obst"]

    def mean_for(dataset: str, pipeline: str, field: str) -> float:
        vals = [getattr(r, field) for r in exp_rows if r.dataset_name == dataset and r.pipeline == pipeline]
        return statistics.mean(vals) if vals else float("nan")

    x = list(range(len(datasets)))

    plt.figure()
    for p in pipelines:
        y = [mean_for(d, p, "compression_ratio") for d in datasets]
        plt.plot(x, y, marker="o", label=p)
    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.ylabel("Compressed Bytes / Original Bytes")
    plt.title("Experiment 1: Compression Ratio by Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "exp1_compression_ratio.png", dpi=200)
    plt.close()

    plt.figure()
    for p in pipelines:
        y = [mean_for(d, p, "encode_ms") for d in datasets]
        plt.plot(x, y, marker="o", label=p)
    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.ylabel("Encode Time (ms)")
    plt.title("Experiment 1: Encode Time by Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "exp1_encode_time.png", dpi=200)
    plt.close()

    plt.figure()
    for p in pipelines:
        y = [mean_for(d, p, "total_ms") for d in datasets]
        plt.plot(x, y, marker="o", label=p)
    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.ylabel("Total Time (ms) (build + encode + decode)")
    plt.title("Experiment 1: Total Runtime by Distribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "exp1_total_time.png", dpi=200)
    plt.close()


def plot_experiment_2(rows: List[MetricRow], outdir: Path) -> None:
    exp_rows = [r for r in rows if r.exp_name == "exp2_size_scaling"]
    if not exp_rows:
        return

    distributions = sorted(set(r.dataset_name for r in exp_rows))
    pipelines = ["huffman", "huffman+obst"]

    for dist in distributions:
        dist_rows = [r for r in exp_rows if r.dataset_name == dist]
        sizes = sorted(set(r.file_size_bytes for r in dist_rows))

        def mean_size(size: int, pipeline: str, field: str) -> float:
            vals = [getattr(r, field) for r in dist_rows if r.file_size_bytes == size and r.pipeline == pipeline]
            return statistics.mean(vals) if vals else float("nan")

        plt.figure()
        for p in pipelines:
            y = [mean_size(s, p, "encode_ms") for s in sizes]
            plt.plot(sizes, y, marker="o", label=p)
        plt.xlabel("File Size (bytes)")
        plt.ylabel("Encode Time (ms)")
        plt.title(f"Experiment 2: Encode Time vs Size ({dist})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"exp2_encode_time_{dist}.png", dpi=200)
        plt.close()

        plt.figure()
        for p in pipelines:
            y = [mean_size(s, p, "compression_ratio") for s in sizes]
            plt.plot(sizes, y, marker="o", label=p)
        plt.xlabel("File Size (bytes)")
        plt.ylabel("Compressed Bytes / Original Bytes")
        plt.title(f"Experiment 2: Compression Ratio vs Size ({dist})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"exp2_compression_ratio_{dist}.png", dpi=200)
        plt.close()

        plt.figure()
        for p in pipelines:
            y = [mean_size(s, p, "total_ms") for s in sizes]
            plt.plot(sizes, y, marker="o", label=p)
        plt.xlabel("File Size (bytes)")
        plt.ylabel("Total Time (ms) (build + encode + decode)")
        plt.title(f"Experiment 2: Total Runtime vs Size ({dist})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(outdir / f"exp2_total_time_{dist}.png", dpi=200)
        plt.close()

        plt.figure()
        y = [mean_size(s, "huffman+obst", "lookup_comparisons_per_symbol") for s in sizes]
        plt.plot(sizes, y, marker="o")
        plt.xlabel("File Size (bytes)")
        plt.ylabel("Comparisons per Symbol (avg)")
        plt.title(f"Experiment 2: OBST Lookup Cost vs Size ({dist})")
        plt.tight_layout()
        plt.savefig(outdir / f"exp2_lookup_cost_{dist}.png", dpi=200)
        plt.close()


def plot_experiment_3(rows: List[MetricRow], outdir: Path) -> None:
    exp_rows = [r for r in rows if r.exp_name == "exp3_pipeline_compare"]
    if not exp_rows:
        return

    datasets = sorted(set(r.dataset_name for r in exp_rows))
    x = list(range(len(datasets)))
    pipelines = ["huffman", "huffman+obst"]

    def mean_total(dataset: str, pipeline: str) -> float:
        vals = [r.total_ms for r in exp_rows if r.dataset_name == dataset and r.pipeline == pipeline]
        return statistics.mean(vals) if vals else float("nan")

    plt.figure()
    for p in pipelines:
        y = [mean_total(d, p) for d in datasets]
        plt.plot(x, y, marker="o", label=p)
    plt.xticks(x, datasets, rotation=20, ha="right")
    plt.ylabel("Total Time (ms) (build + encode + decode)")
    plt.title("Experiment 3: End-to-End Time by Dataset")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "exp3_total_time.png", dpi=200)
    plt.close()





# Main 

def parse_csv_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, default="results", help="Output directory for CSV and plots")
    ap.add_argument("--runs", type=int, default=5, help="Repetitions per configuration (>=3 recommended for timing)")
    ap.add_argument("--seed", type=int, default=123, help="Base random seed")

    # Experiment toggles
    ap.add_argument("--no_exp1", action="store_true", help="Disable experiment 1 (distribution)")
    ap.add_argument("--no_exp2", action="store_true", help="Disable experiment 2 (size scaling)")
    ap.add_argument("--no_exp3", action="store_true", help="Disable experiment 3 (pipeline compare)")

    # Experiment 1 controls
    ap.add_argument("--exp1_size_kb", type=int, default=512, help="Experiment 1 fixed file size in KB")
    ap.add_argument("--exp1_generators", type=str, default="uniform256,zipf128,repetitive90,english_like",
                    help="Comma-separated dataset generator names for experiment 1")

    # Experiment 2 controls
    ap.add_argument("--exp2_min_kb", type=int, default=4, help="Experiment 2 min size in KB (power-of-two growth)")
    ap.add_argument("--exp2_max_mb", type=int, default=8, help="Experiment 2 max size in MB (power-of-two growth)")
    ap.add_argument("--exp2_generators", type=str, default="uniform256,zipf128,repetitive90",
                    help="Comma-separated dataset generator names for experiment 2")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    safe_mkdir(outdir)

    rows: List[MetricRow] = []

    pipelines = ("huffman", "huffman+obst")

    # Experiment 1: distributions (fixed size)
    if not args.no_exp1:
        fixed_size = max(1, args.exp1_size_kb) * 1024
        gen_names = parse_csv_list(args.exp1_generators)

        for gen_name in gen_names:
            for run_id in range(1, args.runs + 1):
                dataset_name, data = generate_dataset(gen_name, fixed_size, args.seed + run_id)
                for pipeline in pipelines:
                    row = run_one(data, pipeline)
                    row.exp_name = "exp1_distribution"
                    row.dataset_name = dataset_name
                    row.run_id = run_id
                    rows.append(row)

    # Experiment 2: size scaling (multiple sizes, powers of 2)
    if not args.no_exp2:
        min_bytes = max(1, args.exp2_min_kb) * 1024
        max_bytes = max(1, args.exp2_max_mb) * 1024 * 1024

        sizes: List[int] = []
        s = min_bytes
        while s <= max_bytes:
            sizes.append(s)
            s *= 2

        gen_names = parse_csv_list(args.exp2_generators)

        for gen_name in gen_names:
            for size_b in sizes:
                for run_id in range(1, args.runs + 1):
                    dataset_name, data = generate_dataset(gen_name, size_b, args.seed + 10_000 + size_b + run_id)
                    for pipeline in pipelines:
                        row = run_one(data, pipeline)
                        row.exp_name = "exp2_size_scaling"
                        row.dataset_name = dataset_name
                        row.run_id = run_id
                        rows.append(row)

    # Experiment 3: pipeline compare on mixed (generated only by default)
    if not args.no_exp3:
        mixed_specs = [
            ("english_like", 1 * 1024 * 1024),
            ("uniform256",   1 * 1024 * 1024),
            ("zipf128",      1 * 1024 * 1024),
            ("repetitive90", 1 * 1024 * 1024),
            ("repetitive99", 1 * 1024 * 1024),
            ("uniform128",   1 * 1024 * 1024),
        ]

        for gen_name, size_b in mixed_specs:
            for run_id in range(1, args.runs + 1):
                dataset_name, data = generate_dataset(gen_name, size_b, args.seed + 200_000 + run_id + size_b)
                for pipeline in pipelines:
                    row = run_one(data, pipeline)
                    row.exp_name = "exp3_pipeline_compare"
                    row.dataset_name = dataset_name + f"_{size_b//1024}kb"
                    row.run_id = run_id
                    rows.append(row)

    # Write raw and summary
    metrics_csv = outdir / "metrics.csv"
    summary_csv = outdir / "summary.csv"
    write_csv(metrics_csv, rows)
    group_summary(rows, summary_csv)

    # Plots
    plot_experiment_1(rows, outdir)
    plot_experiment_2(rows, outdir)
    plot_experiment_3(rows, outdir)

    ok_rate = sum(r.correctness_ok for r in rows) / max(1, len(rows))
    print(f"Wrote {len(rows)} rows to {metrics_csv}")
    print(f"Wrote grouped summary to {summary_csv}")
    print(f"Correctness rate across all runs: {ok_rate:.3f}")
    print("Charts saved in:", outdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
