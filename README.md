# Huffman vs Huffman+OBST — Final Experiments
> Author: Jacob Mitchell   

## Goal
Runs our 3 experiments with MANY repeated runs and produces:
- `metrics.csv` with raw measurements for every run
- `summary.csv` with grouped means/stdev/total runtime
- PNG charts

## Conda
- `conda env create -f environment.yml`
- `conda activate huffobst`

## Files required (in same folder)
- `huffman.py`
- `obst.py`
- `experiments.py`

## Recommended runs
### Default
```bash
python experiments.py --outdir results --runs 7
```

### Bigger scaling
```bash
python experiments.py --outdir results --runs 5 --exp2_max_mb 16
```

### Change which synthetic datasets are used
Experiment 1:
```bash
python experiments.py --outdir results --runs 5 --exp1_generators uniform256,zipf128,repetitive90,english_like,repetitive99
```

Experiment 2:
```bash
python experiments.py --outdir results --runs 5 --exp2_generators uniform256,zipf128,repetitive90 --exp2_min_kb 8 --exp2_max_mb 16
```

## Dataset helper behavior
If you pass an unknown dataset name, the script will not crash.
It will fall back to a safe default dataset and label it as a fallback in outputs.

## Presentation data
- `summary.csv` for clean tables (mean/stdev)
- Key charts:
  - `exp1_compression_ratio.png`, `exp1_encode_time.png`, `exp1_total_time.png`
  - `exp2_*` plots (scaling trends)
  - `exp3_total_time.png` (end-to-end comparison)
