# Two competing ageing channels of lithium iron phosphate cathodes

Code and data for the manuscript *Two competing ageing channels of lithium
iron phosphate cathodes* (submitted to Nature Communications).

## Repository layout

```
.
├── LICENSE                Apache License 2.0
├── README.md              This file
├── requirements.txt       Python dependencies
├── verify_claims.py       Numeric verification suite
├── Code_File_Descriptions_EN.txt   Per-file roles (English)
├── 代码文件说明_中文.txt            Per-file roles (中文)
├── src/                   Computation and figure modules (19 files)
└── data/                  Results data and figure source data
    ├── data_laws.txt            Scaling-relation values
    ├── data_metrics.txt         Model metrics and distribution shift
    ├── data_severson.txt        Severson external validation
    └── figure_data/             Per-figure source data (CSV)
```

## Computation modules (src/)

| File | Role |
|---|---|
| lfp_micro.py | Electrode microstructure reconstruction (RVE, packing, graph) |
| lfp_dataset.py | Coupled degradation simulator and synthetic cell population |
| laws.py | Closed-form scaling relations and their provenance checks |
| model.py | Physics-guided graph-transformer network (microscale + macroscale) |
| run_experiments.py | Training, baselines, and distribution-shift protocols |
| analysis.py | Permutation / attention attribution analysis |
| micro_stats.py | As-built network statistics (Supplementary Table 1) |
| probe_struct.py | Severson .mat format parsing |
| severson_validate.py / severson_b1.py / severson_b3.py / severson_struct_validate.py | External validation on the Severson et al. dataset |

## Figure modules (src/)

| File | Role |
|---|---|
| fig_nc.py | Composite main figures (Fig 1–7) |
| fig_laws.py | Scaling-relation figures (Fig 2–4) |
| fig_model.py | Model-performance figures (Fig 5) |
| fig_results.py | Shared result panels |
| fig_severson.py | Severson validation figures |
| fig_science.py | Scientific illustrations (microstructure / graph abstraction) |
| fig_attention_sup.py | Attention-profile supplementary figure |

## Data (data/)

- `data_laws.txt` — numeric values of the 11 scaling relations.
- `data_metrics.txt` — per-model R² and distribution-shift metrics.
- `data_severson.txt` — Severson external-validation results.
- `data/figure_data/` — per-figure source data (CSV) that reproduces
  Figures 1–7 and Supplementary Fig. 1.

## Verification

`verify_claims.py` checks every numeric claim in the manuscript against the
produced results (run after `run_experiments.py` / `analysis.py` /
`micro_stats.py`). It prints PASS/FAIL per claim.

## Dependencies

Python 3.10+, numpy, scipy, matplotlib, torch, scikit-learn, pillow, h5py
(CPU is sufficient).

## Reproducibility

Random seeds are fixed throughout. Run order: `lfp_micro.py` →
`lfp_dataset.py` → `laws.py` → `run_experiments.py` → `analysis.py` →
`micro_stats.py` → `verify_claims.py`; the `fig_*.py` scripts run after the
results are produced.
