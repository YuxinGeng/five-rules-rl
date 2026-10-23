# Learning to cooperate in a changing world: How caring about the future promotes cooperation across scales

Code accompanying the paper, including simulations and plotting scripts for
Figures 2–4 and Supplementary Figures S1–S5.

## Code structure

| Directory | Contents |
| --- | --- |
| `fig2/` |  Learning dynamics and equilibria in ont-shot games. |
| `fig3/` | Direct reciprocity, indirect reciprocity, environmental stochasticity, network reciprocity and demographic stochasticity. |
| `fig4/` | Learning over reactive behavioral policies. |
| `figS1_S2/` | Stability regions for GRIM and WSLS. |
| `figS3/` | Simulation for Leading-Eight under indirect reciprocity. |
| `figS4/` | Figure 3d network reciprocity model. |
| `figS5/` | Networked Stag-Hunt game. |
| `common/` | Shared learning, simulation, graph and plotting helpers. |

Simulation data and figures are stored in each experiment's `data/` and
`outputs/` directories.

## Requirements

Python 3.11 or newer, with the following packages:

| Package | Minimum version |
| --- | --- |
| NumPy | 1.26 |
| pandas | 2.1 |
| Matplotlib | 3.10 |
| seaborn | 0.13 |
| Numba | 0.59 |
| SciPy | 1.11 |
| SymPy | 1.12 |
| IPython | 8.0 |
| joblib | 1.3 |

Install from [requirements.txt](requirements.txt) in this directory:

```bash
python -m pip install -r requirements.txt
```

Figure 4 video generation also requires FFmpeg.
