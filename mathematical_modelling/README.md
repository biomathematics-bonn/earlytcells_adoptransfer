# ODE models of CD4 T-cell differentiation after immunization

This repository contains the ordinary differential equation (ODE) models used to describe the population kinetics of Th0-, Th1-, and Tfh-associated CD4 T-cell states after immunization. Six alternative models compare the possible sequential arrangements of precursor, proliferating precursor, and effector cell states.

The workflow includes model simulation, parameter fitting, multistart optimization and model comparison by Akaike information criterion (AIC).

Data: see the data availability statement of the manuscript.

## Order of the scripts

| Step | Script | Purpose | Main output |
|---|---|---|---|
| 1 | `fit.py` | Fit one model variant to the absolute cell-count data | `fit.png`, `fit_summary.csv` |
| 2 | `model_selection.py` | Plot differences in AIC between the six models | `model_selection.png` |

The supporting modules are:

| Module | Purpose |
|---|---|
| `model.py` | ODE definitions, initial conditions, numerical integration, and observable mapping |
| `parameters.py` | Initial parameter values, bounds, fixed/fitted status, and parameter constraints |
| `data.py` | Loading, combining, and filtering the experimental data |
| `fit_helper.py` | Residual calculation, local optimization, multistart fitting, and confidence-interval utilities |
| `diagnostics.py` | Plot ordering, labels, colors, and logarithmic fit plotting |
| `plot_sim.py` | Parameter sampling, ensemble simulation, and uncertainty plots with cached trajectories |
| `plot_error.py` | Plotting of fitted trajectories, data, and model uncertainty from cached results |

Run all commands from the repository root, i.e. the directory containing `fit.py`.

## Model variants

Each model contains parallel Th0, Th1, and Tfh differentiation branches. The variants differ in the order of the three measured states within each branch.

| Manuscript model | Code name | Differentiation order |
|---|---|---|
| (i) | `int_pre_eff` | precursor → proliferating precursor → effector |
| (ii) | `pre_int_eff` | proliferating precursor → precursor → effector |
| (iii) | `int_eff_pre` | precursor → effector → proliferating precursor |
| (iv) | `pre_eff_int` | proliferating precursor → effector → precursor |
| (v) | `eff_pre_int` | effector → proliferating precursor → precursor |
| (vi) | `eff_int_pre` | effector → precursor → proliferating precursor |

Model-specific parameter values and bounds are defined in `parameters.py`.

## Expected project layout

```text
ode_model/
├── README.md
├── data/
│   └── ki67_analysis_detail_absolute_counts_summary.csv
├── model.py
├── parameters.py
├── data.py
├── fit_helper.py
├── diagnostics.py
├── fit.py
├── plot_sim.py
├── plot_error.py
├── model_selection.py
├── ode_fitting.yml
└── fit_outputs/          # created automatically
```

## Parameter fitting

Fit a single model from its parameter defaults with:

```bash
python fit.py --model-variant int_pre_eff
```

For multistart fitting, specify the number of local fits and a random seed.

```bash
python fit.py \
    --model-variant int_pre_eff \
    --multi-start 50 \
    --seed 1
```

Available model names are:

```text
int_pre_eff
pre_int_eff
int_eff_pre
pre_eff_int
eff_pre_int
eff_int_pre
```

The default fitting method is `leastsq`. A different method supported by `lmfit` can be selected using `--method`.

Each fit creates a timestamped directory:

```text
fit_outputs/<timestamp>_<model_variant>/
├── fit.png
└── fit_summary.csv
```

For a multistart fit, the directory name additionally contains `_multistart<N>`.

`fit.png` shows the best-fit trajectories and experimental means with SEM error bars on logarithmic y-axes. `fit_summary.csv` contains the fitting method, number of function evaluations, number of data points and fitted variables, chi-square, AIC, BIC, and the final parameter values. The `remark` column reports either the initial value used for a fitted parameter or `fixed`.

## Model selection

The AIC values used in `model_selection.py` correspond to the six model variants. After updating `MODEL_AIC_VALUES` with the final fit results, create the model-selection plot with:

```bash
python model_selection.py
```

The horizontal bars show the difference in AIC relative to model (i):

```text
Delta AIC = AIC(model) - AIC(model i)
```

The plot is saved as `model_selection.png`.

## Paper figures and tables

The `fit.png` file generated for model (i), `int_pre_eff`, corresponds to Figure 3d. The model-comparison plot `model_selection.png` corresponds to Figure 3e. Supplementary Table 3 is based on `fit_summary.csv` from model (i).

## Environment

The provided conda environment contains Python 3.12 and the principal dependencies `numpy`, `pandas`, `scipy`, `lmfit`, and `matplotlib`.

Create and activate the environment with:

```bash
conda env create --name ode_fitting -f ode_fitting.yml
conda activate ode_fitting
```
