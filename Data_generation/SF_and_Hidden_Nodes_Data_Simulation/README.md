# Inserting Hidden Nodes to DAG

This repository contains a runnable synthetic iid data generator with hidden unfair mechanisms. The `results/` directory contains generated artifacts from prior runs and can be regenerated with the current defaults below.

## Files

- `Data_Generation.py`: data-generation framework.
- `Run_Data_Generation.py`: instantiates `Data_Generation.py` and runs the default simulation.
- `results/observed_datasets/*.pkl`: generated observed datasets, saved one dataset per pickle to keep large runs memory-safe.
- `results/full_data/*.csv`: full mixed-type datasets containing `S0`, hidden `S1`/`S2`, all hidden `U` nodes, all observed `X` nodes, and `Y`.
- `results/observed_data/*.csv`: model-facing mixed-type datasets containing only `S0`, observed `X` nodes, and `Y`.
- `results/observed_data_encoded/*.csv`: optional one-hot encoded model-facing datasets, created only with `--save-encoded`. One categorical DAG node may become multiple preprocessing columns here.
- `results/node_metadata/*_node_metadata.json`: per-node role, observability, final type, distribution/cardinality, SEM noise family, and model-input metadata.
- `results/SF_data_generation_descriptions.csv`: optional metadata table for each generated dataset, created only with `--save-description-csv`.
- `results/SF_data_generation_first_observed_dataset.csv`: optional first observed dataset preview, created only with `--save-preview-csv`.
- `results/SF_data_generation_run_summary.json`: summary of the run configuration and outputs.

The SEM still generates latent continuous values first. A deterministic post-processing layer then converts nodes into mixed final variable types:

- `S0`: observed binary categorical protected feature.
- `S1`: hidden 3-class categorical protected feature.
- `S2`: hidden continuous protected feature.
- hidden `U` nodes: closest-to-30% categorical and 70% continuous.
- observed `X` nodes: closest-to-30% categorical and 70% continuous.
- continuous `X`/`U` variables use varied realistic scale families such as `score_0_100`, `positive_skewed`, `age_18_70`, and `standard_normal`.
- categorical variables are integer coded from latent propensities with quantile-based thresholds.
- SEM noise for final continuous nodes is also mixed by node instead of always Gaussian. Supported continuous noise families include `gaussian`, `gumbel`, `cauchy`, `laplace`, `logistic`, and `student_t`; this is recorded as `sem_noise_distribution` in node metadata.

## Default Run

Generate a dataset bundle with the current defaults:

```powershell
python Run_Data_Generation.py
```

Default configuration:

- samples per base node: `1000`
- seed runs per dataset/causal graph configuration: `10` (`0..9`)
- base graph: scale-free (`SF`)
- base nodes: `10`, `20`
- edge density API value: `0.4`
- hidden U ratios: `0.0`, `0.5`, `1.0`, `2.0`
- hidden U roles: each hidden U node independently samples `confounder`, `mediator`, or `collider` with equal default weights
- nonlinearity: `linear_relu_50`
- output directory: `results`
- default data CSV count: `160` = 2 base-node settings x 4 U-ratios x 10 seeds x 2 data versions (`full_data` and `observed_data`)

## Re-run

Install the core dependencies:

```powershell
pip install numpy pandas networkx scipy scikit-learn
```

Then run:

```powershell
python Run_Data_Generation.py --output-dir results
```

Useful overrides:

```powershell
python Run_Data_Generation.py --samples-per-node 1000 --seed-runs 10 --base-nodes 10 20 --u-ratios 0 0.5 1 2 --nonlinearity linear_relu_50 --output-dir results
```

The current simulation protocol enforces `--seed-runs 10`, so each dataset/causal graph configuration is generated with exactly 10 different random seeds.
Use `--save-encoded` only when one-hot encoded observed CSVs are needed in addition to the default 160 data CSVs.
Use `--save-metadata-csv` only when CSV metadata is needed in addition to the default JSON metadata.
Use `--save-description-csv` or `--save-preview-csv` only when those auxiliary CSVs are needed; by default the generated CSV files are exactly the `full_data` and `observed_data` data files.

## Output Shape

The default run creates 80 dataset configurations:

```text
2 base-node settings x 4 U-ratios x 10 seeds = 80
```

Each configuration is exported as two data CSVs:

- `full_data`: includes hidden nodes.
- `observed_data`: hides `S1`, `S2`, and all `U` nodes.

So the default run creates 160 data CSVs:

```text
80 configurations x 2 data versions = 160
```

Observed dataset shapes are:

- 10 base nodes x 1000 samples/node -> observed datasets with shape `10000 x 8`
- 20 base nodes x 1000 samples/node -> observed datasets with shape `20000 x 18`

The metadata file has shape:

```text
80 x 41
```

To force one fixed sample count for every dataset, use `--samples`; it overrides `--samples-per-node`:

```powershell
python Run_Data_Generation.py --samples 10000
```
