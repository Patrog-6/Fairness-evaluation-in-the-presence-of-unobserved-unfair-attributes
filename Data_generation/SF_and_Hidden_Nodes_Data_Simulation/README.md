# Pure Synthetic Data Generation: Scale-Free DAGs with Hidden Nodes

This directory contains the pure synthetic data-generation pipeline used in the main project. It creates i.i.d. scale-free DAG datasets with an observed protected attribute, hidden protected attributes, hidden structural `U` variables, observed `X` variables, and target `Y`.

## Files

- `Data_Generation.py`: data-generation framework.
- `Run_Data_Generation.py`: instantiates `Data_Generation.py` and runs the default simulation.

When `Run_Data_Generation.py` is run, it writes generated artifacts to the selected
`--output-dir` (`results` by default):

- `observed_datasets/*.pkl`: observed datasets, saved one dataset per pickle to keep large runs memory-safe.
- `full_data/*.csv`: full mixed-type datasets containing `S0`, hidden `S1`/`S2`, hidden `U` nodes when present, observed `X` nodes, and `Y`. These are retained for ground-truth analysis, diagnostics, and fairness evaluation; hidden `S1`, `S2`, and `U` variables are not intended to be available to ordinary deployable models.
- `observed_data/*.csv`: model-facing mixed-type datasets containing only `S0`, observed `X` variables, and `Y`.
- `node_metadata/*_node_metadata.json`: per-node role, observability, final type, distribution/cardinality, SEM noise family, and model-input metadata.
- `SF_data_generation_run_summary.json`: summary of the run configuration and generated outputs.
- `observed_data_encoded/*.csv`: optional one-hot encoded model-facing datasets, created only with `--save-encoded`. One categorical DAG node may become multiple preprocessing columns here.
- `node_metadata/*_node_metadata.csv`: optional CSV node metadata, created only with `--save-metadata-csv`.
- `SF_data_generation_descriptions.csv`: optional run-level description table, created only with `--save-description-csv`.
- `SF_data_generation_first_observed_dataset.csv` and `SF_data_generation_first_full_dataset.csv`: optional first-dataset previews, created only with `--save-preview-csv`.

The submitted repository retains the pre-generated `full_data` CSVs from the default 80-configuration batch under `results/generated_160_csv_10_seeds/full_data/`. Other paths above describe artifacts produced by the generator when it is run and should not be read as a list of files that are all currently committed.

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
- U-ratio definition: `n_U = round(r_U * n_base)`, so 10 base nodes with U ratio `0.5` produce 5 hidden U nodes, and 10 base nodes with U ratio `2.0` produce 20 hidden U nodes
- hidden U roles: each hidden U node independently samples `confounder`, `mediator`, or `collider` with equal default weights
- nonlinearity: `linear_relu_50`
- output directory: `results`
- default generated data CSV count: `160` = 2 base-node settings x 4 U-ratios x 10 seeds x 2 data versions (`full_data` and `observed_data`)

## Re-run

Install the core dependencies:

```powershell
pip install -r requirements.txt
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
Use `--save-full-debug` only when full hidden-node pickle frames and full causal matrices are needed.

## Output Shape

The default run creates 80 dataset configurations:

```text
2 base-node settings x 4 U-ratios x 10 seeds = 80
```

Each configuration is exported as two data CSVs:

- `full_data`: contains `S0`, hidden `S1`/`S2`, hidden `U` nodes when present, observed `X` nodes, and `Y`.
- `observed_data`: contains only the model-facing variables, `S0`, observed `X` variables, and `Y`.

So the default run creates 160 data CSVs:

```text
80 configurations x 2 data versions = 160
```

Observed dataset shapes are:

- 10 base nodes x 1000 samples/node -> observed datasets with shape `10000 x 8`
- 20 base nodes x 1000 samples/node -> observed datasets with shape `20000 x 18`

Filename suffixes such as `_8_nodes_full_data.csv` and `_18_nodes_full_data.csv` refer to the number of model-facing observed nodes, not to the total number of columns or nodes in the corresponding `full_data` file.

When `--save-description-csv` is enabled, the run-level description table has 80 rows under the default configuration (`2 base-node settings x 4 U-ratios x 10 seeds = 80 configurations`). Its current schema contains 41 columns.

To force one fixed sample count for every dataset, use `--samples`; it overrides `--samples-per-node`:

```powershell
python Run_Data_Generation.py --samples 10000
```
