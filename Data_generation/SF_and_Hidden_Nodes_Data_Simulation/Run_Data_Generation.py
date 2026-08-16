import argparse
import importlib.util
import json
import pickle
import sys
from pathlib import Path

import pandas as pd


FRAMEWORK_DIR = Path(__file__).resolve().parent
MODULE_PATH = FRAMEWORK_DIR / "Data_Generation.py"
DEFAULT_OUTPUT_DIR = FRAMEWORK_DIR / "results"
DEFAULT_BASE_NODES = [10, 20]
DEFAULT_U_RATIOS = [0.0, 0.5, 1.0, 2.0]
DEFAULT_SEED_RUNS = 10
DEFAULT_SAMPLES_PER_NODE = 1000


def load_data_generation_module():
    """Load the hidden-mechanism generator from this folder."""
    spec = importlib.util.spec_from_file_location("data_generation", MODULE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_nonlinearity(module, name):
    if name == "linear":
        return [(1.0, module.lin_func)]
    if name == "linear_relu_50":
        return [(0.5, module.lin_func), (0.5, module.relu_func)]
    if name == "linear_30_relu_70":
        return [(0.3, module.lin_func), (0.7, module.relu_func)]
    if name == "linear_10_relu_90":
        return [(0.1, module.lin_func), (0.9, module.relu_func)]
    raise ValueError(f"Unsupported nonlinearity: {name}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Instantiate Data_Generation.py and generate simulated iid data."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Fixed samples per dataset. Overrides --samples-per-node when set.",
    )
    parser.add_argument(
        "--samples-per-node",
        type=int,
        default=DEFAULT_SAMPLES_PER_NODE,
        help="Samples per base node when --samples is not set.",
    )
    parser.add_argument(
        "--seed-runs",
        type=int,
        default=DEFAULT_SEED_RUNS,
        help=(
            "Number of random seeds to run for each dataset/causal graph "
            "configuration. The current simulation protocol requires 10."
        ),
    )
    parser.add_argument("--base-nodes", type=int, nargs="+", default=DEFAULT_BASE_NODES)
    parser.add_argument("--edge-density", type=float, default=0.4)
    parser.add_argument(
        "--u-ratios",
        "--u-ratio",
        dest="u_ratios",
        type=float,
        nargs="+",
        default=DEFAULT_U_RATIOS,
    )
    parser.add_argument("--beta-lower", type=float, default=0.5)
    parser.add_argument("--beta-upper", type=float, default=2.0)
    parser.add_argument("--noise", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--nonlinearity",
        choices=["linear", "linear_relu_50", "linear_30_relu_70", "linear_10_relu_90"],
        default="linear_relu_50",
    )
    parser.add_argument(
        "--save-full-debug",
        action="store_true",
        help="Also save full hidden-node frames and full causal matrices.",
    )
    parser.add_argument(
        "--save-encoded",
        action="store_true",
        help="Also save optional one-hot encoded observed_data CSVs.",
    )
    parser.add_argument(
        "--save-metadata-csv",
        action="store_true",
        help="Also save node metadata as CSV. JSON metadata is always saved.",
    )
    parser.add_argument(
        "--save-description-csv",
        action="store_true",
        help="Also save the run description table as CSV.",
    )
    parser.add_argument(
        "--save-preview-csv",
        action="store_true",
        help="Also save first full/observed dataset preview CSVs.",
    )
    args = parser.parse_args()
    if args.seed_runs != DEFAULT_SEED_RUNS:
        parser.error(
            f"--seed-runs must be {DEFAULT_SEED_RUNS}; the current simulation "
            "protocol requires exactly 10 random seeds per dataset/causal graph "
            "configuration."
        )
    return args


def samples_for_base_nodes(args, base_nodes):
    if args.samples is not None:
        return args.samples
    return base_nodes * args.samples_per_node


def format_ratio_for_filename(u_ratio):
    return str(float(u_ratio)).replace(".", "p")


def dataset_pickle_name(description, nonlinear_pattern):
    seed_run = int(description["Seed_Run"])
    base_nodes = int(description["Number_Base_Nodes_SF"])
    observed_nodes = int(description["Number_Nodes"])
    u_ratio = format_ratio_for_filename(description["U_Ratio"])
    return (
        f"SF_Large_Sample_Size_Dataset_{nonlinear_pattern}_"
        f"{base_nodes}_base_nodes_u_{u_ratio}_seed_{seed_run}_{observed_nodes}_nodes.pkl"
    )


def save_single_observed_dataset(path, description, causal_matrix, weighted_matrix, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            [
                description.values.tolist(),
                causal_matrix,
                weighted_matrix,
                frame,
            ],
            f,
        )


def save_single_full_debug_dataset(path, causal_matrix, weighted_matrix, frame):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(
            {
                "full_causal_matrix": causal_matrix,
                "full_weighted_causal_matrix": weighted_matrix,
                "full_frame": frame,
            },
            f,
        )


def named_full_frame(frame):
    metadata = frame.attrs.get("node_metadata", [])
    if len(metadata) == 0:
        return frame.copy()

    name_by_node = {
        int(entry["node_id"]): entry["node_name"]
        for entry in metadata
    }
    named_frame = frame.copy()
    named_frame.columns = [
        name_by_node.get(int(column), str(column))
        for column in named_frame.columns
    ]
    return named_frame


def named_observed_frame(frame):
    observed_metadata = frame.attrs.get("observed_node_metadata", [])
    if len(observed_metadata) == 0:
        return frame.copy()

    metadata_by_column = {
        int(entry["observed_column"]): entry
        for entry in observed_metadata
    }
    named_frame = frame.copy()
    named_frame.columns = [
        metadata_by_column[column]["node_name"]
        for column in range(0, named_frame.shape[1])
    ]
    return named_frame


def one_hot_encode_observed_frame(frame):
    observed_metadata = frame.attrs.get("observed_node_metadata", [])
    if len(observed_metadata) == 0:
        return frame.copy()

    named_frame = named_observed_frame(frame)
    feature_frames = []
    target_frame = None

    for entry in sorted(observed_metadata, key=lambda item: int(item["observed_column"])):
        column_name = entry["node_name"]
        series = named_frame[column_name]
        if entry["role"] == "Y":
            target_frame = series.reset_index(drop=True).to_frame(column_name)
            continue

        if entry["final_type"] == "categorical":
            categories = list(range(0, int(entry["num_categories"])))
            categorical_series = pd.Categorical(series.astype(int), categories=categories)
            encoded = pd.get_dummies(
                categorical_series,
                prefix=column_name,
                prefix_sep="__",
                dtype=int,
            )
            feature_frames.append(encoded.reset_index(drop=True))
        else:
            feature_frames.append(series.reset_index(drop=True).to_frame(column_name))

    if target_frame is not None:
        feature_frames.append(target_frame)
    return pd.concat(feature_frames, axis=1)


def save_mixed_dataset_artifacts(output_dir, pickle_name, observed_frame, full_frame,
                                 save_encoded=False, save_metadata_csv=False):
    stem = Path(pickle_name).stem
    full_data_dir = output_dir / "full_data"
    observed_data_dir = output_dir / "observed_data"
    metadata_dir = output_dir / "node_metadata"
    directories = [full_data_dir, observed_data_dir, metadata_dir]
    if save_encoded:
        encoded_data_dir = output_dir / "observed_data_encoded"
        directories.append(encoded_data_dir)
    for path in directories:
        path.mkdir(parents=True, exist_ok=True)

    full_data_path = full_data_dir / f"{stem}_full_data.csv"
    observed_data_path = observed_data_dir / f"{stem}_observed_data.csv"
    metadata_json_path = metadata_dir / f"{stem}_node_metadata.json"

    named_full_frame(full_frame).to_csv(full_data_path, index=False)
    named_observed_frame(observed_frame).to_csv(observed_data_path, index=False)

    metadata = full_frame.attrs.get("node_metadata", [])
    with open(metadata_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    artifacts = {
        "full_data_csv": str(full_data_path.relative_to(output_dir)),
        "observed_data_csv": str(observed_data_path.relative_to(output_dir)),
        "node_metadata_json": str(metadata_json_path.relative_to(output_dir)),
    }
    if save_encoded:
        encoded_data_path = encoded_data_dir / f"{stem}_observed_data_encoded.csv"
        one_hot_encode_observed_frame(observed_frame).to_csv(encoded_data_path, index=False)
        artifacts["observed_data_encoded_csv"] = str(encoded_data_path.relative_to(output_dir))
    if save_metadata_csv:
        metadata_csv_path = metadata_dir / f"{stem}_node_metadata.csv"
        pd.DataFrame(metadata).to_csv(metadata_csv_path, index=False)
        artifacts["node_metadata_csv"] = str(metadata_csv_path.relative_to(output_dir))
    return artifacts


def main():
    args = parse_args()
    module = load_data_generation_module()
    causal_transformation = build_nonlinearity(module, args.nonlinearity)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_dir = output_dir / "observed_datasets"
    full_debug_output_dir = output_dir / "full_debug_datasets"

    print("Starting Data_Generation simulation...")
    print(f"Base nodes: {args.base_nodes}")
    print(f"Samples per base node: {args.samples_per_node}")
    if args.samples is not None:
        print(f"Fixed samples per dataset override: {args.samples}")
    print(f"Hidden U ratios: {args.u_ratios}")
    print(f"Seed runs: 0..{args.seed_runs - 1}")
    print(f"Output directory: {output_dir}")

    all_descriptions = []
    generated_pickles = []
    generated_full_debug_pickles = []
    generated_full_data_csvs = []
    generated_observed_data_csvs = []
    generated_encoded_data_csvs = []
    generated_metadata_jsons = []
    generated_metadata_csvs = []
    generated_sample_sizes = {}
    nonlinear_pattern = None
    first_observed_frame = None
    first_full_frame = None
    first_encoded_frame = None
    first_observed_shape = None
    num_generated_datasets = 0

    for base_nodes in args.base_nodes:
        num_samples = samples_for_base_nodes(args=args, base_nodes=base_nodes)
        generated_sample_sizes[str(base_nodes)] = num_samples
        print(f"Generating base nodes={base_nodes}, samples={num_samples}...")

        for seed_run in range(args.seed_runs):
            dgp = module.Data_Generation_Process(
                beta_lower_limit=args.beta_lower,
                betta_upper_limit_values=[args.beta_upper],
                cont_noise=args.noise,
                nr_nodes_values=[base_nodes],
                edge_desnity_values=[args.edge_density],
                data_scale_values=["standardized"],
                num_samples=num_samples,
                nonlinearities=[causal_transformation],
                u_ratios=args.u_ratios,
                sf_target_selection_method="r2",
                seed_run_values=[seed_run],
            )

            (
                current_descriptions,
                current_observed_causal_matrices,
                current_observed_weighted_matrices,
                current_observed_frames,
                current_full_causal_matrices,
                current_full_weighted_matrices,
                current_full_frames,
            ) = dgp.large_scale_simulation(graph_type="SF")

            if nonlinear_pattern is None:
                nonlinear_pattern = dgp._transformation_name(causal_transformation)

            all_descriptions.append(current_descriptions)

            for frame_index in range(current_descriptions.shape[0]):
                description = current_descriptions.iloc[frame_index]
                pickle_name = dataset_pickle_name(
                    description=description,
                    nonlinear_pattern=nonlinear_pattern,
                )
                pickle_path = dataset_output_dir / pickle_name
                save_single_observed_dataset(
                    path=pickle_path,
                    description=description,
                    causal_matrix=current_observed_causal_matrices[frame_index],
                    weighted_matrix=current_observed_weighted_matrices[frame_index],
                    frame=current_observed_frames[frame_index],
                )
                generated_pickles.append(str(pickle_path.relative_to(output_dir)))

                mixed_artifacts = save_mixed_dataset_artifacts(
                    output_dir=output_dir,
                    pickle_name=pickle_name,
                    observed_frame=current_observed_frames[frame_index],
                    full_frame=current_full_frames[frame_index],
                    save_encoded=args.save_encoded,
                    save_metadata_csv=args.save_metadata_csv,
                )
                generated_full_data_csvs.append(mixed_artifacts["full_data_csv"])
                generated_observed_data_csvs.append(mixed_artifacts["observed_data_csv"])
                generated_metadata_jsons.append(mixed_artifacts["node_metadata_json"])
                if args.save_encoded:
                    generated_encoded_data_csvs.append(mixed_artifacts["observed_data_encoded_csv"])
                if args.save_metadata_csv:
                    generated_metadata_csvs.append(mixed_artifacts["node_metadata_csv"])

                if first_observed_frame is None:
                    first_observed_frame = current_observed_frames[frame_index]
                    first_full_frame = current_full_frames[frame_index]
                    if args.save_encoded:
                        first_encoded_frame = one_hot_encode_observed_frame(first_observed_frame)
                    first_observed_shape = list(first_observed_frame.shape)

                if args.save_full_debug:
                    full_debug_path = full_debug_output_dir / pickle_name
                    save_single_full_debug_dataset(
                        path=full_debug_path,
                        causal_matrix=current_full_causal_matrices[frame_index],
                        weighted_matrix=current_full_weighted_matrices[frame_index],
                        frame=current_full_frames[frame_index],
                    )
                    generated_full_debug_pickles.append(str(full_debug_path.relative_to(output_dir)))

                num_generated_datasets += 1

            print(f"  Seed {seed_run} complete ({num_generated_datasets} datasets total).")

    descriptions = pd.concat(all_descriptions, ignore_index=True)

    summary = {
        "samples": args.samples,
        "samples_per_node": args.samples_per_node,
        "samples_by_base_nodes": generated_sample_sizes,
        "seed_runs": args.seed_runs,
        "base_nodes": args.base_nodes,
        "edge_density": args.edge_density,
        "u_ratios": args.u_ratios,
        "nonlinearity": args.nonlinearity,
        "nonlinear_pattern": nonlinear_pattern,
        "num_generated_datasets": num_generated_datasets,
        "first_observed_shape": first_observed_shape,
        "observed_dataset_dir": dataset_output_dir.name,
        "observed_dataset_pickles": generated_pickles,
        "full_data_csv_dir": "full_data",
        "full_data_csvs": generated_full_data_csvs,
        "observed_data_csv_dir": "observed_data",
        "observed_data_csvs": generated_observed_data_csvs,
        "node_metadata_dir": "node_metadata",
        "node_metadata_jsons": generated_metadata_jsons,
        "save_full_debug": bool(args.save_full_debug),
        "save_encoded": bool(args.save_encoded),
        "save_metadata_csv": bool(args.save_metadata_csv),
        "save_description_csv": bool(args.save_description_csv),
        "save_preview_csv": bool(args.save_preview_csv),
        "data_csv_count": len(generated_full_data_csvs) + len(generated_observed_data_csvs),
    }
    if args.save_description_csv:
        descriptions_path = output_dir / "SF_data_generation_descriptions.csv"
        descriptions.to_csv(descriptions_path, index=False)
        summary["description_csv"] = descriptions_path.name
    if args.save_preview_csv:
        first_dataset_path = output_dir / "SF_data_generation_first_observed_dataset.csv"
        named_observed_frame(first_observed_frame).to_csv(first_dataset_path, index=False)
        first_full_dataset_path = output_dir / "SF_data_generation_first_full_dataset.csv"
        named_full_frame(first_full_frame).to_csv(first_full_dataset_path, index=False)
        summary["first_observed_dataset_csv"] = first_dataset_path.name
        summary["first_full_dataset_csv"] = first_full_dataset_path.name
    if args.save_encoded:
        if args.save_preview_csv:
            first_encoded_dataset_path = output_dir / "SF_data_generation_first_observed_dataset_encoded.csv"
            first_encoded_frame.to_csv(first_encoded_dataset_path, index=False)
            summary["first_observed_dataset_encoded_csv"] = first_encoded_dataset_path.name
        summary["observed_data_encoded_csv_dir"] = "observed_data_encoded"
        summary["observed_data_encoded_csvs"] = generated_encoded_data_csvs
    if args.save_metadata_csv:
        summary["node_metadata_csvs"] = generated_metadata_csvs
    if args.save_full_debug:
        summary["full_debug_dataset_dir"] = full_debug_output_dir.name
        summary["full_debug_dataset_pickles"] = generated_full_debug_pickles

    summary_path = output_dir / "SF_data_generation_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Simulation finished.")
    print(f"Generated datasets: {num_generated_datasets}")
    print(f"First observed dataset shape: {first_observed_shape}")
    print(f"Run summary JSON: {summary_path}")
    print(f"Observed dataset pickle directory: {dataset_output_dir}")
    print(f"Full mixed-data CSV directory: {output_dir / 'full_data'}")
    print(f"Observed mixed-data CSV directory: {output_dir / 'observed_data'}")
    print(f"Node metadata directory: {output_dir / 'node_metadata'}")
    if args.save_description_csv:
        print(f"Descriptions CSV: {descriptions_path}")


if __name__ == "__main__":
    main()
