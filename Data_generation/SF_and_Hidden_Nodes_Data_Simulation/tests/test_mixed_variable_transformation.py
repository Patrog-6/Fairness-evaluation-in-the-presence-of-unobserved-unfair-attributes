import unittest

import numpy as np
import pandas as pd

from Data_Generation import Data_Generation_Process, lin_func
from Run_Data_Generation import one_hot_encode_observed_frame


class MixedVariableTransformationTest(unittest.TestCase):
    def _make_generator(self):
        return Data_Generation_Process(
            beta_lower_limit=0.5,
            betta_upper_limit_values=[1.5],
            cont_noise=1.0,
            nr_nodes_values=[20],
            edge_desnity_values=[0.4],
            data_scale_values=["standardized"],
            num_samples=500,
            nonlinearities=[[(1.0, lin_func)]],
            u_ratios=[1.0],
            sf_target_selection_method="r2",
            target_type="binary",
            seed_run_values=[11],
        )

    def _first_dataset(self):
        (
            descriptions,
            _observed_adjacency_matrices,
            _observed_weighted_matrices,
            observed_frames,
            _full_adjacency_matrices,
            _full_weighted_matrices,
            full_frames,
        ) = self._make_generator().large_scale_simulation(graph_type="SF")
        return descriptions.iloc[0], observed_frames[0], full_frames[0]

    def test_mixed_types_metadata_and_observed_data_contract(self):
        description, observed_frame, full_frame = self._first_dataset()
        metadata = full_frame.attrs["node_metadata"]
        metadata_by_name = {entry["node_name"]: entry for entry in metadata}
        observed_nodes = list(description["Observed_Nodes_Full"])
        hidden_u_nodes = list(description["Hidden_U_Nodes_Full"])
        x_nodes = list(description["X_Nodes"])

        self.assertEqual(metadata_by_name["S0"]["final_type"], "categorical")
        self.assertEqual(metadata_by_name["S0"]["num_categories"], 2)
        self.assertTrue(set(full_frame.loc[:, 0].unique()).issubset({0, 1}))
        self.assertAlmostEqual(float(full_frame.loc[:, 0].mean()), 0.5, delta=0.05)
        self.assertTrue(metadata_by_name["S0"]["observed"])
        self.assertTrue(metadata_by_name["S0"]["included_in_model_input"])

        self.assertEqual(metadata_by_name["S1"]["final_type"], "categorical")
        self.assertEqual(metadata_by_name["S1"]["num_categories"], 3)
        self.assertTrue(set(full_frame.loc[:, 1].unique()).issubset({0, 1, 2}))
        self.assertFalse(metadata_by_name["S1"]["observed"])
        self.assertNotIn(1, observed_nodes)

        self.assertEqual(metadata_by_name["S2"]["final_type"], "continuous")
        self.assertEqual(metadata_by_name["S2"]["distribution_type"], "standard_normal")
        self.assertTrue(np.issubdtype(full_frame.loc[:, 2].dtype, np.floating))
        self.assertFalse(metadata_by_name["S2"]["observed"])
        self.assertNotIn(2, observed_nodes)

        u_metadata = [entry for entry in metadata if entry["role"] == "U"]
        x_metadata = [entry for entry in metadata if entry["role"] == "X"]
        self.assertEqual(len(u_metadata), len(hidden_u_nodes))
        self.assertEqual(len(x_metadata), len(x_nodes))
        self.assertEqual(
            sum(entry["final_type"] == "categorical" for entry in u_metadata),
            int(np.floor((len(hidden_u_nodes) * 0.30) + 0.5)),
        )
        self.assertEqual(
            sum(entry["final_type"] == "categorical" for entry in x_metadata),
            int(np.floor((len(x_nodes) * 0.30) + 0.5)),
        )

        categorical_cardinalities = [
            entry["num_categories"]
            for entry in u_metadata + x_metadata
            if entry["final_type"] == "categorical"
        ]
        self.assertTrue(set(categorical_cardinalities).issubset({2, 3, 4, 5}))

        continuous_distributions = {
            entry["distribution_type"]
            for entry in u_metadata + x_metadata
            if entry["final_type"] == "continuous"
        }
        self.assertGreaterEqual(len(continuous_distributions), 2)

        continuous_noise_distributions = {
            entry["sem_noise_distribution"]
            for entry in metadata
            if entry["final_type"] == "continuous"
        }
        self.assertTrue(continuous_noise_distributions.issubset({
            "gaussian",
            "gumbel",
            "cauchy",
            "laplace",
            "logistic",
            "student_t",
        }))
        self.assertGreaterEqual(len(continuous_noise_distributions), 2)
        self.assertTrue(any(
            noise_distribution != "gaussian"
            for noise_distribution in continuous_noise_distributions
        ))

        self.assertTrue(set(hidden_u_nodes).isdisjoint(set(observed_nodes)))
        self.assertTrue(set([1, 2]).isdisjoint(set(observed_nodes)))
        self.assertEqual(observed_frame.shape[1], len(observed_nodes))
        self.assertIn("raw_data", full_frame.attrs)
        self.assertEqual(full_frame.attrs["raw_data"].shape, full_frame.shape)

        encoded_observed_frame = one_hot_encode_observed_frame(observed_frame)
        self.assertIn("Y", encoded_observed_frame.columns)
        self.assertEqual(encoded_observed_frame.columns[-1], "Y")
        self.assertTrue(any(column.startswith("S0__") for column in encoded_observed_frame.columns))

    def test_mixed_type_assignment_is_seed_reproducible(self):
        _description_a, observed_frame_a, full_frame_a = self._first_dataset()
        _description_b, observed_frame_b, full_frame_b = self._first_dataset()

        self.assertEqual(
            full_frame_a.attrs["node_metadata"],
            full_frame_b.attrs["node_metadata"],
        )
        pd.testing.assert_frame_equal(full_frame_a, full_frame_b)
        pd.testing.assert_frame_equal(observed_frame_a, observed_frame_b)


if __name__ == "__main__":
    unittest.main()
