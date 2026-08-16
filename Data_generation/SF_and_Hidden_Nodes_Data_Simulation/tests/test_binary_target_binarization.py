import unittest

import numpy as np

from Data_Generation import Data_Generation_Process, lin_func


class BinaryTargetBinarizationTest(unittest.TestCase):
    def _make_generator(self, target_type="binary"):
        return Data_Generation_Process(
            beta_lower_limit=0.5,
            betta_upper_limit_values=[1.5],
            cont_noise=1.0,
            nr_nodes_values=[10],
            edge_desnity_values=[0.4],
            data_scale_values=["standardized"],
            num_samples=240,
            nonlinearities=[[(1.0, lin_func)]],
            u_ratios=[1.0],
            sf_target_selection_method="r2",
            target_type=target_type,
            seed_run_values=[7],
        )

    def test_binary_target_keeps_graph_and_observed_shapes(self):
        binary_generator = self._make_generator(target_type="binary")
        (
            binary_description,
            binary_adjacency_matrices,
            binary_weighted_matrices,
            binary_frames,
            binary_full_adjacency_matrices,
            binary_full_weighted_matrices,
            binary_full_frames,
        ) = binary_generator.large_scale_simulation(graph_type="SF")

        continuous_generator = self._make_generator(target_type="continuous")
        (
            _continuous_description,
            continuous_adjacency_matrices,
            continuous_weighted_matrices,
            continuous_frames,
            continuous_full_adjacency_matrices,
            continuous_full_weighted_matrices,
            continuous_full_frames,
        ) = continuous_generator.large_scale_simulation(graph_type="SF")

        description = binary_description.iloc[0]
        observed_frame = binary_frames[0]
        full_frame = binary_full_frames[0]
        target_column = int(description["Target_Column"])
        target_node = int(description["Target_Node"])
        observed_nodes = list(description["Observed_Nodes_Full"])
        hidden_u_nodes = list(description["Hidden_U_Nodes_Full"])
        feature_columns = list(description["Feature_Columns"])

        target_values = observed_frame.iloc[:, target_column]
        self.assertEqual(target_column, observed_frame.shape[1] - 1)
        self.assertTrue(set(target_values.unique()).issubset({0, 1}))
        self.assertEqual(set(target_values.unique()), {0, 1})
        self.assertTrue(np.issubdtype(target_values.dtype, np.integer))
        self.assertNotIn(target_column, feature_columns)

        self.assertTrue(hidden_u_nodes)
        self.assertGreater(full_frame.shape[1], observed_frame.shape[1])
        self.assertTrue(set(hidden_u_nodes).issubset(set(full_frame.columns)))
        self.assertTrue(set([1, 2]).issubset(set(full_frame.columns)))
        self.assertTrue(set(hidden_u_nodes).isdisjoint(set(observed_nodes)))
        self.assertTrue(set([1, 2]).isdisjoint(set(observed_nodes)))

        self.assertEqual(observed_frame.shape[0], 240)
        self.assertEqual(full_frame.shape[0], 240)
        self.assertEqual(continuous_frames[0].shape[0], observed_frame.shape[0])
        self.assertEqual(observed_frame.shape[1], len(observed_nodes))

        np.testing.assert_array_equal(
            binary_adjacency_matrices[0],
            continuous_adjacency_matrices[0],
        )
        np.testing.assert_array_equal(
            binary_weighted_matrices[0],
            continuous_weighted_matrices[0],
        )
        np.testing.assert_array_equal(
            binary_full_adjacency_matrices[0],
            continuous_full_adjacency_matrices[0],
        )
        np.testing.assert_array_equal(
            binary_full_weighted_matrices[0],
            continuous_full_weighted_matrices[0],
        )

        np.testing.assert_array_equal(
            full_frame.loc[:, target_node].to_numpy(),
            target_values.to_numpy(),
        )
        self.assertTrue(np.issubdtype(full_frame.loc[:, target_node].dtype, np.integer))
        self.assertIn("target_continuous", observed_frame.attrs)
        self.assertIn("target_continuous", full_frame.attrs)
        self.assertEqual(observed_frame.attrs["target_continuous"].shape[0], 240)

        positive_rate = float(description["Target_Positive_Rate"])
        self.assertGreater(positive_rate, 0.0)
        self.assertLess(positive_rate, 1.0)
        self.assertGreaterEqual(positive_rate, 0.35)
        self.assertLessEqual(positive_rate, 0.65)

        self.assertEqual(description["Target_Type"], "binary")
        self.assertEqual(description["Target_Binarization_Method"], "quantile")
        self.assertEqual(float(description["Target_Binarization_Quantile"]), 0.5)
        self.assertTrue(np.isfinite(float(description["Target_Binarization_Threshold"])))


if __name__ == "__main__":
    unittest.main()
