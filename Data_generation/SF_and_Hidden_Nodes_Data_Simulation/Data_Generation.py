import pandas as pd
import networkx as nx
import numpy as np
import pickle
from math import ceil, erf, sqrt

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - numpy fallback keeps the generator usable.
    scipy_stats = None

try:
    from sklearn.linear_model import LinearRegression
except ImportError:  # pragma: no cover - numpy fallback keeps the generator usable.
    LinearRegression = None


def lin_func(x):
    """Linear causal transformation."""
    return x


def relu_func(x):
    """ReLU causal transformation."""
    return np.maximum(0, x)


def _predict_with_linear_regression(X, y):
    """Fit a linear regression with intercept and return fitted values."""
    if LinearRegression is not None:
        model = LinearRegression(fit_intercept=True)
        model.fit(X, y)
        return model.predict(X)

    design = np.column_stack([np.ones(X.shape[0]), X])
    coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    return design @ coef


def compute_r2_scores(data):
    """
    Compute R2 for each variable by regressing it on all other variables.

    data shape: (n_samples, n_nodes)
    """
    data = np.asarray(data)
    if data.ndim != 2:
        raise ValueError('data must be a 2D array')

    n_nodes = data.shape[1]
    r2_scores = np.zeros(n_nodes)
    if n_nodes <= 1:
        return r2_scores

    for t in range(n_nodes):
        y = data[:, t]
        X = np.delete(data, t, axis=1)
        y_var = np.var(y)
        if y_var == 0:
            r2_scores[t] = 0.0
            continue

        y_pred = _predict_with_linear_regression(X, y)
        residuals = y - y_pred
        r2_scores[t] = 1.0 - np.var(residuals) / y_var

    return r2_scores


def r2_probabilities_for_candidates(r2_scores, candidate_nodes):
    """
    Convert candidate-node R2 scores into sampling probabilities.

    candidate_nodes should exclude S, hidden U nodes, and invalid outcomes.
    """
    candidate_nodes = list(candidate_nodes)
    if len(candidate_nodes) == 0:
        raise ValueError('candidate_nodes must contain at least one node')

    candidate_scores = np.array([r2_scores[node] for node in candidate_nodes], dtype=float)
    candidate_scores = np.nan_to_num(candidate_scores, nan=0.0, posinf=0.0, neginf=0.0)
    candidate_scores = np.maximum(candidate_scores, 0.0)
    total_score = np.sum(candidate_scores)

    if total_score <= 0.0:
        return np.full(len(candidate_nodes), 1.0 / len(candidate_nodes))

    return candidate_scores / total_score


def select_target_by_r2(data, candidate_nodes):
    """
    Sample Y from candidate nodes with probabilities proportional to R2 scores.

    candidate_nodes should exclude S, hidden U nodes, and invalid outcomes.
    """
    candidate_nodes = list(candidate_nodes)
    if len(candidate_nodes) == 0:
        raise ValueError('candidate_nodes must contain at least one node')

    r2_scores = compute_r2_scores(data)
    probabilities = r2_probabilities_for_candidates(
        r2_scores=r2_scores,
        candidate_nodes=candidate_nodes
    )
    selected_node = np.random.choice(candidate_nodes, p=probabilities)
    return int(selected_node), r2_scores


def _binarize_target_values(
    y,
    method="quantile",
    quantile=0.5,
    threshold=None,
    positive_rule="greater_equal",
):
    """
    Convert a continuous latent target score into a binary 0/1 target.

    Parameters
    ----------
    y : array-like
        Continuous latent target values.
    method : {"quantile", "median", "zero", "threshold"}
        Binarization rule.
    quantile : float
        Quantile used when method is "quantile" or "median".
    threshold : float or None
        Explicit threshold used when method is "threshold".
    positive_rule : {"greater_equal", "greater"}
        Whether values equal to the threshold should be assigned to class 1.

    Returns
    -------
    y_binary : np.ndarray
        Integer 0/1 target vector.
    threshold_used : float
        Threshold used for binarization.
    positive_rate : float
        Mean of y_binary.
    """
    y = np.asarray(y, dtype=float)

    if y.ndim != 1:
        raise ValueError("Target vector y must be one-dimensional.")

    if np.any(~np.isfinite(y)):
        raise ValueError("Target vector contains NaN or infinite values.")

    method_key = str(method).lower()

    if method_key == "median":
        threshold_used = float(np.quantile(y, 0.5))
    elif method_key == "quantile":
        if quantile < 0.0 or quantile > 1.0:
            raise ValueError("quantile must be between 0 and 1.")
        threshold_used = float(np.quantile(y, quantile))
    elif method_key == "zero":
        threshold_used = 0.0
    elif method_key == "threshold":
        if threshold is None:
            raise ValueError("threshold must be provided when method='threshold'.")
        threshold_used = float(threshold)
    else:
        raise ValueError(
            "method must be one of {'quantile', 'median', 'zero', 'threshold'}."
        )

    if positive_rule == "greater_equal":
        y_binary = (y >= threshold_used).astype(np.int64)
    elif positive_rule == "greater":
        y_binary = (y > threshold_used).astype(np.int64)
    else:
        raise ValueError("positive_rule must be 'greater_equal' or 'greater'.")

    positive_rate = float(np.mean(y_binary))

    if len(np.unique(y_binary)) < 2:
        raise ValueError(
            "Binarization produced only one class. "
            f"method={method_key}, threshold={threshold_used}, "
            f"positive_rate={positive_rate}."
        )

    return y_binary, threshold_used, positive_rate


CONTINUOUS_DISTRIBUTION_TYPES = [
    "score_0_100",
    "signed_minus10_10",
    "standard_normal",
    "positive_skewed",
    "proportion_0_1",
    "experience_0_40",
    "age_18_70",
    "workload_0_80",
]

SEM_NOISE_DISTRIBUTION_TYPES = [
    "gaussian",
    "gumbel",
    "cauchy",
    "laplace",
    "logistic",
    "student_t",
]


def _stable_seed_from_values(*values):
    """Create a deterministic 32-bit seed without relying on Python hash state."""
    seed = 2166136261
    for value in values:
        for character in str(value):
            seed ^= ord(character)
            seed = (seed * 16777619) % (2 ** 32)
        seed ^= 255
        seed = (seed * 16777619) % (2 ** 32)
    return int(seed)


def _closest_integer_count(total, ratio):
    """Return the closest integer allocation for total * ratio."""
    if total <= 0:
        return 0
    return min(total, max(0, int(np.floor((float(total) * float(ratio)) + 0.5))))


def _normalise_probabilities(probabilities):
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or probabilities.size == 0:
        raise ValueError("probabilities must be a non-empty one-dimensional array.")
    if np.any(probabilities < 0.0):
        raise ValueError("probabilities must be non-negative.")
    probability_sum = float(np.sum(probabilities))
    if probability_sum <= 0.0:
        raise ValueError("At least one probability must be positive.")
    return probabilities / probability_sum


def _allocate_counts(total, probabilities):
    """Allocate integer counts using largest remainders."""
    probabilities = _normalise_probabilities(probabilities)
    if total <= 0:
        return [0 for _ in probabilities]

    expected = probabilities * int(total)
    counts = np.floor(expected).astype(int)
    remaining = int(total) - int(np.sum(counts))
    if remaining > 0:
        fractional_order = np.argsort(-(expected - counts), kind="mergesort")
        for index in fractional_order[:remaining]:
            counts[index] += 1
    return [int(count) for count in counts]


def standardize_vector(x):
    """Convert a vector to z-scores, using zeros for constant vectors."""
    x = np.asarray(x, dtype=float)
    mean = float(np.mean(x))
    std = float(np.std(x))
    if not np.isfinite(std) or std <= 0.0:
        return np.zeros_like(x, dtype=float)
    return (x - mean) / std


def normal_cdf_transform(z):
    """Apply the standard normal CDF to a vector."""
    z = np.asarray(z, dtype=float)
    vectorized_erf = np.vectorize(erf, otypes=[float])
    return 0.5 * (1.0 + vectorized_erf(z / sqrt(2.0)))


def transform_continuous_node(raw_values, distribution_type, rng=None):
    """
    Transform one latent continuous node into a realistic numeric scale.

    rng is accepted for API consistency with the categorical transformer and for
    future distribution families that may need deterministic random parameters.
    """
    del rng
    z = standardize_vector(raw_values)
    probability_scale = normal_cdf_transform(z)

    if distribution_type == "score_0_100":
        return 100.0 * probability_scale
    if distribution_type == "signed_minus10_10":
        return (20.0 * probability_scale) - 10.0
    if distribution_type == "standard_normal":
        return z
    if distribution_type == "positive_skewed":
        return np.exp(1.0 + (0.75 * np.clip(z, -4.0, 4.0)))
    if distribution_type == "proportion_0_1":
        return probability_scale
    if distribution_type == "experience_0_40":
        return 40.0 * probability_scale
    if distribution_type == "age_18_70":
        return 18.0 + (52.0 * probability_scale)
    if distribution_type == "workload_0_80":
        return 80.0 * probability_scale

    raise ValueError(f"Unsupported continuous distribution type: {distribution_type}")


def _standardize_noise_sample(noise_values, distribution_type):
    noise_values = np.asarray(noise_values, dtype=float)
    noise_values = np.nan_to_num(noise_values, nan=0.0, posinf=0.0, neginf=0.0)
    if distribution_type in {"cauchy", "student_t"}:
        noise_values = np.clip(noise_values, -25.0, 25.0)
    return standardize_vector(noise_values)


def _sample_noise_with_scipy(distribution_type, size):
    random_state = np.random.mtrand._rand
    if distribution_type == "gaussian":
        return scipy_stats.norm.rvs(size=size, random_state=random_state)
    if distribution_type == "gumbel":
        return scipy_stats.gumbel_r.rvs(size=size, random_state=random_state)
    if distribution_type == "cauchy":
        return scipy_stats.cauchy.rvs(size=size, random_state=random_state)
    if distribution_type == "laplace":
        return scipy_stats.laplace.rvs(size=size, random_state=random_state)
    if distribution_type == "logistic":
        return scipy_stats.logistic.rvs(size=size, random_state=random_state)
    if distribution_type == "student_t":
        return scipy_stats.t.rvs(df=3.0, size=size, random_state=random_state)
    raise ValueError(f"Unsupported SEM noise distribution: {distribution_type}")


def _sample_noise_with_numpy(distribution_type, size):
    if distribution_type == "gaussian":
        return np.random.normal(size=size)
    if distribution_type == "gumbel":
        return np.random.gumbel(size=size)
    if distribution_type == "cauchy":
        return np.random.standard_cauchy(size=size)
    if distribution_type == "laplace":
        return np.random.laplace(size=size)
    if distribution_type == "logistic":
        return np.random.logistic(size=size)
    if distribution_type == "student_t":
        return np.random.standard_t(df=3.0, size=size)
    raise ValueError(f"Unsupported SEM noise distribution: {distribution_type}")


def sample_sem_noise(distribution_type, scale, size):
    """Draw SEM noise from a continuous family and normalize it to the scale."""
    distribution_type = str(distribution_type).lower()
    if distribution_type not in SEM_NOISE_DISTRIBUTION_TYPES:
        raise ValueError(f"Unsupported SEM noise distribution: {distribution_type}")

    if scipy_stats is not None:
        noise_values = _sample_noise_with_scipy(distribution_type, size=size)
    else:
        noise_values = _sample_noise_with_numpy(distribution_type, size=size)
    return float(scale) * _standardize_noise_sample(
        noise_values=noise_values,
        distribution_type=distribution_type,
    )


def default_class_probabilities(num_categories):
    if num_categories == 2:
        return [0.5, 0.5]
    if num_categories == 3:
        return [0.5, 0.3, 0.2]
    if num_categories == 4:
        return [0.4, 0.3, 0.2, 0.1]
    if num_categories == 5:
        return [0.35, 0.25, 0.2, 0.15, 0.05]
    if num_categories < 2:
        raise ValueError("Categorical nodes must have at least two categories.")
    return [1.0 / num_categories for _ in range(num_categories)]


def _discretize_by_rank(raw_values, class_probabilities):
    raw_values = np.asarray(raw_values, dtype=float)
    class_probabilities = _normalise_probabilities(class_probabilities)
    categories = np.zeros(raw_values.shape[0], dtype=np.int64)
    counts = _allocate_counts(raw_values.shape[0], class_probabilities)
    sorted_indices = np.argsort(raw_values, kind="mergesort")

    start = 0
    for category, count in enumerate(counts):
        stop = start + count
        categories[sorted_indices[start:stop]] = int(category)
        start = stop
    return categories


def discretize_by_quantiles(raw_values, class_probabilities):
    """Convert latent propensities into integer-coded categories."""
    raw_values = np.asarray(raw_values, dtype=float)
    if raw_values.ndim != 1:
        raise ValueError("raw_values must be one-dimensional.")
    if raw_values.size == 0:
        return np.array([], dtype=np.int64)

    class_probabilities = _normalise_probabilities(class_probabilities)
    cumulative_probabilities = np.cumsum(class_probabilities)[:-1]
    thresholds = np.quantile(raw_values, cumulative_probabilities)
    categories = np.digitize(raw_values, thresholds, right=False).astype(np.int64)

    if len(np.unique(categories)) < min(len(class_probabilities), raw_values.size):
        return _discretize_by_rank(raw_values, class_probabilities)
    return categories


def transform_categorical_node(raw_values, num_categories,
                               class_probabilities=None, rng=None):
    """
    Convert one latent continuous propensity into integer-coded categories.

    rng is accepted to keep the helper compatible with optional future
    categorical noise, but no noise is applied by default.
    """
    del rng
    if class_probabilities is None:
        class_probabilities = default_class_probabilities(num_categories)
    if len(class_probabilities) != int(num_categories):
        raise ValueError("class_probabilities must match num_categories.")
    return discretize_by_quantiles(raw_values, class_probabilities)


def assign_node_types(nodes, categorical_ratio, rng):
    """Assign nodes to categorical or continuous final types."""
    nodes = [int(node) for node in nodes]
    categorical_count = _closest_integer_count(len(nodes), categorical_ratio)
    if categorical_count == 0:
        categorical_nodes = set()
    else:
        categorical_nodes = set(
            int(node)
            for node in rng.choice(nodes, size=categorical_count, replace=False)
        )
    return {
        int(node): "categorical" if int(node) in categorical_nodes else "continuous"
        for node in nodes
    }


def assign_categorical_cardinalities(categorical_nodes, rng):
    """Assign 50% binary, 30% three-class, and 20% four/five-class variables."""
    categorical_nodes = [int(node) for node in categorical_nodes]
    if len(categorical_nodes) == 0:
        return {}

    shuffled_nodes = [int(node) for node in rng.permutation(categorical_nodes)]
    binary_count, three_class_count, high_cardinality_count = _allocate_counts(
        total=len(shuffled_nodes),
        probabilities=[0.5, 0.3, 0.2],
    )

    cardinalities = {}
    cursor = 0
    for node in shuffled_nodes[cursor:cursor + binary_count]:
        cardinalities[node] = 2
    cursor += binary_count

    for node in shuffled_nodes[cursor:cursor + three_class_count]:
        cardinalities[node] = 3
    cursor += three_class_count

    for node in shuffled_nodes[cursor:cursor + high_cardinality_count]:
        cardinalities[node] = int(rng.choice([4, 5]))

    return cardinalities


class Data_Generation_Process():
    """
    IID SF data generator with hidden unfair nodes and R2-based target choice.

    Node convention for the SF base graph:
    - S_obs is fixed at node 0 and is the only protected node fed to the model.
    - S_unobs_1 and S_unobs_2 are fixed at nodes 1 and 2.
    - S_unobs nodes are regular SF graph nodes, but removed from model data.
    - Candidate X/Y nodes start at node 3.
    - Y is sampled for SF graphs with probabilities proportional to R2 by default.

    Hidden U convention:
    - U nodes are appended after the SF base graph nodes.
    - U counts are generated for ratios 0%, 50%, 100%, and 200% by default.
    - Each U node independently samples a confounder, mediator, or collider role.
    """

    S_OBS_NODE = 0
    S_UNOBS_NODES = [1, 2]
    PROTECTED_UNFAIR_NODES = [0, 1, 2]
    U_ROLES = ['confounder', 'mediator', 'collider']

    def __init__(self,
                beta_lower_limit,
                betta_upper_limit_values,
                cont_noise,
                nr_nodes_values,
                edge_desnity_values,
                data_scale_values,
                num_samples,
                nonlinearities,
                num_u_children=1,
                u_child_selection='auto',
                inject_direct_u_to_y=False,
                keep_background_edges=True,
                sf_target_selection_method='r2',
                u_ratios=None,
                u_role_proportions=None,
                num_unobs_s_children=None,
                excluded_target_nodes=None,
                target_type="binary",
                target_binarization_method="quantile",
                target_binarization_quantile=0.5,
                target_binarization_threshold=None,
                target_positive_rule="greater_equal",
                keep_continuous_target=True,
                num_seed_runs=10,
                seed_run_values=None):

        self.beta_lower_limit = beta_lower_limit
        self.betta_upper_limit_values = betta_upper_limit_values
        self.cont_noise = cont_noise
        self.nr_nodes_values = nr_nodes_values
        self.edge_desnity_values = edge_desnity_values
        self.data_scale_values = data_scale_values
        self.num_samples = num_samples
        self.nonlinearities = nonlinearities
        self.num_u_children = num_u_children
        self.u_child_selection = u_child_selection
        self.inject_direct_u_to_y = inject_direct_u_to_y
        self.keep_background_edges = keep_background_edges
        self.sf_target_selection_method = sf_target_selection_method
        self.u_ratios = [0.0, 0.5, 1.0, 2.0] if u_ratios is None else list(u_ratios)
        self.u_role_proportions = self._normalise_u_role_proportions(u_role_proportions)
        self.num_unobs_s_children = num_unobs_s_children
        self.excluded_target_nodes = [] if excluded_target_nodes is None else list(excluded_target_nodes)
        self.target_type = target_type
        self.target_binarization_method = target_binarization_method
        self.target_binarization_quantile = target_binarization_quantile
        self.target_binarization_threshold = target_binarization_threshold
        self.target_positive_rule = target_positive_rule
        self.keep_continuous_target = keep_continuous_target
        if seed_run_values is None:
            self.seed_run_values = list(range(0, int(num_seed_runs)))
        else:
            self.seed_run_values = [int(seed_run) for seed_run in seed_run_values]
        self.num_seed_runs = len(self.seed_run_values)
        if self.num_seed_runs < 1:
            raise ValueError('num_seed_runs must be at least 1')

        super(Data_Generation_Process, self).__init__()

    def _normalise_u_role_proportions(self, u_role_proportions):
        """Return non-negative role sampling weights."""
        if u_role_proportions is None:
            return {'confounder': 1.0, 'mediator': 1.0, 'collider': 1.0}

        if isinstance(u_role_proportions, dict):
            weights = {role: float(u_role_proportions.get(role, 0.0))
                       for role in self.U_ROLES}
        else:
            if len(u_role_proportions) != 3:
                raise ValueError('u_role_proportions must have three entries')
            weights = {role: float(weight)
                       for role, weight in zip(self.U_ROLES, u_role_proportions)}

        if any(weight < 0 for weight in weights.values()):
            raise ValueError('U role proportions must be non-negative')
        if sum(weights.values()) <= 0:
            raise ValueError('At least one U role proportion must be positive')
        return weights

    def generate_dag(self, num_nodes, edge_density, seed=None):
        """ER generation is intentionally disabled in this SF-only version."""
        raise NotImplementedError('Data_Generation only supports SF graphs')

    def generate_scale_free_dag(self, num_nodes, edges_per_new_node, seed=None):
        """
        Generate an SF/Barabasi-Albert background DAG.

        The BA model first creates an undirected scale-free graph. We then orient
        each edge from the smaller id to the larger id. Edges among S_obs and the
        two S_unobs nodes are removed so that the three unfair nodes do not
        directly affect one another.
        """
        if num_nodes < 5:
            raise ValueError('SF hidden-mechanism data requires at least 5 nodes')

        edges_per_new_node = int(edges_per_new_node)
        if edges_per_new_node < 1 or edges_per_new_node >= num_nodes:
            raise ValueError('edges_per_new_node must satisfy 1 <= m < num_nodes')

        graph = nx.barabasi_albert_graph(n=num_nodes, m=edges_per_new_node, seed=seed)
        dag = nx.DiGraph()
        dag.add_nodes_from(graph.nodes())
        dag.add_edges_from([(u, v) if u < v else (v, u) for (u, v) in graph.edges()])

        self._remove_edges_among_s_nodes(dag)

        assert nx.is_directed_acyclic_graph(dag)
        return dag

    def _scale_free_connectivity_from_edge_density(self, nr_nodes, edge_density):
        """
        Convert an ER-style edge density value into a BA m value.

        For a BA graph, roughly m new edges are added with each new node. The
        mapping below keeps the old edge_density API usable while giving SF
        graphs a reasonable connectivity level.
        """
        if edge_density >= 1:
            return min(int(edge_density), nr_nodes - 1)
        return max(1, min(nr_nodes - 1, int(ceil(edge_density * (nr_nodes - 1) / 2))))

    def _num_u_nodes_from_ratio(self, num_base_nodes, u_ratio):
        """Convert a U ratio into an integer number of hidden U nodes."""
        u_ratio = float(u_ratio)
        if u_ratio < 0:
            raise ValueError('u_ratio must be non-negative')
        return int(round(num_base_nodes * u_ratio))

    def assign_s_x_y_u_roles(self, num_base_nodes, target_node=None, u_ratio=0.0):
        """Assign S_obs, S_unobs, candidate X/Y, and appended U nodes."""
        if num_base_nodes < 5:
            raise ValueError('At least S_obs, two S_unobs nodes, one X node, and Y are required')

        protected_unfair_nodes = list(self.PROTECTED_UNFAIR_NODES)
        protected_unfair_nodes.extend(self.excluded_target_nodes)
        protected_unfair_nodes = sorted(set(protected_unfair_nodes))

        candidate_target_nodes = [node for node in range(3, num_base_nodes)
                                  if node not in protected_unfair_nodes]
        if len(candidate_target_nodes) == 0:
            raise ValueError('No valid candidate nodes are available for Y')

        if target_node is not None and target_node not in candidate_target_nodes:
            raise ValueError('target_node must be one of the valid candidate target nodes')

        x_nodes = [node for node in candidate_target_nodes if node != target_node]
        num_u_nodes = self._num_u_nodes_from_ratio(num_base_nodes, u_ratio)
        hidden_u_nodes = list(range(num_base_nodes, num_base_nodes + num_u_nodes))

        return {'S_Obs_Node': self.S_OBS_NODE,
                'S_Unobs_Nodes': list(self.S_UNOBS_NODES),
                'Protected_Unfair_Nodes': protected_unfair_nodes,
                'Candidate_Target_Nodes': candidate_target_nodes,
                'Target_Node': target_node,
                'X_Nodes': x_nodes,
                'Hidden_U_Nodes': hidden_u_nodes,
                'U_Ratio': float(u_ratio)}

    def _allocate_u_roles(self, hidden_u_nodes):
        """Randomly sample one role for each hidden U node."""
        role_nodes = {role: [] for role in self.U_ROLES}
        if len(hidden_u_nodes) == 0:
            return role_nodes

        weights = np.array([self.u_role_proportions[role] for role in self.U_ROLES],
                           dtype=float)
        probabilities = weights / weights.sum()
        sampled_roles = np.random.choice(self.U_ROLES,
                                         size=len(hidden_u_nodes),
                                         p=probabilities)
        for u_node, role in zip(hidden_u_nodes, sampled_roles):
            role_nodes[str(role)].append(u_node)

        return role_nodes

    def _remove_edges_among_s_nodes(self, graph):
        """Remove any causal edges among S_obs, S_unobs_1, and S_unobs_2."""
        protected_set = set(self.PROTECTED_UNFAIR_NODES)
        protected_edges = [(u, v) for u, v in graph.edges()
                           if u in protected_set and v in protected_set]
        graph.remove_edges_from(protected_edges)
        return protected_edges

    def _sf_unobs_s_children(self, graph, roles):
        """Return X children of S_unobs nodes that were generated by the SF DAG."""
        x_nodes = set(roles['X_Nodes'])
        selected_x_children = set()
        for s_unobs in roles['S_Unobs_Nodes']:
            selected_x_children.update(child for child in graph.successors(s_unobs)
                                       if child in x_nodes)
        return sorted(selected_x_children)

    def inject_unobserved_s_edges(self, base_graph, roles, num_children=None):
        """
        Keep S_unobs edges exactly as generated by the SF DAG.

        S_unobs_1 and S_unobs_2 are not extra inserted nodes. They are nodes 1
        and 2 in the SF graph. This step only removes any causal edges among
        the three S nodes and records the SF-generated S_unobs -> X children.
        """
        graph = base_graph.copy()
        self._remove_edges_among_s_nodes(graph)
        selected_x_children = self._sf_unobs_s_children(graph=graph, roles=roles)

        assert nx.is_directed_acyclic_graph(graph)
        return graph, selected_x_children

    def _x_nodes_that_can_reach_y(self, graph, roles):
        """Return X nodes with a directed path to Y."""
        target_node = roles['Target_Node']
        return sorted(
            [node for node in roles['X_Nodes'] if nx.has_path(graph, node, target_node)],
            key=lambda node: (-graph.degree(node), -graph.out_degree(node), node)
        )

    def _pick_reachable_x_nodes(self, reachable_x_nodes, start_index, count):
        """Pick deterministic X nodes from the reachable-to-Y candidate list."""
        if len(reachable_x_nodes) == 0:
            return []
        return [reachable_x_nodes[(start_index + offset) % len(reachable_x_nodes)]
                for offset in range(min(count, len(reachable_x_nodes)))]

    def _add_confounder_edges(self, full_graph, u_node, idx, s_nodes, target_node, reachable_x_nodes):
        """
        Add confounder edges.

        Supported definitions:
        1. U -> one of S and U -> Y.
        2. U -> X -> Y and U -> Y.
        3. U -> multiple X nodes where each X reaches Y.
        """
        confounder_pattern = idx % 3
        # Determing which type of confounder U would be
        #idx = 0 -> pattern 0
        #idx = 1 -> pattern 1
        #idx = 2 -> pattern 2
        #idx = 3 -> pattern 0
        #idx = 4 -> pattern 1
        #idx = 5 -> pattern 2

        if confounder_pattern == 0 or len(reachable_x_nodes) == 0:
            s_child = s_nodes[idx % len(s_nodes)]
            full_graph.add_edge(u_node, s_child)
            full_graph.add_edge(u_node, target_node)
            return
        # Pattern 0: S <- U -> Y
        # Question: only one S will be connected to U, is that okay?

        if confounder_pattern == 1:
            x_child = self._pick_reachable_x_nodes(reachable_x_nodes, idx, 1)[0]
            full_graph.add_edge(u_node, x_child)
            full_graph.add_edge(u_node, target_node)
            return
        # Pattern 1: U -> X -> ... -> Y (U points to an "reachable_x"), and U -> Y

        x_children = self._pick_reachable_x_nodes(reachable_x_nodes, idx, 2)
        if len(x_children) < 2:
            full_graph.add_edge(u_node, x_children[0])
            full_graph.add_edge(u_node, target_node)
            return
        for x_child in x_children:
            full_graph.add_edge(u_node, x_child)
        # Pattern 2：U -> X1 -> ... -> Y
        #            U -> X2 -> ... -> Y
        # That is to say, U pointing to two reachable_x

    def _add_mediator_edges(self, full_graph, u_node, idx, s_nodes, target_node, reachable_x_nodes):
        """
        Add mediator edges for hidden unfair paths.

        Supported definitions:
        S -> U -> Y, S -> U -> X -> Y, or S -> U -> X1 -> ... -> Xn -> Y.
        """
        s_parent = s_nodes[idx % len(s_nodes)]
        full_graph.add_edge(s_parent, u_node)
        # For a mediator, it must satisfy: S -> U exists

        mediator_pattern = idx % 3 # determines U's next connection
        if mediator_pattern == 0 or len(reachable_x_nodes) == 0:
            full_graph.add_edge(u_node, target_node)
            return
        # Case 1: idx % 3 == 0, generating S -> U -> Y

        x_children = self._pick_reachable_x_nodes(
            reachable_x_nodes=reachable_x_nodes,
            start_index=idx,
            count=1 if mediator_pattern == 1 else 2
        )
        for x_child in x_children:
            full_graph.add_edge(u_node, x_child)
        # Case 2: idx % 3 == 1, generating S -> U -> X -> ... -> Y
        # That is to say, U connects to a reachable_x 
        # else: Case 3: S -> U -> X1 -> ... -> Y, 
        #                      -> X2 -> ... -> Y, 
        # That is to say, U connects to two reachable_x
        # Question: Should we consider the case where one U is connected to more X ?


    def _add_collider_edges(self, full_graph, u_node, idx, s_nodes, target_node, reachable_x_nodes):
        """
        Add collider edges.

        Collider U receives incoming edges from two of three groups:
        Y, S nodes, and X nodes that can reach Y.
        """
        collider_pattern = idx % 3

        if collider_pattern == 0 or len(reachable_x_nodes) == 0:
            full_graph.add_edge(target_node, u_node)
            full_graph.add_edge(s_nodes[idx % len(s_nodes)], u_node)
            return
        # Pattern 0：Y -> U <- S

        if collider_pattern == 1:
            x_parent = self._pick_reachable_x_nodes(reachable_x_nodes, idx, 1)[0]
            full_graph.add_edge(target_node, u_node)
            full_graph.add_edge(x_parent, u_node)
            return
        # Pattern 1：Y -> U <- X

        x_parent = self._pick_reachable_x_nodes(reachable_x_nodes, idx, 1)[0]
        full_graph.add_edge(s_nodes[idx % len(s_nodes)], u_node)
        full_graph.add_edge(x_parent, u_node)
        # Pattern 2：S -> U <- X

    def inject_hidden_mechanisms(self, base_graph, roles):
        """
        Add all U nodes to the same SF graph.

        Confounder U matches one of:
        - U -> one of S and U -> Y.
        - U -> X -> Y and U -> Y.
        - U -> multiple X nodes, each with a directed path to Y.

        Mediator U matches hidden unfair paths:
        - S -> U -> Y.
        - S -> U -> X -> Y.
        - S -> U -> X1 -> ... -> Xn -> Y.

        Collider U receives incoming edges from two of:
        - Y -> U.
        - S -> U.
        - X -> U, where X has a directed path to Y.
        """
        if roles['Target_Node'] is None:
            raise ValueError('Target node must be selected before injecting U nodes')

        if self.keep_background_edges:
            full_graph = base_graph.copy()
        else:
            full_graph = nx.DiGraph()
            full_graph.add_nodes_from(base_graph.nodes())

        hidden_u_nodes = roles['Hidden_U_Nodes']
        full_graph.add_nodes_from(hidden_u_nodes)

        target_node = roles['Target_Node']
        s_nodes = [roles['S_Obs_Node']] + roles['S_Unobs_Nodes']
        role_nodes = self._allocate_u_roles(hidden_u_nodes)
        reachable_x_nodes = self._x_nodes_that_can_reach_y(graph=base_graph, roles=roles)

        for idx, u_node in enumerate(role_nodes['confounder']):
            self._add_confounder_edges(full_graph=full_graph,
                                       u_node=u_node,
                                       idx=idx,
                                       s_nodes=s_nodes,
                                       target_node=target_node,
                                       reachable_x_nodes=reachable_x_nodes)

        for idx, u_node in enumerate(role_nodes['mediator']):
            self._add_mediator_edges(full_graph=full_graph,
                                     u_node=u_node,
                                     idx=idx,
                                     s_nodes=s_nodes,
                                     target_node=target_node,
                                     reachable_x_nodes=reachable_x_nodes)

        for idx, u_node in enumerate(role_nodes['collider']):
            self._add_collider_edges(full_graph=full_graph,
                                     u_node=u_node,
                                     idx=idx,
                                     s_nodes=s_nodes,
                                     target_node=target_node,
                                     reachable_x_nodes=reachable_x_nodes)

        assert nx.is_directed_acyclic_graph(full_graph)
        return full_graph, role_nodes

    def observed_nodes_from_roles(self, roles):
        """Return model-fed nodes, with the selected Y placed last."""
        if roles['Target_Node'] is None:
            raise ValueError('Target node must be selected before building observed nodes')
        return [roles['S_Obs_Node']] + list(roles['X_Nodes']) + [roles['Target_Node']]

    def remove_hidden_nodes_from_matrix(self, matrix, observed_nodes):
        """Drop hidden rows and columns from a full adjacency matrix."""
        return matrix[np.ix_(observed_nodes, observed_nodes)]

    def sample_beta(self, beta_lower_limit, beta_upper_limit):
        """Sample one nonzero structural coefficient with random sign."""
        if np.random.randint(0, 2) == 0:
            return np.random.uniform(-beta_upper_limit, -beta_lower_limit, size=1)[0]
        return np.random.uniform(beta_lower_limit, beta_upper_limit, size=1)[0]

    def apply_transformation(self, dot_product, transformation):
        """Apply one causal transformation chosen from the configured mixture."""
        transformation_func_index = np.random.choice(
            a=[func_index for func_index in range(0, len(transformation))],
            p=[func[0] for func in transformation]
        )
        return transformation[transformation_func_index][1](dot_product)

    def _simulate_single_equation(self, X, w, scale, causal_transformation, n,
                                  noise_distribution="gaussian"):
        """Simulate one structural equation from its parent samples."""
        z = sample_sem_noise(
            distribution_type=noise_distribution,
            scale=scale,
            size=n,
        )
        if len(w) > 0:
            x = self.apply_transformation(dot_product=X @ w,
                                          transformation=causal_transformation) + z
        else:
            x = z
        return x

    def simulate_sem(self,
                 G,
                 W,
                 n,
                 causal_transformation,
                 graph_type,
                 noise_scale=None,
                 noise_distributions=None):
        """
        Simulate iid samples from the additive-noise SEM defined by G and W.
        """
        d = W.shape[0]
        if noise_scale is None:
            scale_vec = np.ones(d)
        elif np.isscalar(noise_scale):
            scale_vec = noise_scale * np.ones(d)
        else:
            if len(noise_scale) != d:
                raise ValueError('noise scale must be a scalar or has length d')
            scale_vec = noise_scale

        if noise_distributions is None:
            noise_distribution_vec = ["gaussian" for _ in range(0, d)]
        elif isinstance(noise_distributions, dict):
            noise_distribution_vec = [
                str(noise_distributions.get(node, "gaussian")).lower()
                for node in range(0, d)
            ]
        else:
            if len(noise_distributions) != d:
                raise ValueError('noise_distributions must be a dict or has length d')
            noise_distribution_vec = [
                str(noise_distribution).lower()
                for noise_distribution in noise_distributions
            ]

        if np.isinf(n):
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X

        ordered_vertices = list(nx.topological_sort(G))
        assert len(ordered_vertices) == d
        X = np.zeros([n, d])

        for j in ordered_vertices:
            parents = list(G.predecessors(j))
            X[:, j] = self._simulate_single_equation(X[:, parents],
                                                W[parents, j],
                                                scale_vec[j],
                                                causal_transformation=causal_transformation,
                                                n=n,
                                                noise_distribution=noise_distribution_vec[j])
        return X

    def get_avg_number_edges_ER_graph(self, frames_descriptions,
                                      save_path_edge_mapping):
        """
        Keep the original helper that maps ER edge counts to SF m values.
        """
        avg_number_edges = {}

        avg_number_edges['Nodes_10'] = {0.2: [],
                                    0.3: [],
                                    0.4: []}

        avg_number_edges['Nodes_20'] = {0.2: [],
                                    0.3: [],
                                    0.4: []}

        avg_number_edges['Nodes_50'] = {0.2: [],
                                    0.3: [],
                                    0.4: []}

        avg_number_edges['Nodes_100'] = {0.2: [],
                                    0.3: [],
                                    0.4: []}

        for idx in range(0, frames_descriptions.shape[0]):
            if frames_descriptions.iloc[idx, 1] == 10:
                avg_number_edges['Nodes_10'][frames_descriptions.iloc[idx, 2]].append(frames_descriptions.iloc[idx, 3])
            elif frames_descriptions.iloc[idx, 1] == 20:
                avg_number_edges['Nodes_20'][frames_descriptions.iloc[idx, 2]].append(frames_descriptions.iloc[idx, 3])
            elif frames_descriptions.iloc[idx, 1] == 50:
                avg_number_edges['Nodes_50'][frames_descriptions.iloc[idx, 2]].append(frames_descriptions.iloc[idx, 3])
            else:
                avg_number_edges['Nodes_100'][frames_descriptions.iloc[idx, 2]].append(frames_descriptions.iloc[idx, 3])

        for node_key in avg_number_edges.keys():
            avg_number_edges[node_key][0.2] = {'e': int(ceil(np.mean(avg_number_edges[node_key][0.2]))),
                                            'd': int(node_key.split('_')[1]),
                                            'k': int(ceil(np.mean(avg_number_edges[node_key][0.2]) / int(node_key.split('_')[1])))}

            avg_number_edges[node_key][0.3] = {'e': int(ceil(np.mean(avg_number_edges[node_key][0.3]))),
                                            'd': int(node_key.split('_')[1]),
                                            'k': int(ceil(np.mean(avg_number_edges[node_key][0.3]) / int(node_key.split('_')[1])))}

            avg_number_edges[node_key][0.4] = {'e': int(ceil(np.mean(avg_number_edges[node_key][0.4]))),
                                            'd': int(node_key.split('_')[1]),
                                            'k': int(ceil(np.mean(avg_number_edges[node_key][0.4]) / int(node_key.split('_')[1])))}

        avg_number_edges['Nodes_10'][0.4]['k'] = avg_number_edges['Nodes_10'][0.4]['k'] + 1

        with open(save_path_edge_mapping, 'wb') as f:
            pickle.dump(avg_number_edges, f)

        return avg_number_edges

    def _connectivity_list_for_graph(self, graph_type_key, nr_nodes, avg_number_edges):
        """Choose BA m values for SF graph generation."""
        if graph_type_key not in ['SF', 'SCALE_FREE']:
            raise ValueError("This Data_Generation generator only supports 'SF'")

        if avg_number_edges is not None:
            connectivity_list = []
            for ed_dns in [0.2, 0.3, 0.4]:
                node_key = 'Nodes_' + str(nr_nodes)
                if node_key in avg_number_edges and ed_dns in avg_number_edges[node_key].keys():
                    k_value = avg_number_edges[node_key][ed_dns]['k']
                    connectivity_list.append(max(1, min(int(k_value), nr_nodes - 1)))
            return connectivity_list

        return [self._scale_free_connectivity_from_edge_density(nr_nodes=nr_nodes,
                                                                edge_density=edge_density)
                for edge_density in self.edge_desnity_values]

    def _generate_base_graph(self, graph_type_key, nr_nodes, connectivity, seed=None):
        """Generate the SF background DAG before hidden nodes are injected."""
        if graph_type_key not in ['SF', 'SCALE_FREE']:
            raise ValueError("This Data_Generation generator only supports 'SF'")

        return self.generate_scale_free_dag(num_nodes=nr_nodes,
                                            edges_per_new_node=connectivity,
                                            seed=seed)

    def _matrix_from_graph(self, graph, num_nodes):
        """Convert a NetworkX DAG into a binary adjacency matrix."""
        adjacency_matrix = np.zeros(shape=(num_nodes, num_nodes))
        for edge in graph.edges:
            adjacency_matrix[edge[0]][edge[1]] = 1
        return adjacency_matrix

    def _weighted_adjacency_from_graph(self, graph, num_nodes, beta_upper_limit):
        """Create a weighted adjacency matrix for the supplied graph."""
        adjacency_matrix = self._matrix_from_graph(graph=graph, num_nodes=num_nodes)
        betas = np.array([self.sample_beta(beta_lower_limit=self.beta_lower_limit,
                                         beta_upper_limit=beta_upper_limit)
                        for _ in range(0, num_nodes * num_nodes)])
        weighted_adjacency = np.reshape(betas, (num_nodes, num_nodes)) * adjacency_matrix
        weighted_adjacency = np.where(weighted_adjacency == 0.0, 0.0, weighted_adjacency)
        return adjacency_matrix, weighted_adjacency

    def _transformation_name(self, function_transformation):
        """Return the original string label for the transformation mixture."""
        if function_transformation == [(1.0, lin_func)]:
            return 'Linear_100%'
        if function_transformation == [(0.5, lin_func), (0.5, relu_func)]:
            return 'Linear_ReLU_50%'
        if function_transformation == [(0.3, lin_func), (0.7, relu_func)]:
            return 'Linear_30%_ReLU_70%'
        return 'Linear_10%_ReLU_90%'

    def _select_sf_target(self,
                          base_graph,
                          roles,
                          beta_upper_limit,
                          causal_transformation,
                          graph_type,
                          sf_target_selection_method):
        """Select the SF target node by R2-weighted sampling or legacy last-node behavior."""
        base_num_nodes = len(base_graph.nodes())
        base_adjacency, base_weighted_adjacency = self._weighted_adjacency_from_graph(
            graph=base_graph,
            num_nodes=base_num_nodes,
            beta_upper_limit=beta_upper_limit
        )
        del base_adjacency

        base_data = self.simulate_sem(G=base_graph,
                                      W=base_weighted_adjacency,
                                      n=self.num_samples,
                                      causal_transformation=causal_transformation,
                                      graph_type=graph_type,
                                      noise_scale=self.cont_noise)
    #generating one DAG, a set of random SEM, and a data set generated    

        candidate_nodes = roles['Candidate_Target_Nodes']
        method_key = str(sf_target_selection_method).lower()
        if method_key == 'last_node':
            r2_scores = compute_r2_scores(base_data)
            return max(candidate_nodes), r2_scores
        if method_key == 'r2':
            return select_target_by_r2(base_data, candidate_nodes)

        raise ValueError("sf_target_selection_method must be 'r2' or 'last_node'")

    def _target_binarization_metadata(self, dataframe, y_continuous,
                                      threshold_used, positive_rate):
        """Attach latent-target metadata without adding model feature columns."""
        if self.keep_continuous_target:
            dataframe.attrs["target_continuous"] = y_continuous
        dataframe.attrs["target_type"] = self.target_type
        dataframe.attrs["target_binarization_method"] = self.target_binarization_method
        dataframe.attrs["target_binarization_quantile"] = self.target_binarization_quantile
        dataframe.attrs["target_binarization_threshold"] = threshold_used
        dataframe.attrs["target_positive_rate"] = positive_rate

    def _mixed_type_rng(self, seed_run, nr_nodes, connectivity, beta_upper_limit,
                        data_scale, u_ratio, target_node):
        """Return a transformation RNG that does not perturb SEM graph sampling."""
        return np.random.RandomState(
            _stable_seed_from_values(
                "mixed_types",
                seed_run,
                nr_nodes,
                connectivity,
                beta_upper_limit,
                data_scale,
                u_ratio,
                target_node,
            )
        )

    def _node_name(self, node, roles):
        node = int(node)
        if node == self.S_OBS_NODE:
            return "S0"
        if node == self.S_UNOBS_NODES[0]:
            return "S1"
        if node == self.S_UNOBS_NODES[1]:
            return "S2"
        if node == roles['Target_Node']:
            return "Y"
        if node in roles['Hidden_U_Nodes']:
            return f"U{roles['Hidden_U_Nodes'].index(node)}"
        if node in roles['X_Nodes']:
            return f"X{node}"
        return f"N{node}"

    def _node_role(self, node, roles):
        node = int(node)
        if node in self.PROTECTED_UNFAIR_NODES:
            return "S"
        if node in roles['Hidden_U_Nodes']:
            return "U"
        if node == roles['Target_Node']:
            return "Y"
        if node in roles['X_Nodes']:
            return "X"
        return "X"

    def _assign_continuous_distributions(self, continuous_nodes, rng):
        """Assign varied continuous scale families deterministically."""
        continuous_nodes = [int(node) for node in continuous_nodes]
        if len(continuous_nodes) == 0:
            return {}

        shuffled_nodes = [int(node) for node in rng.permutation(continuous_nodes)]
        shuffled_distributions = [
            str(distribution_type)
            for distribution_type in rng.permutation(CONTINUOUS_DISTRIBUTION_TYPES)
        ]
        return {
            int(node): shuffled_distributions[index % len(shuffled_distributions)]
            for index, node in enumerate(shuffled_nodes)
        }

    def _assign_sem_noise_distributions(self, metadata_by_node, rng):
        """Assign varied continuous SEM noise families to final continuous nodes."""
        noise_distributions = {
            int(node): "gaussian"
            for node in metadata_by_node.keys()
        }
        continuous_nodes = [
            int(node)
            for node, metadata in metadata_by_node.items()
            if metadata["final_type"] == "continuous"
        ]
        if len(continuous_nodes) == 0:
            return noise_distributions

        shuffled_nodes = [int(node) for node in rng.permutation(continuous_nodes)]
        shuffled_noise_types = [
            str(noise_distribution)
            for noise_distribution in rng.permutation(SEM_NOISE_DISTRIBUTION_TYPES)
        ]
        for index, node in enumerate(shuffled_nodes):
            noise_distributions[node] = shuffled_noise_types[index % len(shuffled_noise_types)]
        return noise_distributions

    def _metadata_entry(self, node, roles, observed_nodes_set, final_type,
                        distribution_type=None, num_categories=None,
                        class_probabilities=None, sem_noise_distribution=None):
        role = self._node_role(node=node, roles=roles)
        node = int(node)
        observed = node in observed_nodes_set
        included_in_model_input = observed and role in {"S", "X"}
        return {
            "node_id": node,
            "node_name": self._node_name(node=node, roles=roles),
            "role": role,
            "observed": bool(observed),
            "final_type": final_type,
            "distribution_type": distribution_type,
            "num_categories": None if num_categories is None else int(num_categories),
            "class_probabilities": (
                None if class_probabilities is None
                else [float(probability) for probability in class_probabilities]
            ),
            "sem_noise_distribution": sem_noise_distribution,
            "included_in_model_input": bool(included_in_model_input),
            "included_in_observed_data": bool(observed),
        }

    def _build_node_metadata(self, roles, full_num_nodes, target_type_key, rng):
        """Build final type metadata for S, U, X, and Y nodes."""
        observed_nodes = self.observed_nodes_from_roles(roles=roles)
        observed_nodes_set = set(int(node) for node in observed_nodes)

        x_type_assignments = assign_node_types(
            nodes=roles['X_Nodes'],
            categorical_ratio=0.30,
            rng=rng,
        )
        u_type_assignments = assign_node_types(
            nodes=roles['Hidden_U_Nodes'],
            categorical_ratio=0.30,
            rng=rng,
        )
        categorical_nodes = [
            node for node, final_type in {
                **x_type_assignments,
                **u_type_assignments,
            }.items()
            if final_type == "categorical"
        ]
        categorical_cardinalities = assign_categorical_cardinalities(
            categorical_nodes=categorical_nodes,
            rng=rng,
        )

        continuous_nodes = [
            node for node, final_type in {
                **x_type_assignments,
                **u_type_assignments,
            }.items()
            if final_type == "continuous"
        ]
        continuous_distributions = self._assign_continuous_distributions(
            continuous_nodes=continuous_nodes,
            rng=rng,
        )

        metadata_by_node = {}
        target_node = int(roles['Target_Node'])

        for node in range(0, int(full_num_nodes)):
            if node == self.S_OBS_NODE:
                metadata_by_node[node] = self._metadata_entry(
                    node=node,
                    roles=roles,
                    observed_nodes_set=observed_nodes_set,
                    final_type="categorical",
                    num_categories=2,
                    class_probabilities=[0.5, 0.5],
                )
            elif node == self.S_UNOBS_NODES[0]:
                metadata_by_node[node] = self._metadata_entry(
                    node=node,
                    roles=roles,
                    observed_nodes_set=observed_nodes_set,
                    final_type="categorical",
                    num_categories=3,
                    class_probabilities=[0.5, 0.3, 0.2],
                )
            elif node == self.S_UNOBS_NODES[1]:
                metadata_by_node[node] = self._metadata_entry(
                    node=node,
                    roles=roles,
                    observed_nodes_set=observed_nodes_set,
                    final_type="continuous",
                    distribution_type="standard_normal",
                )
            elif node == target_node:
                if target_type_key == "binary":
                    metadata_by_node[node] = self._metadata_entry(
                        node=node,
                        roles=roles,
                        observed_nodes_set=observed_nodes_set,
                        final_type="categorical",
                        num_categories=2,
                        class_probabilities=[0.5, 0.5],
                    )
                else:
                    metadata_by_node[node] = self._metadata_entry(
                        node=node,
                        roles=roles,
                        observed_nodes_set=observed_nodes_set,
                        final_type="continuous",
                        distribution_type="latent_sem_continuous",
                    )
            elif node in x_type_assignments:
                final_type = x_type_assignments[node]
                if final_type == "categorical":
                    num_categories = categorical_cardinalities[node]
                    metadata_by_node[node] = self._metadata_entry(
                        node=node,
                        roles=roles,
                        observed_nodes_set=observed_nodes_set,
                        final_type="categorical",
                        num_categories=num_categories,
                        class_probabilities=default_class_probabilities(num_categories),
                    )
                else:
                    metadata_by_node[node] = self._metadata_entry(
                        node=node,
                        roles=roles,
                        observed_nodes_set=observed_nodes_set,
                        final_type="continuous",
                        distribution_type=continuous_distributions[node],
                    )
            elif node in u_type_assignments:
                final_type = u_type_assignments[node]
                if final_type == "categorical":
                    num_categories = categorical_cardinalities[node]
                    metadata_by_node[node] = self._metadata_entry(
                        node=node,
                        roles=roles,
                        observed_nodes_set=observed_nodes_set,
                        final_type="categorical",
                        num_categories=num_categories,
                        class_probabilities=default_class_probabilities(num_categories),
                    )
                else:
                    metadata_by_node[node] = self._metadata_entry(
                        node=node,
                        roles=roles,
                        observed_nodes_set=observed_nodes_set,
                        final_type="continuous",
                        distribution_type=continuous_distributions[node],
                    )
            else:
                metadata_by_node[node] = self._metadata_entry(
                    node=node,
                    roles=roles,
                    observed_nodes_set=observed_nodes_set,
                    final_type="continuous",
                    distribution_type="standard_normal",
                )

        sem_noise_distributions = self._assign_sem_noise_distributions(
            metadata_by_node=metadata_by_node,
            rng=rng,
        )
        for node, noise_distribution in sem_noise_distributions.items():
            metadata_by_node[int(node)]["sem_noise_distribution"] = noise_distribution
        return metadata_by_node

    def _sem_noise_distribution_map(self, metadata_by_node):
        return {
            int(node): metadata["sem_noise_distribution"]
            for node, metadata in metadata_by_node.items()
        }

    def _metadata_list(self, metadata_by_node):
        return [
            metadata_by_node[node].copy()
            for node in sorted(metadata_by_node.keys())
        ]

    def _observed_metadata_list(self, metadata_by_node, observed_nodes):
        observed_metadata = []
        for observed_column, node in enumerate(observed_nodes):
            entry = metadata_by_node[int(node)].copy()
            entry["observed_column"] = int(observed_column)
            observed_metadata.append(entry)
        return observed_metadata

    def _attach_mixed_type_attrs(self, dataframe, metadata_by_node, observed_nodes,
                                 raw_data=None):
        dataframe.attrs["node_metadata"] = self._metadata_list(metadata_by_node)
        dataframe.attrs["observed_node_metadata"] = self._observed_metadata_list(
            metadata_by_node=metadata_by_node,
            observed_nodes=observed_nodes,
        )
        dataframe.attrs["observed_nodes"] = [int(node) for node in observed_nodes]
        if raw_data is not None:
            dataframe.attrs["raw_data"] = raw_data.copy()

    def _apply_mixed_type_transformations(self, latent_dataframe, raw_dataframe,
                                          roles, metadata_by_node, rng):
        """Transform latent SEM values into final mixed-type observed values."""
        observed_nodes = self.observed_nodes_from_roles(roles=roles)
        target_node = int(roles['Target_Node'])
        mixed_dataframe = latent_dataframe.copy()

        for node, metadata in metadata_by_node.items():
            if node == target_node:
                continue

            raw_values = latent_dataframe.loc[:, node].to_numpy(copy=True)
            if metadata["final_type"] == "continuous":
                mixed_dataframe[node] = pd.Series(
                    transform_continuous_node(
                        raw_values=raw_values,
                        distribution_type=metadata["distribution_type"],
                        rng=rng,
                    ),
                    index=mixed_dataframe.index,
                    dtype=float,
                )
            elif metadata["final_type"] == "categorical":
                mixed_dataframe[node] = pd.Series(
                    transform_categorical_node(
                        raw_values=raw_values,
                        num_categories=metadata["num_categories"],
                        class_probabilities=metadata["class_probabilities"],
                        rng=rng,
                    ),
                    index=mixed_dataframe.index,
                    dtype=np.int64,
                )
            else:
                raise ValueError(f"Unsupported final_type: {metadata['final_type']}")

        self._attach_mixed_type_attrs(
            dataframe=mixed_dataframe,
            metadata_by_node=metadata_by_node,
            observed_nodes=observed_nodes,
            raw_data=raw_dataframe,
        )
        return mixed_dataframe, metadata_by_node

    def _update_target_metadata(self, metadata_by_node, target_node, positive_rate):
        target_metadata = metadata_by_node[int(target_node)]
        if target_metadata["final_type"] == "categorical":
            target_metadata["class_probabilities"] = [
                float(1.0 - positive_rate),
                float(positive_rate),
            ]

    def _assert_binary_target_invariants(self, observed_dataframe, observed_nodes,
                                         target_column):
        """Check that target binarization changed only the label values."""
        target_values = observed_dataframe.iloc[:, target_column]
        unique_target_values = set(target_values.unique())

        if not unique_target_values.issubset({0, 1}):
            raise AssertionError(
                f"Binary target contains unexpected values: {unique_target_values}"
            )

        if len(unique_target_values) != 2:
            raise AssertionError(
                f"Binary target must contain both classes, got {unique_target_values}"
            )

        if not np.issubdtype(target_values.dtype, np.integer):
            raise AssertionError(
                f"Binary target should have integer dtype, got {target_values.dtype}"
            )

        if observed_dataframe.shape[1] != len(observed_nodes):
            raise AssertionError("Binarization changed the number of observed columns.")

        if target_column != observed_dataframe.shape[1] - 1:
            raise AssertionError("Target column should remain the final observed column.")

    def large_scale_simulation(self,
                          graph_type='SF',
                          avg_number_edges=None,
                          num_u_children=None,
                          u_child_selection=None,
                          u_ratios=None,
                          sf_target_selection_method=None):
        """
        Generate SF observed data with S_unobs and U nodes hidden.

        Returns:
        [descriptions,
         observed_true_causal_matrices,
         observed_true_weighted_causal_matrices,
         observed_frames,
         full_true_causal_matrices,
         full_true_weighted_causal_matrices,
         full_frames]
        """
        graph_type_key = str(graph_type).upper()
        if graph_type_key not in ['SF', 'SCALE_FREE']:
            raise ValueError("Data_Generation only supports graph_type='SF'")

        if num_u_children is None:
            num_u_children = self.num_unobs_s_children
        if u_child_selection is None:
            u_child_selection = self.u_child_selection
        if u_ratios is None:
            u_ratios = self.u_ratios
        if sf_target_selection_method is None:
            sf_target_selection_method = self.sf_target_selection_method

        seed_runs = []
        nr_nodes_array = []
        nr_base_nodes_array = []
        nr_nodes_full_array = []
        connectivity_array = []
        function_transformation_array = []
        data_scale_array = []
        beta_upper_array = []
        number_edges_array = []
        number_edges_full_array = []

        s_obs_node_array = []
        s_unobs_nodes_array = []
        protected_unfair_nodes_array = []
        target_node_array = []
        target_column_array = []
        target_r2_score_array = []
        r2_scores_array = []
        candidate_target_nodes_array = []
        excluded_target_nodes_array = []
        x_nodes_array = []
        feature_columns_array = []
        observed_nodes_array = []
        hidden_u_nodes_array = []
        u_ratio_array = []
        num_u_nodes_array = []
        u_confounder_nodes_array = []
        u_mediator_nodes_array = []
        u_collider_nodes_array = []
        s_unobs_children_array = []
        hidden_selection_array = []
        target_type_array = []
        target_binarization_method_array = []
        target_binarization_quantile_array = []
        target_binarization_threshold_array = []
        target_positive_rate_array = []
        node_metadata_array = []
        observed_node_metadata_array = []
        raw_data_available_array = []

        frames = []
        true_causal_matrices = []
        true_weighted_causal_matrices = []

        full_frames = []
        full_true_causal_matrices = []
        full_true_weighted_causal_matrices = []

        for seed_run in self.seed_run_values:
            np.random.seed(seed_run)
            for nr_nodes in self.nr_nodes_values:
                connectivity_list = self._connectivity_list_for_graph(graph_type_key=graph_type_key,
                                                                     nr_nodes=nr_nodes,
                                                                     avg_number_edges=avg_number_edges)

                for connectivity in connectivity_list:
                    for function_transformation in self.nonlinearities:
                        for beta_upper_limit in self.betta_upper_limit_values:
                            base_graph = self._generate_base_graph(graph_type_key=graph_type_key,
                                                                 nr_nodes=nr_nodes,
                                                                 connectivity=connectivity,
                                                                 seed=seed_run)
                            base_roles = self.assign_s_x_y_u_roles(num_base_nodes=nr_nodes,
                                                                   target_node=None,
                                                                   u_ratio=0.0)
                            selected_target_node, r2_scores = self._select_sf_target(
                                base_graph=base_graph,
                                roles=base_roles,
                                beta_upper_limit=beta_upper_limit,
                                causal_transformation=function_transformation,
                                graph_type=graph_type,
                                sf_target_selection_method=sf_target_selection_method
                            )

                            for data_scale in self.data_scale_values:
                                for u_ratio in u_ratios:
                                    roles = self.assign_s_x_y_u_roles(num_base_nodes=nr_nodes,
                                                                     target_node=selected_target_node,
                                                                     u_ratio=u_ratio)

                                    graph_with_s_unobs, s_unobs_children = self.inject_unobserved_s_edges(
                                        base_graph=base_graph,
                                        roles=roles,
                                        num_children=num_u_children
                                    )

                                    full_graph, role_nodes = self.inject_hidden_mechanisms(
                                        base_graph=graph_with_s_unobs,
                                        roles=roles
                                    )

                                    full_num_nodes = nr_nodes + len(roles['Hidden_U_Nodes'])
                                    full_adjacency_matrix, full_weighted_adjacency = self._weighted_adjacency_from_graph(
                                        graph=full_graph,
                                        num_nodes=full_num_nodes,
                                        beta_upper_limit=beta_upper_limit
                                    )
                                    full_true_causal_matrices.append(full_adjacency_matrix)
                                    full_true_weighted_causal_matrices.append(full_weighted_adjacency)

                                    observed_nodes = self.observed_nodes_from_roles(roles=roles)
                                    target_node = roles['Target_Node']
                                    target_column = len(observed_nodes) - 1
                                    target_type_key = str(self.target_type).lower()
                                    mixed_rng = self._mixed_type_rng(
                                        seed_run=seed_run,
                                        nr_nodes=nr_nodes,
                                        connectivity=connectivity,
                                        beta_upper_limit=beta_upper_limit,
                                        data_scale=data_scale,
                                        u_ratio=u_ratio,
                                        target_node=target_node,
                                    )
                                    node_metadata_by_node = self._build_node_metadata(
                                        roles=roles,
                                        full_num_nodes=full_num_nodes,
                                        target_type_key=target_type_key,
                                        rng=mixed_rng,
                                    )

                                    raw_sem_values = self.simulate_sem(
                                        G=full_graph,
                                        W=full_weighted_adjacency,
                                        n=self.num_samples,
                                        causal_transformation=function_transformation,
                                        graph_type=graph_type,
                                        noise_scale=self.cont_noise,
                                        noise_distributions=self._sem_noise_distribution_map(
                                            metadata_by_node=node_metadata_by_node
                                        ),
                                    )
                                    raw_full_dataframe = pd.DataFrame(
                                        raw_sem_values,
                                        columns=[col_index for col_index in range(0, raw_sem_values.shape[1])]
                                    )

                                    if data_scale == 'standardized':
                                        latent_full_dataframe = (
                                            raw_full_dataframe - raw_full_dataframe.mean(axis=0)
                                        ) / raw_full_dataframe.std(axis=0)
                                        latent_full_dataframe = latent_full_dataframe.replace(
                                            [np.inf, -np.inf],
                                            np.nan,
                                        ).fillna(0.0)
                                    else:
                                        latent_full_dataframe = raw_full_dataframe.copy()

                                    full_dataframe, node_metadata_by_node = (
                                        self._apply_mixed_type_transformations(
                                            latent_dataframe=latent_full_dataframe,
                                            raw_dataframe=raw_full_dataframe,
                                            roles=roles,
                                            metadata_by_node=node_metadata_by_node,
                                            rng=mixed_rng,
                                        )
                                    )

                                    # Fifth meeting design decision:
                                    # Hidden / unobserved nodes participate in the full data generating process
                                    # and may affect Y, but they must be removed before constructing train X and
                                    # test X for fairness processors. The target Y is represented as a binary
                                    # categorical outcome.
                                    if target_type_key == "binary":
                                        y_continuous = latent_full_dataframe.loc[:, target_node].to_numpy(copy=True)
                                        y_binary, threshold_used, positive_rate = _binarize_target_values(
                                            y=y_continuous,
                                            method=self.target_binarization_method,
                                            quantile=self.target_binarization_quantile,
                                            threshold=self.target_binarization_threshold,
                                            positive_rule=self.target_positive_rule,
                                        )
                                        full_dataframe[target_node] = pd.Series(
                                            y_binary,
                                            index=full_dataframe.index,
                                            dtype=np.int64
                                        )
                                        self._update_target_metadata(
                                            metadata_by_node=node_metadata_by_node,
                                            target_node=target_node,
                                            positive_rate=positive_rate,
                                        )
                                        self._attach_mixed_type_attrs(
                                            dataframe=full_dataframe,
                                            metadata_by_node=node_metadata_by_node,
                                            observed_nodes=observed_nodes,
                                            raw_data=raw_full_dataframe,
                                        )
                                        self._target_binarization_metadata(
                                            dataframe=full_dataframe,
                                            y_continuous=y_continuous,
                                            threshold_used=threshold_used,
                                            positive_rate=positive_rate
                                        )
                                    elif target_type_key == "continuous":
                                        threshold_used = np.nan
                                        positive_rate = np.nan
                                    else:
                                        raise ValueError("target_type must be 'binary' or 'continuous'.")

                                    observed_dataframe = full_dataframe.loc[:, observed_nodes].copy()
                                    observed_dataframe.columns = [col_index for col_index in range(0, observed_dataframe.shape[1])]
                                    self._attach_mixed_type_attrs(
                                        dataframe=observed_dataframe,
                                        metadata_by_node=node_metadata_by_node,
                                        observed_nodes=observed_nodes,
                                        raw_data=None,
                                    )

                                    if target_type_key == "binary":
                                        observed_dataframe[target_column] = pd.Series(
                                            y_binary,
                                            index=observed_dataframe.index,
                                            dtype=np.int64
                                        )
                                        self._target_binarization_metadata(
                                            dataframe=observed_dataframe,
                                            y_continuous=y_continuous,
                                            threshold_used=threshold_used,
                                            positive_rate=positive_rate
                                        )
                                        self._assert_binary_target_invariants(
                                            observed_dataframe=observed_dataframe,
                                            observed_nodes=observed_nodes,
                                            target_column=target_column
                                        )

                                    observed_adjacency_matrix = self.remove_hidden_nodes_from_matrix(
                                        matrix=full_adjacency_matrix,
                                        observed_nodes=observed_nodes
                                    )
                                    observed_weighted_adjacency = self.remove_hidden_nodes_from_matrix(
                                        matrix=full_weighted_adjacency,
                                        observed_nodes=observed_nodes
                                    )

                                    frames.append(observed_dataframe)
                                    true_causal_matrices.append(observed_adjacency_matrix)
                                    true_weighted_causal_matrices.append(observed_weighted_adjacency)
                                    full_frames.append(full_dataframe)

                                    feature_columns = [col for col in range(0, len(observed_nodes))
                                                       if col != target_column]
                                    excluded_target_nodes = sorted(set(roles['Protected_Unfair_Nodes'] +
                                                                       roles['Hidden_U_Nodes']))

                                    seed_runs.append(seed_run)
                                    nr_nodes_array.append(len(observed_nodes))
                                    nr_base_nodes_array.append(nr_nodes)
                                    nr_nodes_full_array.append(full_num_nodes)
                                    connectivity_array.append(connectivity)
                                    number_edges_array.append(int(np.sum(observed_adjacency_matrix)))
                                    number_edges_full_array.append(int(np.sum(full_adjacency_matrix)))
                                    beta_upper_array.append(beta_upper_limit)
                                    data_scale_array.append(data_scale)
                                    s_obs_node_array.append(roles['S_Obs_Node'])
                                    s_unobs_nodes_array.append(roles['S_Unobs_Nodes'])
                                    protected_unfair_nodes_array.append(roles['Protected_Unfair_Nodes'])
                                    target_node_array.append(roles['Target_Node'])
                                    target_column_array.append(target_column)
                                    target_r2_score_array.append(float(r2_scores[roles['Target_Node']]))
                                    r2_scores_array.append(r2_scores.tolist())
                                    candidate_target_nodes_array.append(roles['Candidate_Target_Nodes'])
                                    excluded_target_nodes_array.append(excluded_target_nodes)
                                    x_nodes_array.append(roles['X_Nodes'])
                                    feature_columns_array.append(feature_columns)
                                    observed_nodes_array.append(observed_nodes)
                                    hidden_u_nodes_array.append(roles['Hidden_U_Nodes'])
                                    u_ratio_array.append(float(u_ratio))
                                    num_u_nodes_array.append(len(roles['Hidden_U_Nodes']))
                                    u_confounder_nodes_array.append(role_nodes['confounder'])
                                    u_mediator_nodes_array.append(role_nodes['mediator'])
                                    u_collider_nodes_array.append(role_nodes['collider'])
                                    s_unobs_children_array.append(s_unobs_children)
                                    hidden_selection_array.append(str(u_child_selection).lower())
                                    function_transformation_array.append(self._transformation_name(function_transformation))
                                    target_type_array.append(self.target_type)
                                    target_binarization_method_array.append(self.target_binarization_method)
                                    target_binarization_quantile_array.append(float(self.target_binarization_quantile))
                                    target_binarization_threshold_array.append(float(threshold_used))
                                    target_positive_rate_array.append(float(positive_rate))
                                    node_metadata_array.append(self._metadata_list(node_metadata_by_node))
                                    observed_node_metadata_array.append(
                                        self._observed_metadata_list(
                                            metadata_by_node=node_metadata_by_node,
                                            observed_nodes=observed_nodes,
                                        )
                                    )
                                    raw_data_available_array.append(True)

        all_datasets_frame = pd.DataFrame({'Seed_Run': np.array(seed_runs),
                                    'Number_Nodes': np.array(nr_nodes_array),
                                    'Number_Base_Nodes_SF': np.array(nr_base_nodes_array),
                                    'Number_Nodes_Full': np.array(nr_nodes_full_array),
                                    'K': np.array(connectivity_array),
                                    'Number_Edges': np.array(number_edges_array),
                                    'Number_Edges_Full': np.array(number_edges_full_array),
                                    'Transformation_Function': np.array(function_transformation_array),
                                    'Beta_Upper_Limit': np.array(beta_upper_array),
                                    'Data_Scale': np.array(data_scale_array),
                                    'Graph_Type': np.array([graph_type_key] * len(data_scale_array)),
                                    'S_Obs_Node': np.array(s_obs_node_array),
                                    'S_Unobs_Nodes': s_unobs_nodes_array,
                                    'Protected_Unfair_Nodes': protected_unfair_nodes_array,
                                    'Target_Node': np.array(target_node_array),
                                    'Target_Column': np.array(target_column_array),
                                    'Target_R2_Score': np.array(target_r2_score_array),
                                    'R2_Scores': r2_scores_array,
                                    'Candidate_Target_Nodes': candidate_target_nodes_array,
                                    'Excluded_Target_Nodes': excluded_target_nodes_array,
                                    'X_Nodes': x_nodes_array,
                                    'Feature_Columns': feature_columns_array,
                                    'Observed_Nodes_Full': observed_nodes_array,
                                    'Hidden_U_Nodes_Full': hidden_u_nodes_array,
                                    'U_Ratio': np.array(u_ratio_array),
                                    'Number_U_Nodes': np.array(num_u_nodes_array),
                                    'U_Confounder_Nodes': u_confounder_nodes_array,
                                    'U_Mediator_Nodes': u_mediator_nodes_array,
                                    'U_Collider_Nodes': u_collider_nodes_array,
                                    'S_Unobs_Children_X': s_unobs_children_array,
                                    'Hidden_Mechanism_Selection': np.array(hidden_selection_array),
                                    'SF_Target_Selection_Method': np.array([sf_target_selection_method] * len(data_scale_array)),
                                    'Target_Type': np.array(target_type_array),
                                    'Target_Binarization_Method': np.array(target_binarization_method_array),
                                    'Target_Binarization_Quantile': np.array(target_binarization_quantile_array),
                                    'Target_Binarization_Threshold': np.array(target_binarization_threshold_array),
                                    'Target_Positive_Rate': np.array(target_positive_rate_array),
                                    'Node_Metadata': node_metadata_array,
                                    'Observed_Node_Metadata': observed_node_metadata_array,
                                    'Raw_Data_Available': np.array(raw_data_available_array),
                                    'Core_Causal_Path': np.array(['SF_R2_Y_with_multi_U_roles'] * len(data_scale_array))})

        return [all_datasets_frame,
               true_causal_matrices,
               true_weighted_causal_matrices,
               frames,
               full_true_causal_matrices,
               full_true_weighted_causal_matrices,
               full_frames]

    def sanity_check_sf_r2_target_selection(self,
                                            nr_nodes=None,
                                            connectivity=None,
                                            seed=0):
        """
        Generate one SF graph and verify the R2-weighted target-selection rule.
        """
        np.random.seed(seed)
        if nr_nodes is None:
            nr_nodes = self.nr_nodes_values[0]
        if connectivity is None:
            connectivity = self._scale_free_connectivity_from_edge_density(
                nr_nodes=nr_nodes,
                edge_density=self.edge_desnity_values[0]
            )

        base_graph = self._generate_base_graph(graph_type_key='SF',
                                             nr_nodes=nr_nodes,
                                             connectivity=connectivity,
                                             seed=seed)
        roles = self.assign_s_x_y_u_roles(num_base_nodes=nr_nodes,
                                         target_node=None,
                                         u_ratio=0.0)
        target_node, r2_scores = self._select_sf_target(
            base_graph=base_graph,
            roles=roles,
            beta_upper_limit=self.betta_upper_limit_values[0],
            causal_transformation=self.nonlinearities[0],
            graph_type='SF',
            sf_target_selection_method='r2'
        )

        candidate_nodes = roles['Candidate_Target_Nodes']
        selection_probabilities = r2_probabilities_for_candidates(
            r2_scores=r2_scores,
            candidate_nodes=candidate_nodes
        )

        assert target_node in candidate_nodes
        assert target_node not in roles['Protected_Unfair_Nodes']
        assert np.all(selection_probabilities >= 0.0)
        assert np.isclose(np.sum(selection_probabilities), 1.0)

        return {'target_node': target_node,
                'r2_scores': r2_scores,
                'candidate_nodes': candidate_nodes,
                'selection_probabilities': selection_probabilities,
                'excluded_nodes': roles['Protected_Unfair_Nodes']}

    def save_data(self,
                    frames_descriptions,
                    true_causal_matrices,
                    true_weighted_causal_matrices,
                    frames,
                    nonlinear_pattern,
                    graph_type,
                    sample_size,
                    save_path):
        """Save generated datasets, grouped by observed model-fed node count."""
        data_by_nodes = {}

        for frame_index in range(0, frames_descriptions.shape[0]):
            frame_description = frames_descriptions.loc[[frame_index]].values.tolist()[0]

            current_adjacency_matrix = true_causal_matrices[frame_index]
            current_weighted_adjacency = true_weighted_causal_matrices[frame_index]
            current_dataframe = frames[frame_index]

            if sample_size == 'Small_Sample_Size':
                small_sample_size = int(current_dataframe.shape[0] / 10)
                current_dataframe = current_dataframe.sample(small_sample_size)

            node_count = int(current_dataframe.shape[1])
            if node_count not in data_by_nodes:
                data_by_nodes[node_count] = []

            data_by_nodes[node_count].append([frame_description,
                                              current_adjacency_matrix,
                                              current_weighted_adjacency,
                                              current_dataframe])

        for node_count, node_data in data_by_nodes.items():
            if len(node_data) != 0:
                with open(save_path + graph_type + '_' + sample_size + '_Datasets_' +
                          nonlinear_pattern + '_' + str(node_count) + '_nodes.pkl', 'wb') as f:
                    pickle.dump(node_data, f)
