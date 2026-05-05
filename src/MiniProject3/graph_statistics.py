import argparse
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
ORIGINAL_CWD = Path.cwd()
os.chdir(SCRIPT_DIR)
import graph_vae as gv
os.chdir(ORIGINAL_CWD)


def pyg_graph_to_networkx(data):
    graph = nx.Graph()
    graph.add_nodes_from(range(data.num_nodes))

    edges = data.edge_index.t().detach().cpu().tolist()
    for source, target in edges:
        if source != target:
            graph.add_edge(int(source), int(target))

    return graph


def adjacency_to_networkx(adjacency):
    matrix = adjacency.detach().cpu().numpy()
    return nx.from_numpy_array(matrix)


def training_graphs():
    return [pyg_graph_to_networkx(gv.dataset[int(index)]) for index in gv.train_dataset.indices]


def density_by_node_count(graphs):
    densities = defaultdict(list)
    for graph in graphs:
        num_nodes = graph.number_of_nodes()
        if num_nodes <= 1:
            densities[num_nodes].append(0.0)
        else:
            densities[num_nodes].append(nx.density(graph))

    return {num_nodes: float(np.mean(values)) for num_nodes, values in densities.items()}


def sample_baseline_graphs(num_samples, graphs, seed):
    rng = np.random.default_rng(seed)
    node_counts = [graph.number_of_nodes() for graph in graphs]
    densities = density_by_node_count(graphs)

    sampled_graphs = []
    sampled_sizes = rng.choice(node_counts, size=num_samples, replace=True)
    for num_nodes in sampled_sizes:
        num_nodes = int(num_nodes)
        probability = densities[num_nodes]
        upper = rng.random((num_nodes, num_nodes))
        adjacency = np.triu(upper < probability, k=1).astype(np.float32)
        adjacency = adjacency + adjacency.T
        sampled_graphs.append(nx.from_numpy_array(adjacency))

    return sampled_graphs


def build_vae_model(device, latent_dim, rounds, model_path):
    state_dim = 64
    hidden_dim = 256

    prior = gv.GaussianPrior(latent_dim)
    encoder_net = gv.EncoderNet(gv.node_feature_dim, state_dim, rounds, latent_dim)
    decoder_net = gv.DecoderNet(latent_dim, hidden_dim, gv.N_max)
    encoder = gv.GraphEncoder(encoder_net)
    decoder = gv.BernoulliDecoder(decoder_net)
    model = gv.GraphVAE(prior, encoder, decoder, gv.N_max).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def sample_vae_graphs(model, num_samples):
    node_sizes = gv.sample_num_nodes(n_samples=num_samples)
    return [adjacency_to_networkx(adjacency) for adjacency in model.sample(node_sizes)]


def node_degree_values(graphs):
    values = []
    for graph in graphs:
        values.extend(float(degree) for _, degree in graph.degree())
    return values


def clustering_values(graphs):
    values = []
    for graph in graphs:
        values.extend(float(value) for value in nx.clustering(graph).values())
    return values


def eigenvector_values_for_graph(graph):
    if graph.number_of_nodes() == 0:
        return []

    values = {}
    for component_nodes in nx.connected_components(graph):
        component = graph.subgraph(component_nodes).copy()
        if component.number_of_edges() == 0:
            for node in component.nodes():
                values[node] = 0.0
            continue

        try:
            component_values = nx.eigenvector_centrality(component, max_iter=1000, tol=1e-6)
        except nx.PowerIterationFailedConvergence:
            component_values = nx.eigenvector_centrality_numpy(component)

        values.update({node: float(value) for node, value in component_values.items()})

    return [values[node] for node in graph.nodes()]


def eigenvector_values(graphs):
    values = []
    for graph in graphs:
        values.extend(eigenvector_values_for_graph(graph))
    return values


def collect_statistics(graphs):
    return {
        'Node degree': node_degree_values(graphs),
        'Clustering coefficient': clustering_values(graphs),
        'Eigenvector centrality': eigenvector_values(graphs),
    }


def degree_bins(*series):
    flattened = [int(value) for values in series for value in values]
    min_degree = min(flattened)
    max_degree = max(flattened)
    return np.arange(min_degree - 0.5, max_degree + 1.5, 1.0)


def continuous_bins(*series, num_bins):
    flattened = [float(value) for values in series for value in values]
    minimum = min(flattened)
    maximum = max(flattened)
    if np.isclose(minimum, maximum):
        minimum -= 0.05
        maximum += 0.05
    return np.linspace(minimum, maximum, num_bins + 1)


def plot_statistics_grid(statistics_by_source, output_path, num_bins):
    metrics = ['Node degree', 'Clustering coefficient', 'Eigenvector centrality']
    sources = ['Empirical train', 'Baseline', 'Graph VAE']
    colors = {
        'Empirical train': '#355070',
        'Baseline': '#6d597a',
        'Graph VAE': '#b56576',
    }

    figure, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)

    for row, metric in enumerate(metrics):
        metric_series = [statistics_by_source[source][metric] for source in sources]
        if metric == 'Node degree':
            bins = degree_bins(*metric_series)
        else:
            bins = continuous_bins(*metric_series, num_bins=num_bins)

        for col, source in enumerate(sources):
            axis = axes[row, col]
            axis.hist(
                statistics_by_source[source][metric],
                bins=bins,
                color=colors[source],
                edgecolor='black',
                alpha=0.85,
            )
            if row == 0:
                axis.set_title(source)
            if col == 0:
                axis.set_ylabel(metric)
            axis.set_xlabel(metric)
            axis.set_yscale('linear')

    figure.suptitle('Graph Statistics Comparison', fontsize=16)
    figure.savefig(output_path, dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'xpu', 'mps'])
    parser.add_argument('--latent-dim', type=int, default=32)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--num-samples', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bins', type=int, default=20)
    parser.add_argument('--model', type=Path, default=SCRIPT_DIR / 'smodelGNN.pt')
    parser.add_argument('--output', type=Path, default=SCRIPT_DIR / 'graph_statistics_grid.png')
    args = parser.parse_args()

    device = gv.resolve_device(args.device)
    train_graph_set = training_graphs()
    baseline_graph_set = sample_baseline_graphs(args.num_samples, train_graph_set, args.seed)

    model = build_vae_model(
        device=device,
        latent_dim=args.latent_dim,
        rounds=args.rounds,
        model_path=args.model,
    )
    vae_graph_set = sample_vae_graphs(model, args.num_samples)

    statistics_by_source = {
        'Empirical train': collect_statistics(train_graph_set),
        'Baseline': collect_statistics(baseline_graph_set),
        'Graph VAE': collect_statistics(vae_graph_set),
    }

    plot_statistics_grid(statistics_by_source, args.output, args.bins)
    print(f'Saved statistics grid to {args.output}')


if __name__ == '__main__':
    main()
