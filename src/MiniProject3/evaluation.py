from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import networkx as nx
import torch


class IsomorphismIndex:
    def __init__(self, graphs: list[nx.Graph] | None = None):
        self.buckets: dict[tuple[int, int, str], list[nx.Graph]] = defaultdict(list)
        for graph in graphs or []:
            self.add(graph)

    def add(self, graph: nx.Graph) -> None:
        self.buckets[graph_signature(graph)].append(graph)

    def contains(self, graph: nx.Graph) -> bool:
        candidates = self.buckets.get(graph_signature(graph), [])
        return any(nx.is_isomorphic(graph, candidate) for candidate in candidates)


def pyg_to_nx(data) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(int(data.num_nodes)))
    edges = data.edge_index.detach().cpu().t().tolist()
    graph.add_edges_from((min(u, v), max(u, v)) for u, v in edges if u != v)
    return graph


def adjacency_to_nx(adjacency: torch.Tensor) -> nx.Graph:
    adjacency = adjacency.detach().cpu()
    graph = nx.Graph()
    graph.add_nodes_from(range(adjacency.shape[0]))
    rows, cols = torch.nonzero(torch.triu(adjacency > 0.5, diagonal=1), as_tuple=True)
    graph.add_edges_from(zip(rows.tolist(), cols.tolist()))
    return graph


def graph_signature(graph: nx.Graph) -> tuple[int, int, str]:
    return (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        nx.weisfeiler_lehman_graph_hash(graph),
    )


def novelty_uniqueness_metrics(sampled_adjacencies: list[torch.Tensor], train_dataset) -> dict[str, float]:
    train_graphs = [pyg_to_nx(data) for data in train_dataset]
    train_index = IsomorphismIndex(train_graphs)
    generated_index = IsomorphismIndex()

    novel_samples = 0
    unique_samples = 0
    novel_unique_samples = 0

    for adjacency in sampled_adjacencies:
        graph = adjacency_to_nx(adjacency)
        is_novel = not train_index.contains(graph)
        is_unique = not generated_index.contains(graph)

        if is_novel:
            novel_samples += 1
        if is_unique:
            unique_samples += 1
            generated_index.add(graph)
            if is_novel:
                novel_unique_samples += 1

    total = len(sampled_adjacencies)
    return {
        "Novel (%)": 100.0 * novel_samples / total,
        "Unique (%)": 100.0 * unique_samples / total,
        "Novel and unique (%)": 100.0 * novel_unique_samples / total,
    }


def format_markdown_table(rows: list[dict[str, float | str]]) -> str:
    headers = ["Model", "Novel (%)", "Unique (%)", "Novel and unique (%)"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row[header]
            values.append(f"{value:.2f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def save_metrics(rows: list[dict[str, float | str]], csv_path: Path, markdown_path: Path) -> str:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)

    headers = ["Model", "Novel (%)", "Unique (%)", "Novel and unique (%)"]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)

    table = format_markdown_table(rows)
    markdown_path.write_text(table + "\n", encoding="utf-8")
    return table
