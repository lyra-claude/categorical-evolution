"""
plot_video_figures.py -- Generate publication-quality figures for the GECCO 2026
oral video presentation (~19 min, 20 slides).

Figures:
  1. topology_networks.png       -- 5 migration topologies side-by-side (Slide 9)
  2. topology_networks_lambda2.png -- same, annotated with lambda_2 (Slide 10)
  3. lambda2_vs_diversity.png    -- scatter: lambda_2 vs mean final diversity (Slide 12)
  4. ring_star_inversion.png     -- n=5 vs n=7 inversion diagram (Slide 13)
  5. before_after_topology.png   -- star -> ring fix diagram (Slide 18)

Style matches the paper (plot_multi_domain.py / plot_fingerprints.py):
  - seaborn-v0_8-paper base, serif fonts, no top/right spines
  - RdBu diverging palette for topologies
  - 16:9 aspect ratio (12 x 6.75 in), 300 DPI PNG output

Output: experiments/video_figures/
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import networkx as nx
from pathlib import Path
from scipy.linalg import eigvalsh

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).parent
OUT_DIR = SCRIPT_DIR / "video_figures"
OUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Style (matches plot_multi_domain.py / plot_fingerprints.py)
# ---------------------------------------------------------------------------

TOPOLOGY_ORDER = ["none", "ring", "star", "random", "fully_connected"]

TOPOLOGY_LABELS = {
    "none": "None\n(isolated)",
    "ring": "Ring",
    "star": "Star",
    "random": "Random",
    "fully_connected": "Fully\nConnected",
}

TOPOLOGY_LABELS_SHORT = {
    "none": "None (isolated)",
    "ring": "Ring",
    "star": "Star",
    "random": "Random",
    "fully_connected": "Fully connected",
}

# Diverging RdBu palette -- cool (strict) to warm (lax)
TOPO_COLORS = {
    "none": "#2166AC",
    "ring": "#67A9CF",
    "star": "#5AB4AC",
    "random": "#F4A582",
    "fully_connected": "#B2182B",
}

# Lighter fill colors for nodes
TOPO_NODE_COLORS = {
    "none": "#D1E5F0",
    "ring": "#D1E5F0",
    "star": "#D4EEEB",
    "random": "#FDDBC7",
    "fully_connected": "#F4A582",
}


def setup_style():
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


# ---------------------------------------------------------------------------
# Topology graph builders (K=5 islands)
# ---------------------------------------------------------------------------

def build_topology_graph(name, n=5, seed=42):
    """Build a networkx graph for the given topology name."""
    if name == "none":
        G = nx.empty_graph(n)
    elif name == "ring":
        G = nx.cycle_graph(n)
    elif name == "star":
        G = nx.star_graph(n - 1)
    elif name == "random":
        # Erdos-Renyi with p=0.3, fixed seed for reproducibility
        rng = np.random.default_rng(seed)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        for i in range(n):
            for j in range(i + 1, n):
                if rng.random() < 0.5:
                    G.add_edge(i, j)
        # Ensure at least one edge for visual clarity
        if G.number_of_edges() == 0:
            G.add_edge(0, 1)
    elif name == "fully_connected":
        G = nx.complete_graph(n)
    else:
        raise ValueError(f"Unknown topology: {name}")
    return G


def compute_lambda2(G):
    """Compute algebraic connectivity (second-smallest eigenvalue of Laplacian)."""
    n = G.number_of_nodes()
    if n <= 1:
        return 0.0
    A = nx.to_numpy_array(G, dtype=float)
    D = np.diag(A.sum(axis=1))
    L = D - A
    eigvals = eigvalsh(L)
    return float(eigvals[1])


def get_circular_layout(G, center=(0, 0)):
    """Circular layout for consistent node positioning across topologies."""
    n = G.number_of_nodes()
    pos = {}
    for i in range(n):
        angle = 2 * np.pi * i / n - np.pi / 2  # start from top
        pos[i] = (center[0] + np.cos(angle), center[1] + np.sin(angle))
    return pos


# ---------------------------------------------------------------------------
# Figure 1 & 2: Topology network diagrams (Slides 9, 10)
# ---------------------------------------------------------------------------

def plot_topology_networks(annotate_lambda2=False):
    """Draw 5 topologies side-by-side as network graphs."""
    fig, axes = plt.subplots(1, 5, figsize=(12, 6.75 * 0.45))
    fig.subplots_adjust(wspace=0.05)

    K = 5  # number of islands

    for ax, topo in zip(axes, TOPOLOGY_ORDER):
        G = build_topology_graph(topo, n=K)
        pos = get_circular_layout(G)
        lam2 = compute_lambda2(G)
        color = TOPO_COLORS[topo]
        node_color = TOPO_NODE_COLORS[topo]

        # Draw edges
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color=color,
            width=2.5,
            alpha=0.8,
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_color=node_color,
            edgecolors=color,
            linewidths=2.0,
            node_size=500,
        )

        # Node labels
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            font_size=11,
            font_weight="bold",
            font_family="serif",
        )

        # Title
        title = TOPOLOGY_LABELS[topo]
        ax.set_title(title, fontsize=14, fontweight="bold", color=color, pad=10)

        # Lambda2 annotation
        if annotate_lambda2:
            lam2_str = f"$\\lambda_2 = {lam2:.2f}$"
            ax.text(
                0.5, -0.08, lam2_str,
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=12, color=color,
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor=color,
                    alpha=0.9,
                    linewidth=1.5,
                ),
            )

        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

    # Strict -> Lax arrow below all panels
    fig.text(
        0.15, 0.02, "strict (high diversity)",
        ha="left", va="center", fontsize=11,
        color="#2166AC", style="italic",
    )
    fig.text(
        0.85, 0.02, "lax (low diversity)",
        ha="right", va="center", fontsize=11,
        color="#B2182B", style="italic",
    )
    # Arrow
    arrow_ax = fig.add_axes([0.15, 0.01, 0.70, 0.02])
    arrow_ax.annotate(
        "", xy=(1, 0.5), xytext=(0, 0.5),
        arrowprops=dict(arrowstyle="->", color="0.5", linewidth=1.5),
    )
    arrow_ax.axis("off")

    suffix = "_lambda2" if annotate_lambda2 else ""
    out_path = OUT_DIR / f"topology_networks{suffix}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path.name} saved")
    return out_path


# ---------------------------------------------------------------------------
# Figure 3: Lambda2 vs Final Diversity scatter (Slide 12)
# ---------------------------------------------------------------------------

def load_diversity_data():
    """Load all domain CSVs and compute mean final diversity per topology."""
    domain_files = {
        "OneMax": ["experiment_e_raw.csv", "experiment_e_onemax.csv"],
        "Maze": ["experiment_e_maze.csv"],
        "Sorting Networks": ["experiment_e_sorting_network.csv"],
        "Graph Coloring": ["experiment_e_graph_coloring.csv"],
        "Knapsack": ["experiment_e_knapsack.csv"],
        "No Thanks!": ["experiment_e_nothanks.csv"],
        "Checkers": ["experiment_e_checkers.csv"],
    }

    all_final = {}
    for domain, files in domain_files.items():
        df = None
        for f in files:
            path = SCRIPT_DIR / f
            if path.exists():
                df = pd.read_csv(path)
                break
        if df is None:
            continue

        max_gen = df["generation"].max()
        final = df[df["generation"] == max_gen]
        for topo in TOPOLOGY_ORDER:
            vals = final[final["topology"] == topo]["hamming_diversity"]
            if len(vals) > 0:
                if topo not in all_final:
                    all_final[topo] = []
                all_final[topo].append(vals.mean())

    # Mean across domains
    mean_diversity = {}
    for topo in TOPOLOGY_ORDER:
        if topo in all_final:
            mean_diversity[topo] = np.mean(all_final[topo])

    return mean_diversity


def plot_lambda2_vs_diversity():
    """Scatter plot: lambda_2 vs mean final diversity across topologies."""
    fig, ax = plt.subplots(figsize=(12, 6.75))

    K = 5
    mean_diversity = load_diversity_data()

    xs, ys, labels, colors = [], [], [], []
    for topo in TOPOLOGY_ORDER:
        G = build_topology_graph(topo, n=K)
        lam2 = compute_lambda2(G)
        div = mean_diversity.get(topo, None)
        if div is None:
            continue
        xs.append(lam2)
        ys.append(div)
        labels.append(TOPOLOGY_LABELS_SHORT[topo])
        colors.append(TOPO_COLORS[topo])

    # Scatter points
    for i, (x, y, label, color) in enumerate(zip(xs, ys, labels, colors)):
        ax.scatter(
            x, y,
            s=250, c=color, edgecolors="white", linewidths=2,
            zorder=5,
        )
        # Label offset to avoid overlap
        offset_x, offset_y = 0.08, 0.003
        if label == "Star":
            offset_y = -0.006
        elif label == "Ring":
            offset_y = 0.005
        ax.annotate(
            label,
            (x, y),
            xytext=(x + offset_x, y + offset_y),
            fontsize=13, fontweight="bold", color=color,
            arrowprops=dict(arrowstyle="-", color=color, alpha=0.5),
        )

    # Trend line (linear fit through the points)
    if len(xs) >= 2:
        z = np.polyfit(xs, ys, 1)
        x_line = np.linspace(min(xs) - 0.1, max(xs) + 0.3, 100)
        y_line = np.polyval(z, x_line)
        ax.plot(x_line, y_line, '--', color='0.6', linewidth=1.5,
                alpha=0.7, zorder=1, label=f"Linear trend")

    ax.set_xlabel("$\\lambda_2$ (algebraic connectivity)", fontsize=14)
    ax.set_ylabel("Mean final diversity (across all domains)", fontsize=14)
    ax.set_title(
        "Spectral Prediction: $\\lambda_2$ Determines Diversity",
        fontsize=16, fontweight="bold",
    )

    # Add annotation about monotonicity
    ax.text(
        0.98, 0.95,
        "Higher connectivity\n$\\Rightarrow$ lower diversity",
        transform=ax.transAxes,
        ha="right", va="top",
        fontsize=12, color="0.4",
        style="italic",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                  edgecolor="0.8", alpha=0.9),
    )

    ax.set_xlim(-0.3, max(xs) + 0.8)

    out_path = OUT_DIR / "lambda2_vs_diversity.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path.name} saved")
    return out_path


# ---------------------------------------------------------------------------
# Figure 4: Ring vs Star inversion at n=7 (Slide 13)
# ---------------------------------------------------------------------------

def plot_ring_star_inversion():
    """Two panels: n=5 vs n=7, showing ring/star with lambda2 and diversity."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.75))

    configs = [
        {"n": 5, "label": "$K = 5$ islands", "ax_idx": 0},
        {"n": 7, "label": "$K = 7$ islands", "ax_idx": 1},
    ]

    # Load n=7 data if available
    n7_path = SCRIPT_DIR / "experiment_e_maze_n7.csv"
    n7_data = {}
    if n7_path.exists():
        df7 = pd.read_csv(n7_path)
        max_gen = df7["generation"].max()
        final7 = df7[df7["generation"] == max_gen]
        for topo in ["ring", "star"]:
            vals = final7[final7["topology"] == topo]["hamming_diversity"]
            if len(vals) > 0:
                n7_data[topo] = vals.mean()

    # Load n=5 data (maze)
    n5_path = SCRIPT_DIR / "experiment_e_maze.csv"
    n5_data = {}
    if n5_path.exists():
        df5 = pd.read_csv(n5_path)
        max_gen = df5["generation"].max()
        final5 = df5[df5["generation"] == max_gen]
        for topo in ["ring", "star"]:
            vals = final5[final5["topology"] == topo]["hamming_diversity"]
            if len(vals) > 0:
                n5_data[topo] = vals.mean()

    for cfg in configs:
        n = cfg["n"]
        ax = axes[cfg["ax_idx"]]

        # Build graphs
        G_ring = nx.cycle_graph(n)
        G_star = nx.star_graph(n - 1)
        lam2_ring = compute_lambda2(G_ring)
        lam2_star = compute_lambda2(G_star)

        # Layout: ring on left half, star on right half
        pos_ring = {}
        for i in range(n):
            angle = 2 * np.pi * i / n - np.pi / 2
            pos_ring[i] = (-1.8 + 0.8 * np.cos(angle), 0.8 * np.sin(angle))

        pos_star = {}
        for i in range(n):
            angle = 2 * np.pi * i / n - np.pi / 2
            pos_star[i] = (1.8 + 0.8 * np.cos(angle), 0.8 * np.sin(angle))

        # Draw ring
        nx.draw_networkx_edges(G_ring, pos_ring, ax=ax,
                               edge_color=TOPO_COLORS["ring"], width=2.5, alpha=0.8)
        nx.draw_networkx_nodes(G_ring, pos_ring, ax=ax,
                               node_color=TOPO_NODE_COLORS["ring"],
                               edgecolors=TOPO_COLORS["ring"],
                               linewidths=2.0, node_size=350)
        nx.draw_networkx_labels(G_ring, pos_ring, ax=ax, font_size=9,
                                font_weight="bold", font_family="serif")

        # Draw star
        nx.draw_networkx_edges(G_star, pos_star, ax=ax,
                               edge_color=TOPO_COLORS["star"], width=2.5, alpha=0.8)
        nx.draw_networkx_nodes(G_star, pos_star, ax=ax,
                               node_color=TOPO_NODE_COLORS["star"],
                               edgecolors=TOPO_COLORS["star"],
                               linewidths=2.0, node_size=350)
        nx.draw_networkx_labels(G_star, pos_star, ax=ax, font_size=9,
                                font_weight="bold", font_family="serif")

        # Lambda2 annotations below each network
        ax.text(-1.8, -1.35, f"Ring\n$\\lambda_2 = {lam2_ring:.2f}$",
                ha="center", va="top", fontsize=13, fontweight="bold",
                color=TOPO_COLORS["ring"])
        ax.text(1.8, -1.35, f"Star\n$\\lambda_2 = {lam2_star:.2f}$",
                ha="center", va="top", fontsize=13, fontweight="bold",
                color=TOPO_COLORS["star"])

        # Diversity comparison
        data = n5_data if n == 5 else n7_data
        if data:
            ring_div = data.get("ring", 0)
            star_div = data.get("star", 0)
            winner = "Ring" if ring_div > star_div else "Star"
            winner_color = TOPO_COLORS["ring"] if ring_div > star_div else TOPO_COLORS["star"]

            # Comparison text
            comparison = (
                f"Ring: {ring_div:.4f}\n"
                f"Star: {star_div:.4f}\n"
            )
            # Winner indicator
            if ring_div > star_div:
                indicator = "Ring > Star"
            else:
                indicator = "Star > Ring"

            ax.text(0, -1.35, comparison,
                    ha="center", va="top", fontsize=11, color="0.3",
                    family="monospace")
            ax.text(0, -1.85, indicator,
                    ha="center", va="top", fontsize=14, fontweight="bold",
                    color=winner_color,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor=winner_color, alpha=0.9, linewidth=2))

        # VS text between the two networks
        ax.text(0, 0, "vs", ha="center", va="center", fontsize=16,
                fontweight="bold", color="0.5", style="italic")

        ax.set_title(cfg["label"], fontsize=16, fontweight="bold", pad=15)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-2.4, 1.5)
        ax.set_aspect("equal")
        ax.axis("off")

    fig.suptitle(
        "Ring vs Star: Spectral Inversion at Larger $K$",
        fontsize=18, fontweight="bold", y=0.98,
    )

    out_path = OUT_DIR / "ring_star_inversion.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path.name} saved")
    return out_path


# ---------------------------------------------------------------------------
# Figure 5: Before/After topology fix (Slide 18)
# ---------------------------------------------------------------------------

def plot_before_after_topology():
    """Star (problematic) -> Ring (fixed) with arrow and annotations."""
    fig, ax = plt.subplots(figsize=(12, 6.75))

    K = 5

    G_star = nx.star_graph(K - 1)
    G_ring = nx.cycle_graph(K)
    lam2_star = compute_lambda2(G_star)
    lam2_ring = compute_lambda2(G_ring)

    # Positions: star on left, ring on right
    pos_star = {}
    for i in range(K):
        angle = 2 * np.pi * i / K - np.pi / 2
        pos_star[i] = (-2.8 + 1.0 * np.cos(angle), 1.0 * np.sin(angle))

    pos_ring = {}
    for i in range(K):
        angle = 2 * np.pi * i / K - np.pi / 2
        pos_ring[i] = (2.8 + 1.0 * np.cos(angle), 1.0 * np.sin(angle))

    # Draw star (left, "before")
    nx.draw_networkx_edges(G_star, pos_star, ax=ax,
                           edge_color=TOPO_COLORS["star"], width=3.0, alpha=0.8)
    nx.draw_networkx_nodes(G_star, pos_star, ax=ax,
                           node_color=TOPO_NODE_COLORS["star"],
                           edgecolors=TOPO_COLORS["star"],
                           linewidths=2.5, node_size=600)
    nx.draw_networkx_labels(G_star, pos_star, ax=ax, font_size=12,
                            font_weight="bold", font_family="serif")

    # Draw ring (right, "after")
    nx.draw_networkx_edges(G_ring, pos_ring, ax=ax,
                           edge_color=TOPO_COLORS["ring"], width=3.0, alpha=0.8)
    nx.draw_networkx_nodes(G_ring, pos_ring, ax=ax,
                           node_color=TOPO_NODE_COLORS["ring"],
                           edgecolors=TOPO_COLORS["ring"],
                           linewidths=2.5, node_size=600)
    nx.draw_networkx_labels(G_ring, pos_ring, ax=ax, font_size=12,
                            font_weight="bold", font_family="serif")

    # Labels
    ax.text(-2.8, -1.6, "Star Topology",
            ha="center", fontsize=16, fontweight="bold",
            color=TOPO_COLORS["star"])
    ax.text(-2.8, -2.0, f"$\\lambda_2 = {lam2_star:.2f}$",
            ha="center", fontsize=14, color=TOPO_COLORS["star"])
    ax.text(-2.8, -2.4, "Hub bottleneck\nPremature convergence",
            ha="center", fontsize=11, color="0.4", style="italic")

    ax.text(2.8, -1.6, "Ring Topology",
            ha="center", fontsize=16, fontweight="bold",
            color=TOPO_COLORS["ring"])
    ax.text(2.8, -2.0, f"$\\lambda_2 = {lam2_ring:.2f}$",
            ha="center", fontsize=14, color=TOPO_COLORS["ring"])
    ax.text(2.8, -2.4, "Distributed flow\nDiversity preserved",
            ha="center", fontsize=11, color="0.4", style="italic")

    # "BEFORE" and "AFTER" headers
    ax.text(-2.8, 1.9, "BEFORE", ha="center", fontsize=14, fontweight="bold",
            color="#E15759",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#FDDBC7",
                      edgecolor="#E15759", alpha=0.9, linewidth=1.5))
    ax.text(2.8, 1.9, "AFTER", ha="center", fontsize=14, fontweight="bold",
            color="#59A14F",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#D4EEDB",
                      edgecolor="#59A14F", alpha=0.9, linewidth=1.5))

    # Arrow between topologies
    ax.annotate(
        "",
        xy=(1.3, 0), xytext=(-1.3, 0),
        arrowprops=dict(
            arrowstyle="-|>",
            color="0.3",
            linewidth=3,
            mutation_scale=25,
        ),
    )
    ax.text(0, 0.35, "Topology\nSwitch",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
            color="0.3")

    # Load diversity improvement data
    n5_path = SCRIPT_DIR / "experiment_e_maze.csv"
    if n5_path.exists():
        df = pd.read_csv(n5_path)
        max_gen = df["generation"].max()
        final = df[df["generation"] == max_gen]
        star_div = final[final["topology"] == "star"]["hamming_diversity"].mean()
        ring_div = final[final["topology"] == "ring"]["hamming_diversity"].mean()
        improvement = ((ring_div - star_div) / star_div) * 100

        ax.text(0, -1.2,
                f"Diversity: {star_div:.4f} $\\rightarrow$ {ring_div:.4f}",
                ha="center", fontsize=13, color="0.3")
        sign = "+" if improvement > 0 else ""
        ax.text(0, -1.7,
                f"{sign}{improvement:.1f}% diversity change",
                ha="center", fontsize=15, fontweight="bold",
                color="#59A14F" if improvement > 0 else "#E15759",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#59A14F" if improvement > 0 else "#E15759",
                          alpha=0.9, linewidth=2))

    ax.set_title(
        "Practical Impact: Topology Selection Matters",
        fontsize=18, fontweight="bold", pad=20,
    )

    ax.set_xlim(-4.5, 4.5)
    ax.set_ylim(-3.0, 2.5)
    ax.set_aspect("equal")
    ax.axis("off")

    out_path = OUT_DIR / "before_after_topology.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  -> {out_path.name} saved")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    setup_style()
    print("Generating GECCO 2026 video figures...\n")

    generated = []
    errors = []

    # Figure 1: Topology networks (Slide 9)
    print("Figure 1: Topology network diagrams (Slide 9)...")
    try:
        generated.append(plot_topology_networks(annotate_lambda2=False))
    except Exception as e:
        errors.append(f"topology_networks: {e}")
        print(f"  ERROR: {e}")

    # Figure 2: Topology networks with lambda2 (Slide 10)
    print("Figure 2: Topology networks + lambda2 (Slide 10)...")
    try:
        generated.append(plot_topology_networks(annotate_lambda2=True))
    except Exception as e:
        errors.append(f"topology_networks_lambda2: {e}")
        print(f"  ERROR: {e}")

    # Figure 3: Lambda2 vs diversity scatter (Slide 12)
    print("Figure 3: Lambda2 vs final diversity (Slide 12)...")
    try:
        generated.append(plot_lambda2_vs_diversity())
    except Exception as e:
        errors.append(f"lambda2_vs_diversity: {e}")
        print(f"  ERROR: {e}")

    # Figure 4: Ring vs Star inversion (Slide 13)
    print("Figure 4: Ring vs Star inversion n=5 vs n=7 (Slide 13)...")
    try:
        generated.append(plot_ring_star_inversion())
    except Exception as e:
        errors.append(f"ring_star_inversion: {e}")
        print(f"  ERROR: {e}")

    # Figure 5: Before/After topology fix (Slide 18)
    print("Figure 5: Before/After topology fix (Slide 18)...")
    try:
        generated.append(plot_before_after_topology())
    except Exception as e:
        errors.append(f"before_after_topology: {e}")
        print(f"  ERROR: {e}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  {len(generated)} figures generated successfully")
    for p in generated:
        print(f"    {p}")
    if errors:
        print(f"\n  {len(errors)} errors:")
        for e in errors:
            print(f"    {e}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
