import matplotlib.pyplot as plt
import networkx as nx
import random
import yaml

from gym_duckietown.simulator import logger


class MapGraph:
    """Encapsulates a road-network graph and related path-planning utilities."""

    def __init__(self, yaml_path: str, node_name_offset={'row': +1, 'col': +1}):
        """Load graph from a YAML file containing edges."""
        self.G = self._load_from_yaml(yaml_path)
        self.node_name_offset = node_name_offset

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_from_yaml(self, yaml_path: str) -> nx.Graph:
        """Parse a YAML file and return the corresponding NetworkX graph."""
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
        edges = data.get('edges', [])
        G = nx.Graph()
        G.add_edges_from(edges)
        return G

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def nodes(self):
        """Return the nodes of the underlying graph."""
        return self.G.nodes()
    
    def node_name_from_node_coords(self, row: int, col: int) -> str:
        """Convert node coordinates (row, col) to a node name like 'T_2_3'."""
        return f"T_{row}_{col}"
    
    def node_name_to_tile_coords(self, node_name: str) -> dict:
        """Convert a node name like 'T_2_3' to tile coordinates (row=2+offset, col=3+offset)."""
        parts = node_name.split('_')
        if len(parts) != 3 or parts[0] != 'T':
            raise ValueError(f"Invalid node name format: {node_name}")
        try:
            row = int(parts[1]) + self.node_name_offset['row']
            col = int(parts[2]) + self.node_name_offset['col']
            return {'row': row, 'col': col}
        except ValueError:
            raise ValueError(f"Row and column must be integers in node name: {node_name}")
        
    def tile_coords_to_node_name(self, row: int, col: int) -> str:
        """Convert tile coordinates (row, col) to a node name like 'T_2_3'."""
        node_row = row - self.node_name_offset['row']
        node_col = col - self.node_name_offset['col']
        node_name = self.node_name_from_node_coords(node_row, node_col)
        return node_name
    
    def node_coords_to_tile_coords(self, row: int, col: int) -> dict:
        """Convert node coordinates to tile coordinates."""
        node_name = self.node_name_from_node_coords(row, col)
        tile_coords = self.node_name_to_tile_coords(node_name)
        return tile_coords

    def sample_random_start_finish_nodes(self, get_tile_fn) -> tuple:
        """Sample random start and finish nodes from the graph.

        The finish node is guaranteed not to be a direct neighbour of the
        start node so that a non-trivial path always exists.
        """

        def is_straight(node):
            coords = self.node_name_to_tile_coords(node)
            tile = get_tile_fn(coords['col'], coords['row'])
            tile_type = tile['kind']
            return tile_type == 'straight'

        nodes = list(self.G.nodes())
        straight_nodes = [node for node in nodes if is_straight(node)]
        if len(straight_nodes) < 2:
            raise ValueError("Graph must have at least 2 straight nodes to sample start and finish.")
        start_node = random.sample(straight_nodes, 1)[0]
        logger.debug(f"Sampled start node: {start_node}")
        non_neighbors = set(nx.non_neighbors(self.G, start_node))
        non_neighbors_and_straight = [node for node in non_neighbors if is_straight(node)]
        if len(non_neighbors_and_straight) == 0:
            logger.warning(
                f"No non-neighbors + straight node available for start node {start_node}. Cannot sample finish node."
            )
            return None
        finish_node = random.sample(non_neighbors_and_straight, 1)[0]
        logger.debug(f"Sampled finish node: {finish_node}")
        return start_node, finish_node

    def visualize(self):
        """Visualize the graph with improved styling and layout."""

        def get_position(node: str) -> tuple:
            """Map a node name (e.g. ``'T_2_3'``) to a 2-D plot coordinate.

            Uses ``(col, -row)`` so that the grid appears top-left-to-bottom-right
            when matplotlib's default y-axis orientation is later inverted.
            Returns ``(0, 0)`` for malformed node names.
            """
            parts = node.split('_')
            if len(parts) == 3:
                try:
                    row = int(parts[1])
                    col = int(parts[2])
                    return (col, -row)  # x=col, y=-row (row 0 at top)
                except ValueError:
                    return (0, 0)
            return (0, 0)

        G = self.G
        pos = {node: get_position(node) for node in G.nodes()}

        plt.figure(figsize=(18, 14))

        degree = dict(G.degree())
        node_sizes = [500 * (1 + degree[n] / max(degree.values())) for n in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.9)
        nx.draw_networkx_edges(G, pos, width=1.5, alpha=0.6, edge_color='gray')
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_color='black')

        for node, deg in degree.items():
            if deg > max(degree.values()) * 0.7:
                nx.draw_networkx_nodes(G, pos, nodelist=[node], node_size=700, node_color='orange', alpha=0.8)

        plt.grid(True, alpha=0.2, linestyle='--')
        plt.title('Grid-Based Graph Visualization (T_0_0 at Top-Left)', fontsize=16, pad=20)
        plt.axis('off')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        plt.savefig('iaa26_lab3_graph.jpg', dpi=300, bbox_inches='tight')
        print('Graph saved as "iaa26_lab3_graph.jpg"')
        plt.show()


if __name__ == '__main__':
    graph = MapGraph('iaa26_lab3_graph.yaml')
    graph.visualize()