# coding=utf-8
import torch
from collections import defaultdict
import json

# class GraphEncoder:
#     def __init__(self):
#         pass
    
#     def encode(self, edge_index, mask=None, num_nodes=None):
#         """
#         Encode graph data into a textual description.
        
#         Args:
#             edge_index (torch.Tensor): Tensor of shape (2, E) representing edges in the graph.
#             mask (torch.Tensor, optional): Boolean tensor of shape (N,) indicating which nodes to include.
#             num_nodes (int, optional): Number of nodes in the graph. If not provided, inferred from edge_index.
        
#         Returns:
#             str: Textual description of the graph structure for unmasked nodes.
#         """
#         if num_nodes is None:
#             num_nodes = edge_index.max().item() + 1 if edge_index.numel() > 0 else 0
            
#         # Build adjacency list
#         adj_dict = defaultdict(list)
#         for src, dst in edge_index.t().tolist():
#             adj_dict[src].append(dst)
#             adj_dict[dst].append(src)  # Since edges are bidirectional
        
#         # Deduplicate and sort neighbors
#         for node in adj_dict:
#             adj_dict[node] = sorted(list(set(adj_dict[node])))
        
#         # Determine nodes to describe
#         if mask is not None:
#             nodes_to_describe = [i for i in range(num_nodes) if mask[i].item()]
#         else:
#             nodes_to_describe = list(range(num_nodes))
        
#         if not nodes_to_describe:
#             return "G describes an empty graph with no visible nodes."
            
#         # Build description
#         description = f"G describes a graph among {', '.join(map(str, nodes_to_describe))}. In this graph:"
#         for node in nodes_to_describe:
#             neighbors = adj_dict.get(node, [])
#             if neighbors:
#                 description += f"\n- Node {node} is connected to nodes {', '.join(map(str, neighbors))}."
#             else:
#                 description += f"\n- Node {node} has no connections."
                
#         return description

class GraphEncoder:
    """
    Encodes graph data into various textual descriptions suitable for LLMs,
    balancing understandability and token efficiency.
    """
    def __init__(self, directed=False):
        """
        Initializes the GraphEncoder.

        Args:
            directed (bool): Whether to treat the graph as directed.
                             If False (default), edges are treated as bidirectional
                             when building representations like adjacency lists.
                             If True, edges `(u, v)` are interpreted as `u -> v`.
        """
        self.directed = directed

    def _build_adj_list(self, edge_index, num_nodes):
        """Helper to build an adjacency list dictionary."""
        adj = defaultdict(list)
        if edge_index.numel() == 0:
            return adj # Return empty dict for empty edge_index

        edges = edge_index.t().tolist()
        processed_edges = set() # Used for undirected case to avoid duplicates in adj list creation

        for src, dst in edges:
            # Ensure nodes are within expected range if num_nodes is accurate
            # This basic check helps catch potential inconsistencies but isn't exhaustive
            if src >= num_nodes or dst >= num_nodes:
                 print(f"Warning: Edge ({src}, {dst}) contains node index out of expected range {num_nodes}. Skipping.")
                 continue

            if self.directed:
                adj[src].append(dst)
            else:
                # Add edges in both directions for undirected representation
                # Use processed_edges set to only add each pair once internally if needed,
                # but defaultdict handles multiple appends correctly anyway.
                # Just ensure sorting/deduplication later.
                 adj[src].append(dst)
                 adj[dst].append(src)

        # Deduplicate and sort neighbors for consistent output
        for node in adj:
            adj[node] = sorted(list(set(adj[node])))
        return adj

    def encode(self, edge_index, mask=None, num_nodes=None, style='adj_list'):
        """
        Encode graph data into a textual description based on the chosen style.

        Args:
            edge_index (torch.Tensor): Tensor of shape (2, E) representing edges.
            mask (torch.Tensor, optional): Boolean tensor of shape (N,) indicating nodes to include.
                                           Nodes NOT True in the mask are excluded from the description.
            num_nodes (int, optional): Total number of nodes (N). If None, inferred from edge_index and mask.
                                       Crucial if masked nodes or isolated nodes exist beyond max(edge_index).
            style (str): Encoding style. Options:
                         'adj_list': Compact adjacency list (e.g., "0: 1, 2\n1: 0"). Efficient.
                         'edge_list': Lists unique edges (e.g., "(0, 1)\n(0, 2)"). Efficient.
                         'natural': Verbose natural language description (original style).
                         'json': JSON object with nodes and edges. Structured but can be verbose.

        Returns:
            str: Textual description of the graph structure for unmasked nodes.
        """
        if num_nodes is None:
            if edge_index.numel() > 0:
                max_node_in_edges = edge_index.max().item()
            else:
                 max_node_in_edges = -1 # No edges
            if mask is not None:
                # If mask is provided, num_nodes should cover its length
                num_nodes = len(mask)
                # Ensure num_nodes is also large enough for edges if mask is shorter
                if max_node_in_edges >= num_nodes :
                     # This indicates an inconsistency, but we proceed assuming mask defines the universe
                     print(f"Warning: Max node index in edges ({max_node_in_edges}) exceeds mask length ({num_nodes}).")
                     # Alternatively, could take max(len(mask), max_node_in_edges + 1)
                     # Let's prioritize the mask length as defining N
            else:
                # No mask, infer from edges
                num_nodes = max_node_in_edges + 1 if max_node_in_edges != -1 else 0


        # Determine nodes to describe based on the mask
        if mask is not None:
             # Ensure mask length matches inferred or provided num_nodes if possible
            if len(mask) != num_nodes:
                 print(f"Warning: Mask length ({len(mask)}) differs from inferred/provided num_nodes ({num_nodes}). Using mask length.")
                 num_nodes = len(mask) # Prioritize mask length
            nodes_to_describe_set = {i for i, is_valid in enumerate(mask) if is_valid.item()}
        else:
            nodes_to_describe_set = set(range(num_nodes)) # Describe all nodes if no mask

        if not nodes_to_describe_set:
            return f"Graph ({style}): No nodes selected by mask."

        # --- Adjacency List Style ---
        if style == 'adj_list':
            adj = self._build_adj_list(edge_index, num_nodes)
            parts = []
            # Sort nodes for consistent output order
            sorted_nodes_to_describe = sorted(list(nodes_to_describe_set))
            for node in sorted_nodes_to_describe:
                # Only include neighbors that are *also* in the set of nodes to describe?
                # Decision: Show all neighbors, as it describes the node's connections accurately,
                # even if some neighbors are masked out. This matches original logic.
                # If you only want connections *within* the subgraph, filter neighbors:
                # neighbors = [n for n in adj.get(node, []) if n in nodes_to_describe_set]
                neighbors = adj.get(node, []) # Keep original logic: show all connections from described node
                neighbor_str = ', '.join(map(str, neighbors)) if neighbors else ""
                parts.append(f"{node}: {neighbor_str}")
            return f"Graph (Adj List):\n" + "\n".join(parts)

        # --- Edge List Style ---
        elif style == 'edge_list':
            if edge_index.numel() == 0:
                return f"Graph (Edge List):\nNo edges."

            edges_to_include = set()
            edge_list_parts = []
            edges = edge_index.t().tolist()

            for src, dst in edges:
                # Include edge only if BOTH src and dst are in the nodes_to_describe_set
                if src in nodes_to_describe_set and dst in nodes_to_describe_set:
                    if self.directed:
                        edge_tuple = (src, dst)
                        if edge_tuple not in edges_to_include:
                             edge_list_parts.append(f"{src} -> {dst}")
                             edges_to_include.add(edge_tuple)
                    else:
                        # Normalize edge representation for undirected graph (smaller node first)
                        edge_tuple = tuple(sorted((src, dst)))
                        if edge_tuple not in edges_to_include:
                            edge_list_parts.append(f"({edge_tuple[0]}, {edge_tuple[1]})")
                            edges_to_include.add(edge_tuple)

            # Sort for consistent output
            edge_list_parts.sort()
            prefix = "Graph (Edge List - Directed):" if self.directed else "Graph (Edge List - Undirected):"
            if not edge_list_parts:
                 return f"{prefix}\nNo edges within the selected nodes."
            return prefix + "\n" + "\n".join(edge_list_parts)

        # --- JSON Style ---
        elif style == 'json':
            output_nodes = [{"id": node} for node in sorted(list(nodes_to_describe_set))]
            output_edges = []
            processed_edges = set() # Avoid duplicates in output

            if edge_index.numel() > 0:
                edges = edge_index.t().tolist()
                for src, dst in edges:
                    if src in nodes_to_describe_set and dst in nodes_to_describe_set:
                        if self.directed:
                             edge_repr = (src, dst)
                             if edge_repr not in processed_edges:
                                 output_edges.append({"source": src, "target": dst})
                                 processed_edges.add(edge_repr)
                        else:
                             edge_repr = tuple(sorted((src, dst)))
                             if edge_repr not in processed_edges:
                                 output_edges.append({"source": edge_repr[0], "target": edge_repr[1]})
                                 processed_edges.add(edge_repr)

            # Sort edges for consistency
            output_edges.sort(key=lambda x: (x['source'], x['target']))
            graph_data = {"nodes": output_nodes, "edges": output_edges}
            # Use compact JSON formatting to save tokens
            return f"Graph (JSON):\n" + json.dumps(graph_data, separators=(',', ':'))


        # --- Natural Language Style (Original) ---
        elif style == 'natural':
            adj = self._build_adj_list(edge_index, num_nodes) # Needs full adj list logic
            parts = []
            sorted_nodes_to_describe = sorted(list(nodes_to_describe_set))
            node_list_str = ', '.join(map(str, sorted_nodes_to_describe))
            prefix = f"Graph (Natural):\nDescribes a {'directed' if self.directed else 'undirected'} graph containing nodes {node_list_str}."
            # Note: The description below implies undirected even if self.directed=True
            # because it uses "connected to". A more precise directed description
            # would use "has outgoing edges to" or similar. Let's refine this.
            if not sorted_nodes_to_describe:
                 return f"{prefix}\nNo specific node details as the set is empty."

            parts.append(prefix)
            for node in sorted_nodes_to_describe:
                neighbors = adj.get(node, [])
                # Optional: filter neighbors based on mask? Keeping original behavior for now.
                # neighbors = [n for n in neighbors if n in nodes_to_describe_set]
                if neighbors:
                    neighbor_str = ', '.join(map(str, neighbors))
                    if self.directed:
                         # More accurate phrasing for directed graphs
                         # Check outgoing edges only (already handled by _build_adj_list if directed=True)
                         parts.append(f"- Node {node} has outgoing edges to: {neighbor_str}.")
                    else:
                         parts.append(f"- Node {node} is connected to: {neighbor_str}.")
                else:
                    # Check if node exists in adj even if it has no outgoing edges (for directed)
                    # or no connections at all (for undirected)
                    if node in adj or not self.directed: # Ensure isolated nodes are mentioned
                        parts.append(f"- Node {node} has no {'outgoing edges' if self.directed else 'connections'} shown.") # Clarify based on directed flag

            return "\n".join(parts)


        else:
            raise ValueError(f"Unknown encoding style: {style}. Choose from 'adj_list', 'edge_list', 'natural', 'json'.")