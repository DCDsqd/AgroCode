import networkx as nx
import matplotlib.pyplot as plt


def draw_advanced_graph_from_adjacency_list(adjacency_list):
    """
    Draw a graph based on a given adjacency list with curved edges and color coding to indicate the most active node.
    Curved lines are now used to connect nodes.

    Parameters:
    - adjacency_list (list of lists): Each inner list contains the indices of nodes connected to the node of that index.
    """
    # Create a new graph
    G = nx.Graph()

    # Add edges to the graph based on the adjacency list
    for index, neighbors in enumerate(adjacency_list):
        for neighbor in neighbors:
            G.add_edge(index, neighbor)

    # Find the most active node based on the connections
    most_active_node = max(range(len(adjacency_list)), key=lambda i: len(adjacency_list[i]))
    most_active_color = "red"
    default_node_color = "skyblue"

    # Define positions using a spring layout for more natural look
    pos = nx.spring_layout(G)

    # Draw the nodes with different colors for the most active node
    node_colors = [most_active_color if node == most_active_node else default_node_color for node in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors)

    # Draw the edges with curves
    for start_node, end_node in G.edges():
        arc_rad = 0.1 if start_node != end_node else 0.2  # Increase arc radius for self-loops
        nx.draw_networkx_edges(
            G, pos, edgelist=[(start_node, end_node)], connectionstyle=f'arc3,rad={arc_rad}'
        )

    # Draw the labels
    nx.draw_networkx_labels(G, pos)

    # Set plot options and display
    plt.title("Advanced Custom Graph")
    plt.axis('off')
    plt.show()