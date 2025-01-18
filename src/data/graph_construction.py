def build_correlation_graph(adj_close_data, threshold=0.6):
    # Compute correlation matrix
    correlation_matrix = adj_close_data.pct_change().dropna().corr()

    # Create NetworkX graph
    graph = nx.Graph()

    # Add nodes (stocks)
    for stock in adj_close_data.columns:
        graph.add_node(stock)

    # Add edges based on correlation threshold
    for i, stock1 in enumerate(correlation_matrix.columns):
        for j, stock2 in enumerate(correlation_matrix.columns):
            if i < j and correlation_matrix.iloc[i, j] > threshold:
                graph.add_edge(stock1, stock2, weight=correlation_matrix.iloc[i, j])

    return graph, correlation_matrix

def visualize_graph(graph, correlation_matrix):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)  # Positioning nodes using spring layout

    # Draw nodes
    nx.draw_networkx_nodes(graph, pos, node_size=500, node_color='skyblue')

    # Draw edges with transparency based on correlation weight
    edges, weights = zip(*nx.get_edge_attributes(graph, 'weight').items())
    nx.draw_networkx_edges(graph, pos, edgelist=edges, alpha=0.6, edge_color='gray', width=2)

    # Draw labels
    nx.draw_networkx_labels(graph, pos, font_size=10, font_color='black')

    plt.title("Stock Correlation Graph (Threshold > 0.6)", fontsize=14)
    plt.show()
