def dfs_path(graph, start, goal, visited=None):

    # Create an empty set for visited nodes
    if visited is None:
        visited = set()

    # If we reach the goal, return the goal node
    if start == goal:
        return [start]

    # Mark the current node as visited
    visited.add(start)

    # Explore all neighboring nodes
    for neighbor in graph.get(start, []):

        if neighbor not in visited:

            result_path = dfs_path(graph, neighbor, goal, visited)

            # If a path is found, return it
            if result_path:
                return [start] + result_path

    # If no path is found
    return None


# Define the graph
graph_data = {
    'S': ['A', 'B'],
    'A': ['C', 'D'],
    'B': ['G', 'H'],
    'C': [],
    'D': [],
    'G': [],
    'H': []
}


# Find and print the DFS path
print("DFS Path:", dfs_path(graph_data, 'S', 'G'))