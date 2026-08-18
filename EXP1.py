from collections import deque


def bfs_shortest_path(graph, start, goal):
    queue = deque([start])
    visited = {start}
    parent = {start: None}

    while queue:
        current = queue.popleft()

        # If goal is reached, reconstruct the path
        if current == goal:
            return reconstruct_path(parent, start, goal)

        # Visit all neighboring nodes
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    # If no path is found
    return None


def reconstruct_path(parent, start, goal):
    path = []
    curr = goal

    while curr is not None:
        path.append(curr)
        curr = parent[curr]

    # Reverse the path to get start → goal
    return path[::-1]


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


# Find and print the shortest path
print("BFS Path:", bfs_shortest_path(graph_data, 'S', 'G'))