
import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import os
from matplotlib.animation import FuncAnimation

output_folder = 'simulation_images'
os.makedirs(output_folder, exist_ok=True)

# Hyperparameters
alpha = 0.5
gamma = 0.9
initial_epsilon = 1.0
min_epsilon = 0.1
decay_rate = 0.99
epsilon = initial_epsilon

# Define the 6x6 grid topology with 36 intermediary nodes
nodes = ['tx'] + [f'i{n}' for n in range(0, 36)] + ['rx']

# Update neighbors for a 6x6 irregular grid topology
neighbors = {
    'tx': ['i0'],
    'i0': ['i1', 'i6'],
    'i1': ['i0', 'i2', 'i7'],
    'i2': ['i1', 'i8'],
    'i3': ['i4', 'i9'],
    'i4': ['i3', 'i5', 'i10'],
    'i5': ['i4', 'i11'],
    'i6': ['i0', 'i7', 'i12'],
    'i7': ['i1', 'i6', 'i8', 'i13'],
    'i8': ['i2', 'i7', 'i14'],
    'i9': ['i3', 'i10', 'i15'],
    'i10': ['i4', 'i9', 'i11', 'i16'],
    'i11': ['i5', 'i10', 'i17'],
    'i12': ['i6', 'i13', 'i18'],
    'i13': ['i7', 'i12', 'i14', 'i19'],
    'i14': ['i8', 'i13', 'i20'],
    'i15': ['i9', 'i16', 'i21'],
    'i16': ['i10', 'i15', 'i17', 'i22'],
    'i17': ['i11', 'i16', 'i23'],
    'i18': ['i12', 'i19', 'i24'],
    'i19': ['i13', 'i18', 'i20', 'i25'],
    'i20': ['i14', 'i19', 'i21', 'i26'],
    'i21': ['i15', 'i20', 'i22', 'i27'],
    'i22': ['i16', 'i21', 'i23', 'i28'],
    'i23': ['i17', 'i22', 'i29'],
    'i24': ['i18', 'i30'],
    'i25': ['i19', 'i26'],
    'i26': ['i20', 'i25'],
    'i27': ['i21', 'i28'],
    'i28': ['i22', 'i27'],
    'i29': ['i23', 'i35'],
    'i30': ['i24', 'i31'],
    'i31': ['i30', 'i32'],
    'i32': ['i31', 'i33'],
    'i33': ['i32', 'i34'],
    'i34': ['i33', 'i35'],
    'i35': ['i29', 'i34', 'rx'],
    'rx': ['i35']
}

# Define positions for nodes in the grid
positions = {
    'tx': (0 * 2, 3 * 2),
    'rx': (7 * 2, 3 * 2)
}

for i in range(0, 36):
    row, col = divmod(i, 6)
    positions[f'i{i}'] = ((col + 1) * 2, row * 2)

# Initialize Q-tables, processing times, and node lifetimes
q_table = {node: {dest: np.random.rand(len(nodes)) for dest in nodes} for node in nodes}
processing_time = {node: random.randint(1, 5) for node in nodes}
node_lifetime = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_reconnect_time = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_status = {node: True for node in nodes if node not in ['tx', 'rx']}

functions = ["A", "B", "C"]
functions_sequence = ["A", "B", "C"]
nodes_intermediate = [node for node in nodes if node not in ['tx', 'rx']]
node_functions = {}

def assign_functions_to_nodes():
    function_counts = {func: 0 for func in functions}
    num_nodes = len(nodes_intermediate)

    for node in nodes_intermediate:
        min_assigned_func = min(function_counts, key=function_counts.get)
        node_functions[node] = min_assigned_func
        function_counts[min_assigned_func] += 1

assign_functions_to_nodes()

# Connect and disconnect nodes
def update_node_status():
    for node in node_status:
        if node_status[node]:
            node_lifetime[node] -= 1
            if node_lifetime[node] <= 0:
                node_status[node] = False
                node_reconnect_time[node] = np.random.exponential(scale=10)
                del node_functions[node]
        else:
            node_reconnect_time[node] -= 1
            if node_reconnect_time[node] <= 0:
                node_status[node] = True
                node_lifetime[node] = np.random.exponential(scale=20)
                function_counts = {func: list(node_functions.values()).count(func) for func in functions}
                min_assigned_func = min(function_counts, key=function_counts.get)
                node_functions[node] = min_assigned_func

def select_next_node(q_values, available_nodes):
    available_nodes = [n for n in available_nodes if node_status.get(n, True)]
    if not available_nodes:
        return None

    if random.uniform(0, 1) < epsilon:
        return random.choice(available_nodes)
    else:
        max_q_value = max(q_values[nodes.index(n)] for n in available_nodes)
        best_nodes = [n for n in available_nodes if q_values[nodes.index(n)] == max_q_value]
        return random.choice(best_nodes)

def update_q_value(current_node, next_node, destination, reward):
    if next_node is None:
        return
    current_q = q_table[current_node][destination][nodes.index(next_node)]
    max_next_q = max(q_table[next_node][destination])
    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_next_q)
    q_table[current_node][destination][nodes.index(next_node)] = new_q

def send_packet(tx, rx):
    global epsilon
    current_node = tx
    total_hops = 0
    total_time = 0
    max_hops = 100

    path = [current_node]
    functions_to_process = functions_sequence.copy()
    processed_functions = []

    while not (current_node == rx and len(functions_to_process) == 0):

        if total_hops >= max_hops:
            print(f"El paquete se ha perdido después de {total_hops} hops.")
            return path, total_hops, total_time, processed_functions

        available_nodes = [n for n in neighbors[current_node] if node_status.get(n, True)]
        next_node = select_next_node(q_table[current_node][rx], available_nodes)

        if next_node is None:
            print(f"Nodo {current_node} no puede enviar el paquete, no hay nodos disponibles.")
            return path, total_hops, total_time, processed_functions

        reward = 0

        if functions_to_process:
            expected_function = functions_to_process[0]
            node_function = node_functions.get(next_node, None)

            print(f'current_node: {current_node}, functions_to_process: {functions_to_process}, node_function: {node_function}')
            if node_function == expected_function:
                functions_to_process.pop(0)
                processed_functions.append(node_function)
                reward += 10
            else:
                reward -= 1

        update_q_value(current_node, next_node, rx, reward)

        current_node = next_node
        path.append(current_node)
        total_hops += 1
        total_time += processing_time[current_node]

    if len(functions_to_process) == 0:
        print('Paquete entregado y todas las funciones procesadas.')
    else:
        print('Error: Se llegó al nodo receptor, pero aún quedan funciones por procesar.')

    epsilon = max(min_epsilon, epsilon * decay_rate)

    return path, total_hops, total_time, processed_functions

def plot_network(path, processed_functions, functions_to_process, episode):
    plt.clf()

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from([(node1, node2) for node1 in neighbors for node2 in neighbors[node1]])

    node_colors = []
    node_labels = {}
    applied_functions = []
    missing_functions = functions_sequence.copy()

    for node in nodes:
        if node == 'tx' or node == 'rx':
            node_colors.append('green')
        elif node_status.get(node, True):
            node_colors.append('green')
        else:
            node_colors.append('gray')

        function = node_functions.get(node, "")
        node_labels[node] = f'{node}\nFunc: {function}'

    nx.draw(G, pos=positions, labels=node_labels, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrows=False)

    for i in range(len(path)-1):
        if (path[i+1] != 'rx' and path[i+1] != 'tx'):
            applied_function = node_functions[path[i+1]]

            if applied_function not in applied_functions:
                applied_functions.append(applied_function)

            if applied_function in missing_functions:
                missing_functions.remove(applied_function)

        applied_text = "Applied functions: " + ", ".join(applied_functions) if applied_functions else "No function applied"
        missing_text = "Missing functions: " + ", ".join(missing_functions) if missing_functions else "All functions applied"

        handles = [plt.Line2D([0], [0], color='white', label=applied_text), 
                plt.Line2D([0], [0], color='white', label=missing_text)]

        plt.legend(handles=handles, loc='upper right', fontsize=8)

        edges_in_path = [(path[i], path[i + 1])]
        edge_color = 'blue'
        nx.draw_networkx_edges(G, pos=positions, edgelist=edges_in_path, edge_color=edge_color, width=3, arrows=True)
        plt.title(f'Episodio {episode} - Camino: {" -> ".join(path[:i + 2])}')
        plt.pause(0.00001)

def run_simulation(episodes):
    for episode in range(1, episodes + 1):
        update_node_status()
        path, hops, time, processed_functions = send_packet('tx', 'rx')
        print(f"Episodio {episode}: El paquete realizó {hops} saltos y tardó {time} unidades de tiempo.")
        print(f"-------------------------------------------------")
        plot_network(path, processed_functions, functions_sequence, episode)

    print('Simulación finalizada.')

# Ejecutamos la simulación
run_simulation(100)
