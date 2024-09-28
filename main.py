import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import os

output_folder = 'simulation_images'
os.makedirs(output_folder, exist_ok=True)

# Hyperparameters
alpha = 0.5
gamma = 0.9
initial_epsilon = 1.0
min_epsilon = 0.1
decay_rate = 0.99
epsilon = initial_epsilon

nodes = ['tx', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'rx']

neighbors = {
    'tx': ['i1', 'i2'],
    'i1': ['i2', 'i3', 'i4'],
    'i2': ['i1', 'i3', 'i4'],
    'i3': ['i1', 'i2', 'i4', 'i5', 'i6'],
    'i4': ['i1', 'i2', 'i3', 'i5', 'i6'],
    'i5': ['i3', 'i4', 'i6', 'i7', 'i8'],
    'i6': ['i3', 'i4', 'i5', 'i7', 'i8'],
    'i7': ['i5', 'i6', 'i8', 'i9', 'i10'],
    'i8': ['i5', 'i6', 'i7', 'i9', 'i10'],
    'i9': ['i7', 'i8', 'i10', 'rx'],
    'i10': ['i7', 'i8', 'i9', 'rx'],
    'rx': ['i9', 'i10']
}

positions = {
    'tx': (0, 0),
    'i1': (2, 1), 
    'i2': (2, -1), 
    'i3': (4, 1), 
    'i4': (4, -1),
    'i5': (6, 1),
    'i6': (6, -1),
    'i7': (8, 1),
    'i8': (8, -1),
    'i9': (10, 1),
    'i10': (10, -1),
    'rx': (12, 0)
}

functions = ["A", "B", "C"]
functions_sequence = ["A", "B", "C"]
q_table = {
    node: {dest: np.random.rand(len(nodes)) for dest in nodes} for node in nodes
}
processing_time = {node: random.randint(1, 5) for node in nodes}
node_lifetime = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_reconnect_time = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_status = {node: True for node in nodes if node not in ['tx', 'rx']}  # Initially all nodes are connected

nodes_intermediate = [node for node in nodes if node not in ['tx', 'rx']]
node_functions = {}

def assign_functions_to_nodes():
    function_counts = {func: 0 for func in functions}
    num_nodes = len(nodes_intermediate)
    num_functions = len(functions)

    for node in nodes_intermediate:
        min_assigned_func = min(function_counts, key=function_counts.get)
        node_functions[node] = min_assigned_func
        function_counts[min_assigned_func] += 1

assign_functions_to_nodes()

def update_node_status():
    for node in node_status:
        if node_status[node]:
            node_lifetime[node] -= 1
            if node_lifetime[node] <= 0:
                node_status[node] = False
                node_reconnect_time[node] = np.random.exponential(scale=10)
                # Remove assigned function
                del node_functions[node]
        else:
            node_reconnect_time[node] -= 1
            if node_reconnect_time[node] <= 0:
                node_status[node] = True
                node_lifetime[node] = np.random.exponential(scale=20)
                # Assign a new function to a reconnected node
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
                print(f'before processing function {node_function} from functions to process: {functions_to_process}')
                functions_to_process.pop(0)
                print(f'removing the function {node_function} from functions to process: {functions_to_process}')
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

    path.append(rx)
    total_time += processing_time[rx]

    epsilon = max(min_epsilon, epsilon * decay_rate)

    return path, total_hops, total_time, processed_functions

def plot_network(path, processed_functions, functions_to_process, episode):
    plt.clf()

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    G.clear_edges()

    node_colors = []
    node_labels = {}

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

    edges_in_path = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(G, pos=positions, edgelist=edges_in_path, edge_color='r', width=3, arrows=True)

    plt.title(f'Episode {episode} - Path: {" -> ".join(path)}')
    plt.legend([applied_text, missing_text], loc='upper right', fontsize=8)

    plt.savefig(f'{output_folder}/network_episode_{episode}.png')

def plot_q_tables(episode):
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    fig.suptitle(f'Q-tables per node - Episode {episode}')

    node_list = list(q_table.keys())
    for i, ax in enumerate(axes.flat):
        if i < len(node_list):
            node = node_list[i]

            q_values = np.array(list(q_table[node].values()))

            ax.matshow(q_values, cmap="Blues")
            ax.set_title(f"Q-table: {node}", fontsize=10)
            ax.set_xticks(range(len(nodes)))
            ax.set_xticklabels(nodes, rotation=90, fontsize=8)
            ax.set_yticks(range(len(nodes)))
            ax.set_yticklabels(nodes, fontsize=8)

            for j in range(len(nodes)):
                for k in range(len(nodes)):
                    ax.text(k, j, f'{q_values[j, k]:.2f}', ha='center', va='center', fontsize=6)

    plt.subplots_adjust(hspace=0.5, wspace=0.5)

    plt.savefig(f'{output_folder}/q_tables_episode_{episode}.png')
    plt.close(fig)

def random_color():
    return "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

def animate_network(path, processed_functions, functions_to_process, episode):
    plt.clf()

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    G.clear_edges()

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

        # Clear the legend first to avoid duplicates
        plt.legend().remove()

        # Create a dummy handle for the legend without markers
        handles = [plt.Line2D([0], [0], color='none', label=applied_text), 
                   plt.Line2D([0], [0], color='none', label=missing_text)]

        # Display the legend without circles or squares
        plt.legend(handles=handles, loc='upper right', fontsize=8)

        edges_in_path = [(path[i], path[i + 1])]
        edge_color = random_color()
        nx.draw_networkx_edges(G, pos=positions, edgelist=edges_in_path, edge_color=edge_color, width=3, arrows=True)
        plt.title(f'Episodio {episode} - Camino: {" -> ".join(path[:i + 2])}')
        plt.pause(0.00001)

def run_simulation(episodes):
    plt.figure(figsize=(12, 8))
    print(f'Iniciando simulación de {episodes} episodios...')
    for episode in range(1, episodes + 1):
        update_node_status()  # Actualizar el estado de los nodos en cada episodio
        path, hops, time, processed_functions = send_packet('tx', 'rx')
        print(f"Episodio {episode}: El paquete realizó {hops} saltos y tardó {time} unidades de tiempo.")
        print(f"-------------------------------------------------")
        plot_q_tables(episode)
        animate_network(path, episode, processed_functions, episode)

    plt.show()
    print('Simulación finalizada.')

# Ejecutamos la simulación
run_simulation(100)
