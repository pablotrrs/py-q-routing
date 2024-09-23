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
epsilon = 0.1

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

# Initialize each node parameters
q_table = {
    node: {dest: np.zeros(len(nodes)) for dest in nodes} for node in nodes
}
processing_time = {node: random.randint(1, 5) for node in nodes}
node_lifetime = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_reconnect_time = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_status = {node: True for node in nodes if node not in ['tx', 'rx']}  # Initially all nodes are connected

# Connect and disconnect nodes
def update_node_status():
    for node in node_status:
        if node_status[node]:  # Nodo está conectado
            node_lifetime[node] -= 1
            if node_lifetime[node] <= 0:
                node_status[node] = False  # Desconectar nodo
                node_reconnect_time[node] = random.randint(5, 20)  # Establecer tiempo de reconexión
        else:  # Nodo está desconectado
            node_reconnect_time[node] -= 1
            if node_reconnect_time[node] <= 0:
                node_status[node] = True  # Reconectar nodo
                node_lifetime[node] = random.randint(5, 20)  # Establecer nueva vida útil

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
    current_node = tx
    total_hops = 0
    total_time = 0
    max_hops = 100

    path = [current_node]

    while current_node != rx:
        if total_hops >= max_hops:  # Assume lost packet after number of hops
            print(f"El paquete se ha perdido después de {total_hops} hops.")
            # TODO: sacar esto de la penalización extra
            return path, total_hops, total_time + max_hops * 5

        available_nodes = [n for n in neighbors[current_node] if node_status.get(n, True)]
        next_node = select_next_node(q_table[current_node][rx], available_nodes)

        if next_node is None:
            print(f"Nodo {current_node} no puede enviar el paquete, no hay nodos disponibles.")
            return path, total_hops, total_time

        reward = -processing_time[next_node]
        update_q_value(current_node, next_node, rx, reward)

        current_node = next_node
        path.append(current_node)
        total_hops += 1
        total_time += processing_time[current_node]

    final_reward = -processing_time[rx]
    update_q_value(current_node, rx, rx, final_reward)
    path.append(rx)
    total_time += processing_time[rx]

    return path, total_hops, total_time

def plot_network(path, episode):
    plt.clf()

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    for node in nodes:
        for neighbor in nodes:
            if node != neighbor and q_table[node][neighbor].max() > 0:
                G.add_edge(node, neighbor)

    node_colors = []
    for node in nodes:
        if node == 'tx' or node == 'rx':
            node_colors.append('green')
        elif node_status.get(node, True):
            node_colors.append('green')
        else:
            node_colors.append('gray')

    nx.draw(G, pos=positions, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrows=True)

    edges_in_path = [(path[i], path[i+1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(G, pos=positions, edgelist=edges_in_path, edge_color='r', width=3, arrows=True)

    plt.title(f'Episode {episode} - Path: {" -> ".join(path)}')

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

def animate_network(path, episode):
    plt.clf()

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    for node in nodes:
        for neighbor in nodes:
            if node != neighbor and q_table[node][neighbor].max() > 0:
                G.add_edge(node, neighbor)

    node_colors = []
    for node in nodes:
        if node == 'tx' or node == 'rx':
            node_colors.append('green')
        elif node_status.get(node, True):
            node_colors.append('green')
        else:
            node_colors.append('gray')

    nx.draw(G, pos=positions, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrows=True)

    for node in nodes:
        for neighbor in G.neighbors(node):
            plt.text((positions[node][0] + positions[neighbor][0]) / 2,
                     (positions[node][1] + positions[neighbor][1]) / 2,
                     'V', fontsize=8, color='blue', ha='center')

    for i in range(len(path) - 1):
        edges_in_path = [(path[i], path[i+1])]
        edge_color = random_color()
        nx.draw_networkx_edges(G, pos=positions, edgelist=edges_in_path, edge_color=edge_color, width=3, arrows=True)
        plt.title(f'Episodio {episode} - Camino: {" -> ".join(path[:i + 2])}')
        plt.pause(0.00001)

def run_simulation(episodes):
    plt.figure(figsize=(12, 8))
    print(f'Iniciando simulación de {episodes} episodios...')
    for episode in range(1, episodes + 1):
        update_node_status()  # Actualizar el estado de los nodos en cada episodio
        path, hops, time = send_packet('tx', 'rx')
        print(f"Episodio {episode}: El paquete realizó {hops} saltos y tardó {time} unidades de tiempo.")
        print(f"-------------------------------------------------")
        plot_q_tables(episode)
        animate_network(path, episode)

    plt.show()
    print('Simulación finalizada.')

# Ejecutamos la simulación
run_simulation(1000)
