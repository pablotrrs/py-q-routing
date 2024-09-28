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
epsilon = 0.1

# Define the 6x6 grid topology with 36 intermediary nodes
nodes = ['tx'] + [f'i{n}' for n in range(0, 36)] + ['rx']

# Update neighbors for a 6x6 irregular grid topology
neighbors = {
    'tx': ['i0'],  # Conectamos el nodo Tx al nodo 'i0'
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
    'i25': ['i19', 'i26',],
    'i26': ['i20', 'i25'],
    'i27': ['i21', 'i28'],
    'i28': ['i22', 'i27'],
    'i29': ['i23', 'i35'],
    'i30': ['i24', 'i31'],
    'i31': ['i30', 'i32'],
    'i32': ['i31', 'i33'],
    'i33': ['i32', 'i34'],
    'i34': ['i33', 'i35'],
    'i35': ['i29', 'i34', 'rx'],  # Nodo 'i35' conectado al Rx
    'rx': ['i35']  # Conexión final hacia Rx
}


# Define positions for nodes in the grid
positions = {
    'tx': (0 * 2, 3 * 2),
    'rx': (7 * 2, 3 * 2)
}

for i in range(0, 36):  # Asegúrate de que 'i' va de 0 a 35
    row, col = divmod(i, 6)
    positions[f'i{i}'] = ((col + 1) * 2, row * 2)

# Initialize Q-tables, processing times, and node lifetimes
q_table = {node: {dest: np.zeros(len(nodes)) for dest in nodes} for node in nodes}
processing_time = {node: random.randint(1, 5) for node in nodes}
node_lifetime = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_reconnect_time = {node: random.randint(5, 20) for node in nodes if node not in ['tx', 'rx']}
node_status = {node: True for node in nodes if node not in ['tx', 'rx']}

def update_node_status():
    for node in node_status:
        if node_status[node]:
            node_lifetime[node] -= 1
            if node_lifetime[node] <= 0:
                node_status[node] = False
                node_reconnect_time[node] = np.random.exponential(scale=10)
        else:
            node_reconnect_time[node] -= 1
            if node_reconnect_time[node] <= 0:
                node_status[node] = True
                node_lifetime[node] = np.random.exponential(scale=20)

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
        if total_hops >= max_hops:
            print(f"El paquete se ha perdido después de {total_hops} hops.")
            return path, total_hops, total_time

        available_nodes = [n for n in neighbors[current_node] if node_status.get(n, True)]
        next_node = select_next_node(q_table[current_node][rx], available_nodes)

        if next_node is None:
            print(f"Nodo {current_node} no puede enviar el paquete, no hay nodos disponibles.")
            return path, total_hops, total_time

        update_q_value(current_node, next_node, rx, -1)

        current_node = next_node
        path.append(current_node)
        total_hops += 1
        total_time += processing_time[current_node]

    update_q_value(current_node, rx, rx, 100)

    path.append(rx)
    total_time += processing_time[rx]

    return path, total_hops, total_time

def plot_network(path, episode):
    plt.clf()

    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from([(node1, node2) for node1 in neighbors for node2 in neighbors[node1]])

    node_colors = []
    for node in nodes:
        if node == 'tx' or node == 'rx':
            node_colors.append('green')
        elif node_status.get(node, True):
            node_colors.append('green')
        else:
            node_colors.append('gray')

    nx.draw(G, pos=positions, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrows=False)

    edges_in_path = [(path[i], path[i+1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(G, pos=positions, edgelist=edges_in_path, edge_color='blue', width=3, arrows=True)

    plt.title(f'Episode {episode} - Path: {" -> ".join(path)}')

    plt.savefig(f'{output_folder}/network_episode_{episode}.png')

def plot_q_tables(episode):
    cell_size = 30  # Tamaño de cada celda en píxeles
    pixels_per_inch = 96  # 96 píxeles por pulgada

    # Recorremos cada nodo y creamos una imagen por cada Q-table
    for i in range(0, 36):
        node = f'i{i}'

        # Obtener los valores de la Q-table del nodo actual
        q_values = np.array(list(q_table[node].values()))

        # Calcular el tamaño de la figura en pulgadas (basado en el número de celdas en la Q-table)
        N = q_values.shape[0]  # Número de filas/columnas de la Q-table
        figure_size_in_pixels = N * cell_size  # Total de píxeles para el ancho/alto
        figure_size_in_inches = figure_size_in_pixels / pixels_per_inch  # Convertir a pulgadas

        # Crear la figura con el tamaño calculado
        fig, ax = plt.subplots(figsize=(figure_size_in_inches, figure_size_in_inches))

        # Crear el gráfico de la Q-table del nodo
        cax = ax.matshow(q_values, cmap="RdYlGn", vmin=-100, vmax=100)
        fig.colorbar(cax, ax=ax)

        # Añadir los valores en cada celda
        for (row, col), val in np.ndenumerate(q_values):
            ax.text(col, row, f'{val:.1f}', ha='center', va='center', fontsize=8, color='black')

        # Establecer título y ajustar las etiquetas de los ejes
        ax.set_title(f"Q-table: {node} - Episode {episode}", fontsize=10)

        # Ajuste de las etiquetas de los ejes x e y para más legibilidad
        ax.set_xticks(range(N))
        ax.set_xticklabels([f'{n}' for n in range(N)], rotation=90, fontsize=8)
        ax.set_yticks(range(N))
        ax.set_yticklabels([f'{n}' for n in range(N)], fontsize=8)

        # Aumentar la separación de las etiquetas
        ax.tick_params(axis='x', which='major', pad=10)
        ax.tick_params(axis='y', which='major', pad=10)

        # Mantener las celdas cuadradas
        ax.set_aspect('equal')

        # Ajustar el espaciado para evitar solapamientos
        plt.tight_layout()

        # Guardar la figura individual para el nodo actual
        plt.savefig(f'{output_folder}/q_table_{node}_episode_{episode}.png')
        plt.close(fig)

def animate_network(path):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from([(node1, node2) for node1 in neighbors for node2 in neighbors[node1]])

    node_colors = ['green' if node in ['tx', 'rx'] or node_status.get(node, True) else 'gray' for node in nodes]
    edge_colors = ['gray' for _ in G.edges]

    fig, ax = plt.subplots()
    nx.draw(G, pos=positions, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrows=False, ax=ax)

    def update(frame):
        nonlocal edge_colors
        if frame < len(path) - 1:
            current_edge = (path[frame], path[frame + 1])
            edge_colors = ['blue' if edge == current_edge else color for edge, color in zip(G.edges, edge_colors)]
            ax.clear()
            nx.draw(G, pos=positions, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', ax=ax)
            nx.draw_networkx_edges(G, pos=positions, edge_color=edge_colors, width=3, arrows=True, ax=ax)

    ani = FuncAnimation(fig, update, frames=len(path), interval=1000, repeat=False)
    plt.show()

def run_simulation(episodes):
    for episode in range(1, episodes + 1):
        update_node_status()
        path, hops, time = send_packet('tx', 'rx')
        print(f"Episodio {episode}: El paquete realizó {hops} saltos y tardó {time} unidades de tiempo.")
        if episode == 1 or episode == 5000:
            plot_q_tables(episode)
            plot_network(path, episode)
            #animate_network(path)

    print('Simulación finalizada.')

# Run the simulation
run_simulation(5000)
