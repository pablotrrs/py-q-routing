import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx
import os

# Crear una carpeta para guardar las imágenes si no existe
output_folder = 'simulation_images'
os.makedirs(output_folder, exist_ok=True)

# Parámetros del algoritmo
alpha = 0.5
gamma = 0.9
epsilon = 0.1

# Definir los 10 nodos y sus posiciones
nodes = ['sender', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'receiver']

# Definimos los vecinos de cada nodo
neighbors = {
    'sender': ['i1', 'i2'],
    'i1': ['i2', 'i3', 'i4'],
    'i2': ['i1', 'i3', 'i4'],
    'i3': ['i1', 'i2', 'i4', 'i5', 'i6'],
    'i4': ['i1', 'i2', 'i3', 'i5', 'i6'],
    'i5': ['i3', 'i4', 'i6', 'i7', 'i8'],
    'i6': ['i3', 'i4', 'i5', 'i7', 'i8'],
    'i7': ['i5', 'i6', 'i8', 'i9', 'i10'],
    'i8': ['i5', 'i6', 'i7', 'i9', 'i10'],
    'i9': ['i7', 'i8', 'i10', 'receiver'],
    'i10': ['i7', 'i8', 'i9', 'receiver'],
    'receiver': ['i9', 'i10']
}


# Posiciones para los nodos
positions = {
    'sender': (0, 0),
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
    'receiver': (12, 0)
}

# Actualizamos el tiempo de procesamiento para cada nodo
processing_time = {node: random.randint(1, 5) for node in nodes}

# Inicializamos la Q-table para cada nodo
q_table = {
    node: {dest: np.zeros(len(nodes)) for dest in nodes} for node in nodes
}

# Definir los tiempos de vida (desconexión y reconexión) para los nodos
node_lifetime = {node: random.randint(5, 20) for node in nodes if node not in ['sender', 'receiver']}
node_reconnect_time = {node: random.randint(5, 20) for node in nodes if node not in ['sender', 'receiver']}
node_status = {node: True for node in nodes if node not in ['sender', 'receiver']}  # Inicialmente todos están conectados

# Función para simular desconexión y reconexión de nodos
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

# Función para seleccionar la siguiente acción (nodo) utilizando epsilon-greedy
def select_next_node(q_values, available_nodes):
    available_nodes = [n for n in available_nodes if node_status.get(n, True)]  # Solo considerar nodos vivos
    if not available_nodes:  # Si no hay nodos vivos disponibles
        return None  # No se puede hacer nada

    if random.uniform(0, 1) < epsilon:
        return random.choice(available_nodes)
    else:
        max_q_value = max(q_values[nodes.index(n)] for n in available_nodes)
        best_nodes = [n for n in available_nodes if q_values[nodes.index(n)] == max_q_value]
        return random.choice(best_nodes)

# Función para actualizar la Q-table
def update_q_value(current_node, next_node, destination, reward):
    if next_node is None:
        return
    current_q = q_table[current_node][destination][nodes.index(next_node)]
    max_next_q = max(q_table[next_node][destination])
    new_q = (1 - alpha) * current_q + alpha * (reward + gamma * max_next_q)
    q_table[current_node][destination][nodes.index(next_node)] = new_q

# Función de simulación de envío de un paquete desde sender a receiver
def send_packet(sender, receiver):
    current_node = sender
    total_hops = 0
    total_time = 0
    max_hops = 100  # Límite de hops por paquete

    path = [current_node]
    
    while current_node != receiver:
        if total_hops >= max_hops:  # Si supera el límite de hops
            print(f"El paquete se ha perdido después de {total_hops} hops.")
            # Penalizamos este episodio similar a como se penaliza por el tiempo de procesamiento
            return path, total_hops, total_time + max_hops * 5  # Penalización agregada por hops
                    
        available_nodes = [n for n in neighbors[current_node] if node_status.get(n, True)]
        next_node = select_next_node(q_table[current_node][receiver], available_nodes)

        if next_node is None:
            print(f"Nodo {current_node} no puede enviar el paquete, no hay nodos disponibles.")
            return path, total_hops, total_time  # No se puede continuar, retornar lo logrado

        reward = -processing_time[next_node]
        update_q_value(current_node, next_node, receiver, reward)
        
        current_node = next_node
        path.append(current_node)
        total_hops += 1
        total_time += processing_time[current_node]

    final_reward = -processing_time[receiver]
    update_q_value(current_node, receiver, receiver, final_reward)
    path.append(receiver)
    total_time += processing_time[receiver]

    return path, total_hops, total_time


# Función para visualizar los nodos y la tabla Q
def plot_network(path, episode):
    plt.clf()
    
    G = nx.DiGraph()  # Usar un grafo dirigido
    G.add_nodes_from(nodes)
    
    # Crear conexiones basadas en la Q-table
    for node in nodes:
        for neighbor in nodes:
            if node != neighbor and q_table[node][neighbor].max() > 0:  # Solo conectar si hay un valor positivo en la Q-table
                G.add_edge(node, neighbor)

    # Dibujar la red de nodos con colores según su estado
    node_colors = []
    for node in nodes:
        if node == 'sender' or node == 'receiver':
            node_colors.append('green')  # El sender y receiver siempre están online
        elif node_status.get(node, True):
            node_colors.append('green')  # Nodos activos
        else:
            node_colors.append('gray')  # Nodos inactivos
    
    # Dibujar el grafo con flechas
    nx.draw(G, pos=positions, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrows=True)
    
    # Dibujar el camino del paquete con flechas
    edges_in_path = [(path[i], path[i+1]) for i in range(len(path) - 1)]
    nx.draw_networkx_edges(G, pos=positions, edgelist=edges_in_path, edge_color='r', width=3, arrows=True)
    
    plt.title(f'Episodio {episode} - Camino: {" -> ".join(path)}')
    
    # Guardar el gráfico como una imagen
    plt.savefig(f'{output_folder}/network_episode_{episode}.png')

# Función para visualizar las Q-tables por separado
def plot_q_tables(episode):
    fig, axes = plt.subplots(2, 5, figsize=(20, 10))  # Ajustamos el tamaño de la figura y el layout para 10 nodos
    fig.suptitle(f'Q-tables por nodo - Episodio {episode}')
    
    node_list = list(q_table.keys())
    for i, ax in enumerate(axes.flat):
        if i < len(node_list):
            node = node_list[i]
            
            # Convertimos la Q-table del nodo en una matriz para visualizarla
            q_values = np.array(list(q_table[node].values()))
            
            ax.matshow(q_values, cmap="Blues")
            ax.set_title(f"Q-table: {node}", fontsize=10)
            ax.set_xticks(range(len(nodes)))
            ax.set_xticklabels(nodes, rotation=90, fontsize=8)  # Reducimos el tamaño de las etiquetas
            ax.set_yticks(range(len(nodes)))
            ax.set_yticklabels(nodes, fontsize=8)  # Reducimos el tamaño de las etiquetas

            # Añadimos los valores de la Q-table sobre el gráfico con espaciado
            for j in range(len(nodes)):
                for k in range(len(nodes)):
                    ax.text(k, j, f'{q_values[j, k]:.2f}', ha='center', va='center', fontsize=6)  # Ajustamos el tamaño de fuente

    # Ajustamos el espacio entre subgráficas
    plt.subplots_adjust(hspace=0.5, wspace=0.5)  # Añadimos más espacio entre subplots
    
    # Guardar la imagen de las Q-tables
    plt.savefig(f'{output_folder}/q_tables_episode_{episode}.png')
    plt.close(fig)

# Función para generar un color aleatorio
def random_color():
    return "#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])

# Función para dibujar la red y el camino paso a paso con flechas de colores aleatorios
def animate_network(path, episode):
    plt.clf()
    
    G = nx.DiGraph()  # Crear un grafo dirigido
    G.add_nodes_from(nodes)  # Agregar los nodos al grafo
    
    # Agregar las conexiones (aristas) basadas en la Q-table
    for node in nodes:
        for neighbor in nodes:
            if node != neighbor and q_table[node][neighbor].max() > 0:  # Conectar si hay valor positivo en la Q-table
                G.add_edge(node, neighbor)

    # Crear lista de colores para los nodos dependiendo de si están online o offline
    node_colors = []
    for node in nodes:
        if node == 'sender' or node == 'receiver':
            node_colors.append('green')  # Sender y receiver siempre están online
        elif node_status.get(node, True):  
            node_colors.append('green')  # Nodos activos (online)
        else:
            node_colors.append('gray')  # Nodos inactivos (offline)

    # Dibujar los nodos y las conexiones (aristas)
    nx.draw(G, pos=positions, with_labels=True, node_color=node_colors, node_size=500, font_size=8, font_weight='bold', arrows=True)

    # Mostrar los vecinos de cada nodo
    for node in nodes:
        for neighbor in G.neighbors(node):
            plt.text((positions[node][0] + positions[neighbor][0]) / 2,
                     (positions[node][1] + positions[neighbor][1]) / 2,
                     'V', fontsize=8, color='blue', ha='center')

    # Dibujar el camino del paquete paso a paso, mostrando el recorrido con flechas de colores aleatorios
    for i in range(len(path) - 1):
        edges_in_path = [(path[i], path[i+1])]
        edge_color = random_color()  # Generar un color aleatorio para cada paso
        nx.draw_networkx_edges(G, pos=positions, edgelist=edges_in_path, edge_color=edge_color, width=3, arrows=True)
        plt.title(f'Episodio {episode} - Camino: {" -> ".join(path[:i + 2])}')
        plt.pause(0.00001)  # Pausa para mostrar cada paso del recorrido


# Función principal para ejecutar múltiples episodios y guardar los gráficos
def run_simulation(episodes):
    plt.figure(figsize=(12, 8))
    print(f'Iniciando simulación de {episodes} episodios...')
    for episode in range(1, episodes + 1):
        update_node_status()  # Actualizar el estado de los nodos en cada episodio
        path, hops, time = send_packet('sender', 'receiver')
                #print(f"Episodio {episode}: El paquete tomó el camino {path} con {hops} saltos y tardó {time} unidades de tiempo.")
        print(f"Episodio {episode}: El paquete realizó {hops} saltos y tardó {time} unidades de tiempo.")
        print(f"-------------------------------------------------")
        plot_q_tables(episode)
        animate_network(path, episode)
    
    plt.show()
    print('Simulación finalizada.')

# Ejecutamos la simulación
run_simulation(1000)
