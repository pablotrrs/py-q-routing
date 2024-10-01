import argparse
from simulation import nodes, neighbors, send_packet, update_node_status, node_status, node_functions, functions_sequence, q_table
from visualization import plot_network, plot_q_tables

def run_simulation(episodes, epsilon, alpha, gamma):
    """Runs the simulation for the given number of episodes with specified hyperparameters."""
    global INITIAL_EPSILON, ALPHA, GAMMA
    INITIAL_EPSILON = epsilon
    ALPHA = alpha
    GAMMA = gamma

    for episode in range(1, episodes + 1):
        update_node_status()
        path, hops, time, processed_functions = send_packet('tx', 'rx')
        print(f"Episode {episode}: Packet took {hops} hops and {time} time units.")
        
        # Call plot_network to visualize the current episode
        plot_network(path, processed_functions, functions_sequence, episode, nodes, neighbors, node_status, node_functions)
        plot_q_tables(q_table, episode)

    print('Simulation finished.')

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Run network simulation.')
    parser.add_argument('--episodes', type=int, default=100, help='Number of episodes to run the simulation (default: 100)')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Initial epsilon for exploration (default: 1.0)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Learning rate (alpha) for Q-learning (default: 0.5)')
    parser.add_argument('--gamma', type=float, default=0.9, help='Discount factor (gamma) for Q-learning (default: 0.9)')

    args = parser.parse_args()
    run_simulation(args.episodes, args.epsilon, args.alpha, args.gamma)
