## System Overview

The simulation emulates the behavior of deep learning layers by distributing functions (e.g., A, B, C) across the nodes of a network. Packets must traverse the network, encountering the required functions in the correct sequence to process the data as if it were passing through layers of a deep learning model.

![Network Animation](./assets/network-animation.gif)

This Python-based simulation models the behavior of the network routing system for ESP8266 devices, as implemented in the project hosted at [this repository](https://github.com/FrancoBre/esp-q-mesh-routing). The simulation helps visualize how packets navigate through a distributed mesh network while seeking required processing functions. The aim is to optimize routing and emulate the real-world performance of the hardware-based system.

### Key Features

1. **Dynamic Function Assignment**:
   - Functions (e.g., A, B, C) are assigned to nodes dynamically, and their availability changes over time. Nodes randomly take on different functions, and packets must find a route that allows them to process the required functions in sequence.

2. **Fixed Nodes, Dynamic Functions**:
   - The nodes in the network are fixed, but the functions they host are dynamic and unknown. The network must discover and adapt to these changes while routing packets.

3. **Q-Learning for Route Optimization**:
   - The network learns the optimal routes using Q-Learning, where the Q-table is initialized with random values to encourage exploration. Nodes receive rewards when packets pass through them in the correct function sequence.

4. **Gradual Node Integration**:
   - New nodes can be integrated into the network progressively, allowing the system to adapt and explore new routes.

5. **Packet Processing**:
   - Packets are routed through the network, and each node modifies the packet based on the function it is hosting. The packet must pass through all required functions (e.g., A → B → C) in the correct order.


### Dynamic Function Assignment and Routing

In this system, the functions (A, B, C) dynamically appear and disappear at different nodes. The sender node initializes the packet with a list of required functions in a specific order. As the packet traverses the network, each node checks if it hosts the required function and processes the packet accordingly. The packet must follow the correct function sequence before reaching the receiver.


## Q-Value Calculation

The Q-value for a given state-action pair `(s, a)` is updated using the **Bellman equation**:

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$

Where:
- `Q(s, a)` is the current Q-value of taking action `a` in state `s`.
- `α` is the **learning rate**, which controls how much new information overrides the old information.
- `r` is the **reward** received after taking the action.
- `γ` is the **discount factor**, which determines how much future rewards are considered.
- `max Q(s', a')` is the maximum predicted Q-value for the next state `s'` after taking action `a`.


## Simulation Workflow

1. **Initialization**: The network, Q-tables, and parameters are initialized, with functions dynamically assigned to nodes.
2. **Packet Transmission**: In each episode, the sender sends a packet that must pass through nodes with the required functions in sequence.
3. **Node Status Updates**: Functions dynamically appear/disappear, and the network adapts to these changes.
4. **Q-Table Updates**: The Q-table for each node is updated based on the reward received for each hop and successful processing of functions.
5. **Visualization**: After each episode, the network topology and Q-tables are visualized and saved as images.

![Q-Table Visualization](simulation_images/q_tables_episode_1.png)

### Epsilon-Greedy Strategy

The system uses an **epsilon-greedy strategy** to balance exploration and exploitation. With a probability of `epsilon`, the system will randomly explore a neighboring node. Otherwise, it chooses the neighbor with the highest Q-value for exploitation.


### Node Disconnections

Intermediate nodes can disconnect and reconnect randomly, simulating network instabilities. The network dynamically adapts to these changes by updating the status of each node, and offline nodes do not participate in the routing process.


### Network Animation

The simulation provides an animated visualization of the packet's path through the network in each episode, highlighting the selected route with varying colors for each hop. This allows you to observe how the packet dynamically finds its path through the network, adapting to changes in function availability and node status.


### Understanding the Bellman Equation

The Bellman equation is fundamental in reinforcement learning because it expresses the relationship between the current Q-value and the future Q-values. In essence, it breaks down the decision-making process into two parts:
1. **Immediate Reward (`r`)**: This is the reward the agent receives after performing an action.
2. **Future Reward (`max_{a'} Q(s', a')`)**: This is the best possible reward the agent can achieve in the future, starting from the next state.

By combining these two terms, the algorithm learns not only from the immediate feedback but also takes into account how future decisions will affect the overall outcome.


## Running the Simulation

To run the simulation, follow these steps:

1. **Install Required Libraries**:
   - The simulation uses the following Python libraries: `numpy`, `matplotlib`, `networkx`, and `os`. Install them via `pip` if needed:
     ```bash
     pip install numpy matplotlib networkx
     ```

2. **Execute the Script**:
   - Run the main script to execute the simulation for a specified number of episodes. The default is 100 episodes:
     ```bash
     python3 main.py --episodes 100 --epsilon 1.0 --alpha 0.5 --gamma 0.9
     ```

3. **Output**:
   - The simulation generates visual outputs:
     - A visualization of the network and the path taken by the packet.
     - The Q-tables for each node, showing the learned Q-values after each episode.
   - All images are saved in the `simulation_images/` directory.

## System Structure

1. **main.py**:
   - This is the main entry point of the simulation. It handles command-line argument parsing, initializes the simulation parameters, and runs the simulation for a specified number of episodes. It also orchestrates the visualization of the results.

2. **simulation.py**:
   - This script contains the core logic of the simulation, including the Q-learning algorithm, function assignment to nodes, and packet transmission logic. It manages the dynamic assignment of functions to nodes, updates node statuses (e.g., disconnection/reconnection), and handles Q-table updates after each hop.

3. **visualization.py**:
   - This script is responsible for generating visual representations of the network during the simulation. It creates static and animated visualizations of the packet's path through the network, using color gradients to indicate progress. Additionally, it provides insights into which nodes are processing the required functions and how the network adapts over time.
