Graph Algorithms Implementation
Graph Visualization Example

Project Overview
This project implements fundamental graph algorithms with performance analysis and visualization capabilities. It includes:

Shortest path algorithms: Dijkstra's and Bellman-Ford

Minimum spanning tree algorithms: Prim's and Kruskal's

Graph traversals: BFS and DFS

Graph analysis: Diameter computation and cycle detection

Features
✔️ Comprehensive Algorithm Suite
✔️ Performance Benchmarking
✔️ Visualization Tools
✔️ Automated Data Pipeline
✔️ Detailed Result Logging

Installation
Clone the repository:

bash
git clone https://github.com/yourusername/graph-algorithms.git
cd graph-algorithms
Install dependencies:

bash
pip install -r requirements.txt
Usage
Quick Start
Run the complete pipeline (downloads dataset and executes all algorithms):

bash
python prepare_and_run.py
Manual Execution
Run specific algorithms on your graph file:

python
from graph_algorithms import GraphAlgorithms

# Initialize with your graph file
ga = GraphAlgorithms("your_graph.txt")

# Run individual algorithms
ga.run_bfs(source_node=0)
ga.run_dijkstra(source_node=0)
ga.run_prims()
Creating Test Graphs
Generate a smaller test graph:

bash
python create_small_graph.py
File Structure
graph-algorithms/
├── data/                  # Graph datasets
│   ├── sample_graph.txt   # Main edge list file
│   └── small_graph.txt    # Test subset
├── results/               # Output files
│   ├── visualizations/    # Generated plots
│   ├── traces/            # Algorithm traces
│   └── results.json       # Consolidated results
├── src/                   # Source code
│   ├── graph_algorithms.py # Main implementations
│   ├── prepare_and_run.py  # Automation script
│   └── create_small_graph.py # Test graph generator
├── requirements.txt       # Dependencies
└── README.md              # This file
Output Files
File Pattern	Description
*_result.txt	Algorithm results
*_trace.txt	Step-by-step execution traces
*_mst.png	MST visualizations
*_traversal.png	Traversal paths
results.json	Consolidated metrics
Dependencies
Python 3.8+

NetworkX

Matplotlib

Pandas

tqdm

Dataset Information
The default dataset is the ca-GrQc collaboration network from Stanford SNAP:

Nodes: 5,242

Edges: 14,496

Diameter: 17

Average Degree: 5.53

Contributing
Fork the repository

Create your feature branch (git checkout -b feature/your-feature)

Commit your changes (git commit -m 'Add some feature')

Push to the branch (git push origin feature/your-feature)

Open a pull request
