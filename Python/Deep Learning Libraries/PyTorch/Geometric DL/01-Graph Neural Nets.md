Graph Neural Networks (GNNs) are a specialized type of neural network designed to operate on graph-structured data.  They learn representations of nodes and edges while respecting the graph's structure.

Key points:

* **Input:** Graph data, represented by nodes, edges, and their features.  Adjacency matrix or edge list defines the connections.
* **Output:**  Node embeddings (vector representations), graph embeddings, or edge predictions.
* **Mechanism:** GNNs propagate and aggregate information across the graph through message passing between nodes.  Each node's representation is updated based on its neighbour's features.
* **Types:**  Various GNN architectures exist, including Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs), and GraphSAGE.
* **Applications:**  Widely used in social networks, recommendation systems, drug discovery, and other domains with graph-structured data.

In Graph neural nets we aim to map nodes to d-dimensional embeddings such that similar nodes in the network are embedded close together.

![[Pasted image 20250204094801.png]]

![[Pasted image 20250204094936.png]]
![[Pasted image 20250204095035.png]]

## Different types of graph

Different graph types are crucial for choosing the right GNN architecture and understanding the relationships within your data. Here's a summary in the context of deep learning:

* **Undirected Graphs:**  Edges have no direction, representing symmetric relationships.  For example, in a social network, if "A is friends with B," then "B is friends with A."  GCNs are commonly used with undirected graphs.  @vault mentions [[Undirected Graphs and GCNs]] which might have more information.

* **Directed Graphs:** Edges have a direction, representing asymmetric relationships. For example, in a citation network, "paper A cites paper B" doesn't imply "paper B cites paper A."  Specific GNN variants handle directed edges, adapting the message passing mechanism.

* **Bipartite Graphs:** A graph whose nodes can be divided int two disjoint set **U** & **V** such that every link connects a node in **U** to one in **V**; that is V & V are independent sets. E.g. Actors-to-Movies (they appeared in), User-to-Movies/Products (they rated) etc.

* **Folded/Projected Graphs:** Derived from bipartite graphs by creating connections within one of the two node sets based on their shared connections in the original bipartite graph.
	
	Here's a breakdown:

	- **Original Bipartite Graph:** You have two sets of nodes (e.g., users and items) with connections only between the sets (e.g., purchase history).
	    
	- **Folding/Projection:** You choose one set to focus on (e.g., users). You create connections between two users if they share connections to the same items in the original bipartite graph. The more items two users share, the stronger their connection in the folded graph. This reveals user similarity based on shared item preferences.
	    
	- **Result:** The folded graph is a unipartite graph (single node set) that represents indirect relationships within the chosen set, mediated by the other set.
	![[Pasted image 20250204101731.png]]

* **Heterogeneous Graphs:**  Contain different types of nodes and/or edges. For example, in a knowledge graph, you might have "author" nodes, "paper" nodes, "writes" edges, and "cites" edges.  Heterogeneous GNNs deal with this diversity by incorporating type-specific information during message passing.
	
* **Dynamic Graphs:**  The graph structure changes over time. For example, in a social network, friendships form and break over time. Temporal GNNs (TGNNs) are designed to capture these temporal dynamics.

## Node Features:

1. **Node Degree (k):** The degree of a node v is the number of edges (neighbouring nodes) the node has. Treats all neighbouring nodes equally.
	
	Node degree counts the neighbouring nodes without capturing their importance

2. **Node Centrality (c):** Node centrality measures the importance or influence of a node within a graph.  Different centrality measures quantify importance in different ways. 

	* **Degree Centrality:**  The simplest measure, counting the number of direct neighbours (edges connected to the node).  High degree centrality indicates a node with many connections.
	
	* **Betweenness Centrality:** Measures how often a node lies on the shortest paths between other nodes.  High betweenness centrality suggests a node acts as a bridge or connector.
	  ![[Pasted image 20250204104006.png]]
	
	* **Closeness Centrality:** Measures the average shortest path distance from a node to all other nodes in the graph. High closeness centrality indicates a node that can quickly reach other nodes.
	  ![[Pasted image 20250204104057.png]]
	
	* **Eigenvector Centrality:** Measures a node's influence based on the influence of its neighbours.  A node connected to many influential nodes will have high eigenvector centrality.
	  ![[Pasted image 20250204103935.png]]
	
	* **PageRank Centrality:** Similar to eigenvector centrality, but considers the importance of the neighbours and the number of connections they have.  Used by Google's search algorithm.


3. **Clustering Coefficient:**  The clustering coefficient measures how connected a node's neighbors are to each other. A high clustering coefficient means a node's neighbors tend to form a tightly-knit group. It quantifies the local density of connections around a node.
	![[Pasted image 20250204104308.png]]

4. **Graphlets:** Induced subgraphs of a larger graph. They provide a way to describe the local network structure around a node by counting the occurrences of different graphlet patterns.

	Key aspects of graphlets:

	- **Induced Subgraphs:** A graphlet is formed by selecting a subset of nodes and _all_ the edges that connect them in the original graph.
	- **Isomorphism:** Graphlets are categorized based on their structure, regardless of node labeling. Two graphlets are considered the same if they are isomorphic (have the same shape).
	- **Graphlet Degree Vector (GDV):** A node's GDV is a vector that counts the occurrences of each graphlet type centred around that node. This provides a rich local structural signature.
	  
	_This concept aims to give a local characterization of my surroundings, based at v. for example, if I take two steps out from v, how many triangles are there? how many lines are there? how many V's are there? this vector captures literally all possible topological configurations that you locally have._
	 
	 ![[Pasted image 20250204105348.png]]


![[Pasted image 20250204105838.png]]
![[Pasted image 20250204105814.png]]

---

## Link-Level Predictions

 ![[Pasted image 20250204110135.png]]
 ![[Pasted image 20250204110240.png]]
 ![[Pasted image 20250204110346.png]]![[Pasted image 20250204110346 1.png]]

### Link Level Features:

#### 1.
![[Pasted image 20250204110435.png]]

#### 2.
![[Pasted image 20250204110457.png]]

#### 3. 
![[Pasted image 20250204110545.png]]
![[Pasted image 20250204111302.png]]
![[Pasted image 20250204111348.png]]


## Graph-level Features

We want features that characterize the structure of an entire graph.

![[Pasted image 20250204111602.png]]
#### **Graph Kernels measures the similarity b/w 2 graphs**

Types of graph kernels

1. **Graphlet Kernels:**  functions that measure the similarity between two graphs based on their graphlet distributions.

	Here's a simplified explanation:

	1. **Graphlet Counting:** For each graph, count the occurrences of different graphlet types (e.g., 3-node graphlets, 4-node graphlets). This can be represented as a graphlet degree vector (GDV) for the whole graph or for individual nodes.
	    
	2. **Kernel Function:** A kernel function takes the GDVs of two graphs (or nodes within the graphs) and computes a similarity score. Common kernel functions include the dot product, cosine similarity, or more sophisticated kernels that account for graphlet relationships.
	    
	3. **Similarity Score:** The kernel function outputs a scalar value representing the similarity between the two graphs. A higher score indicates greater structural similarity.
	   
	![[Pasted image 20250204112506.png]]
	![[Pasted image 20250204112552.png]]
	![[Pasted image 20250204112620.png]]


2. **Weisfeiler-Lehman Kernel:** The Weisfeiler-Lehman (WL) kernel is a powerful graph kernel that measures graph similarity based on the iterative color refinement algorithm. It captures structural information by iteratively aggregating and hashing node labels.

	Here's a simplified explanation:
	
	1. **Initial Labelling:** Assign each node a unique initial label (or color).
	
	2. **Iterative Refinement:** For multiple iterations:
	    * Aggregate the labels of each node's neighbours.
	    * Hash the aggregated label to create a new, unique label for each node.  This new label reflects the local neighbourhood structure.
	
	3. **Kernel Calculation:** After a fixed number of iterations, create a feature vector for each graph by counting the occurrences of each unique label.  The WL kernel is then computed as the dot product of these feature vectors.
	
	**Key Advantages:**
	
	* **Captures Structural Information:** The iterative refinement process captures increasingly complex structural patterns.
	* **Efficient Computation:** The hashing scheme allows for efficient computation of the kernel.
	* **Widely Applicable:**  Can be used with various kernel-based machine learning methods.
	
	**Relationship to Graphlet Kernels:**
	
	The WL kernel can be seen as a generalization of graphlet kernels.  It implicitly counts tree-shaped graphlets up to a certain size, determined by the number of iterations.
	
	![[Pasted image 20250204112922.png]]
	![[Pasted image 20250204112956.png]]
	![[Pasted image 20250204113100.png]]
	![[Pasted image 20250204113126.png]]
	![[Pasted image 20250204113145.png]]
	![[Pasted image 20250204113221.png]]
	![[Pasted image 20250204113250.png]]
	![[Pasted image 20250204113320.png]]

   
![[Pasted image 20250204113417.png]]
