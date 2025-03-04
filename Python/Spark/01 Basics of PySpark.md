 PySpark is the Python API for Apache Spark.  _Spark is a distributed computing framework designed for large-scale data processing._ PySpark lets you write Spark applications using Python.  It's excellent for:

* **Big Data Processing:** Handling datasets too large to fit in a single machine's memory.
* **Parallel Processing:** Distributing computations across a cluster of machines for speed.
* **Machine Learning:**  PySpark's `mllib` library provides tools for various machine learning tasks.
* **Data Wrangling:**  Using Spark's Data-frame API for data manipulation and transformation.

 
  ## Spark Architecture:
  ![[Pasted image 20241208120334.png]]
  
  Centred around a **master-slave** (or driver-executor) model, designed for distributed processing of large datasets.  
  
* **Driver Program:** This is the main program that orchestrates the entire Spark job. It runs on a single machine and is responsible for:
    * Reading the input data.
    * Creating a directed acyclic graph (DAG) representing the computation.
    * Optimizing the DAG.
    * Scheduling tasks on the executors.
    * Aggregating the results from the executors.

* **Executors:** These are worker processes that run on different machines in a cluster.  Each executor receives tasks from the driver and executes them. They have their own memory and processing capabilities.

* **Cluster Manager:**  This manages the resources of the cluster (e.g., YARN, Mesos, Kubernetes, or standalone mode).  The driver program registers with the cluster manager to acquire resources for the executors.

* **Data Storage:** Spark supports various data storage systems, including HDFS, local file systems, and cloud storage (e.g., AWS S3).  Data is typically partitioned across the cluster for parallel processing.

In essence, the driver breaks down the job into smaller tasks, distributes them to the executors, and then combines the results. _This parallel processing allows Spark to handle massive datasets efficiently_.  The key is the resilient distributed dataset (RDD), an abstraction that allows Spark to track data lineage and recover from failures.

## Benefits of Spark

![[Pasted image 20241208121251.png]]

**In-Memory Computation:** Spark primarily performs computations in memory, significantly speeding up processing compared to disk-based systems.  It caches intermediate results in memory across the cluster, minimizing redundant computations.  This is crucial for iterative algorithms.

**Lazy Evaluation:** Spark doesn't execute operations immediately. Instead, it builds a directed acyclic graph (DAG) of transformations, delaying execution until an action (like `collect` or `count`) is called. This optimization allows for efficient optimization and avoids unnecessary computations.

**Fault Tolerance:** Spark's RDDs maintain lineage information, allowing it to recover lost data or computation by re-executing the necessary transformations from the original data. This ensures resilience against node failures.

**Partitioning:** Data is divided into partitions, distributed across the cluster.  This parallel processing enables faster execution of operations.  Choosing the right partitioning strategy is crucial for performance optimization.

### SparkSession

 The SparkSession instance is the way Spark executes user-defined manipulations across the cluster. 

```python
spark.range(100).toDF("number")
```

### DataFrames

Simply represents a table of data with rows and columns. 
Spark DataFrame can span thousands of computers. It ’s quite easy  to convert to Pandas (Python) DataFrames to Spark DataFrames.

### Partitions & Transformations

1. _Partitions:_ In order to allow every executor to perform work in parallel, Spark breaks up the data into chunks, called partitions. A partition is a collection of rows that sit on one physical machine. 
   
   If you have one partition, Spark will only have a parallelism of one even if you have thousands of executors. If you have many partitions, but only one executor Spark will still only have a parallelism of one because there is only one computation resource. 

2. 