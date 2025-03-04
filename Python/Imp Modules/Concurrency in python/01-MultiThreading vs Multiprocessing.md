
# Understanding GIL(Global Interpreter Lock):
## What is GIL:
The Global Interpreter Lock is a mutex that protects access to Python objects. ___It ensures that only one thread executes Python bytecode at a time.___ This lock is necessary because Python’s memory management is not thread-safe. ___Without the GIL, simultaneous access to Python objects from multiple threads could lead to inconsistent or corrupted data.___

## Why Does Python Use the GIL?
1. __Simplifies memory management__: Python’s memory management, especially reference counting for garbage collection, is not thread-safe. The GIL ensures that memory management operations, like incrementing or decrementing reference counts, are atomic and safe from race conditions.  
2. **Ease of Integration with C Libraries:** Python is often used as a scripting language to interface with C libraries. Many C libraries are not thread-safe. The GIL provides a simple way to ensure that Python’s interactions with these libraries remain safe and consistent.  
    It also simplifies the integration of C extensions, as developers don’t have to worry about making their code thread-safe.

# Multithreading

**Multithreading** involves _running multiple threads within a single process_. Each thread runs independently but shares the same memory space, making it _useful for tasks that involve a lot of waiting, such as I/O operations_ (reading and writing files, and handling network requests).

## When to Use Multithreading:
- When the program involves I/O-bound tasks, such as reading from or writing to files, network communication, or database operations.
- When tasks can run concurrently and are not CPU-intensive.
- When the application needs to maintain a shared state or memory.

### Pros:
- Efficient CPU utilization
- Lower resource overhead (due to shared resource) compared to multiprocessing
- Easier intercommunication because of shared resources
### Cons:
- GIL in Python can slow the performance of CPU-bound tasks, preventing true parallelism.
- Hard to debug due to potential race conditions and deadlocks

# Multiprocessing:

**Multiprocessing** _involves running multiple processes, each with its own memory space._ This technique is particularly _useful for CPU-bound_ tasks where the main limitation is the CPU’s processing power. _Each process runs independently, allowing true parallelism_, especially on multi-core systems.

Multiprocessing is a concept for a system and not network, if u want to involve multiple system (distributed computing) network to process some task then use Distributed Computing.

## When to Use Multiprocessing:
- For CPU-bound tasks, such as mathematical computations, data processing, or any operation that requires significant CPU resources.
- When tasks need to be truly parallel.
- When separate memory spaces for tasks are beneficial, avoiding shared memory issues.

### Pros:
- True parallelism, is especially useful for CPU-bound tasks, as each process can run on a separate core.
- Each process has its own memory space, reducing the risk of memory corruption.
- Better performance on multi-core systems.

### Cons:
- Higher overhead due to the creation of separate processes.
- More complex inter-process communication (IPC) compared to threading.
- Increased memory usage since each process has its own memory space.


```python
import threading
import multiprocessing

def read_file(file_path:str):
	# This will take time if file is large
	with open(file_path,'r') as file:
		text=file.read_lines(files)
	return text

def clean_data(data):
	processed_data=[]
	for line in data:
		# clean-up oprns
		processed_data.append(data)
	return '\n'.join(processed_data)


# Synchronous way:
file_paths=['file1','file2','file3']
file_content,processed_data=[],[]
for file_path in file_paths:
	file_content.append(read_file(file_path))

for content in file_content():
	processed_data.append(clean_data(content))



# Multithreaded way: For I/O bound tasks
file_content=[]
threads=[]
for file_path in file_paths:
	t=threading.Thread(target=read_file,args=(file_path,)) # will create a thread
	t.start() # will start executing the thread and won't 
	threads.append(t)

# Fetching results
for thread in threads:
	thread.join() # wait for thread to complete its execution
	# Note since I am waiting for thread to complete execution,python won't run anything ahead here unitl
	# it is complete
	file_content.append(thread.result) # get the generated result and put in content



# MultiProcessing way: For CPU intensive bound tasks
processes=[]
processed_data=[]
for content in file_content:
	p=multiprocessing.Process(target=clean_data,args=(content,))
	p.start() # Will start this process in parallel,with its own resources
	process.append(p)

# Getting results
for process in processs:
	process.join() # wait for process to complete its execution
	# Note since I am waiting for process to complete execution,python won't run anything ahead here unitl
	# it is complete
	processed_data.append(process.result) # get the generated result and put in content
```

|           | Multi-Threading                                                                                                                       | Multi-Processing                                                                                                                                                         |     |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --- |
| Resources | Shared resources among different thread                                                                                               | Fresh Resource for each process                                                                                                                                          |     |
| GIL       | limits the effectiveness of multithreading in Python for CPU-intensive operations, as it prevents true parallel execution of threads. | everages multiple CPU cores by running separate processes, each with its own memory space and GIL, enabling true parallelism and efficient utilization of CPU resources. |     |
| Execution | May or may-not concurrent                                                                                                             | Concurrent                                                                                                                                                               |     |

# Alternative of Multi-threading and multi-processing: 

## Asyncio (Asynchronous I/O)

**Asyncio** is a library to write concurrent code using the async/await syntax. It is primarily used for I/O-bound tasks where the program needs to handle multiple connections or perform many I/O operations concurrently without blocking the main thread. It is is a single threaded in nature but does not wait for program taking time.

It is really important and widely used so, is discussed in details in other [[03-AysncIO]]
### Pros:
- Suitable for I/O-bound tasks.
- Doesn’t require multiple threads or processes, avoiding the overhead associated with them.
- Can be more efficient in terms of memory and CPU usage.

### Cons:
- Requires a different programming model (async/await), which can be more complex to understand and implement.

## Concurrent.futures

The `Concurrent.futures` module provides a high-level interface for asynchronously executing callables using threads or processes.

### Pros:
- Simplifies working with threads and processes through a high-level interface.
- Abstracts away the low-level details of thread and process management.

### Cons:
- Still subject to GIL for threads, the higher overhead for processes.

```python
import concurrent.futures

def read_file(file_path:str):
	# This will take time if file is large
	with open(file_path,'r') as file:
		text=file.read_lines(files)
	return text

def clean_data(data):
	processed_data=[]
	for line in data:
		# clean-up oprns
		processed_data.append(data)
	return '\n'.join(processed_data)

file_paths=['file1','file2','file3']

with concurrent.futures.ThreadPoolExecutor() as executor():
	futures=[executor.submit(read_file,file_path) for file_path in file_paths]
	
	for future in concurrent.futures.as_completed(futures):
        result = future.result()
        # Perform additional operations with the result

with concurrent.futures.ProcessPoolExecutor() as executor():
	futures=[executor.submit(clean_data,file_content) for file_content in file_contents]
	processed_data=[future.result() for future in futures]
```