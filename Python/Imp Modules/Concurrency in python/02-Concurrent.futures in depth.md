
The `concurrent.futures` module in Python provides a high-level interface for asynchronously executing tasks (threads & processes) in parallel. It simplifies the process of writing concurrent code by abstracting away the complexities of thread management and synchronization.

Concurrency is a powerful technique that allows multiple tasks to progress simultaneously, improving performance and responsiveness of applications.

`concurrent.futures` leverage concurrency in Python through two main classes: `ThreadPoolExecutor` and `ProcessPoolExecutor`.

# How to works with async executing tasks with `concurrent.futures:`
### 1.Create pool of threads/process:
Create a pool using `ThreadPoolExecutor` or `ProcessPoolExecutor`

### 2.Submit Tasks to Pool:
There are 2 ways to submit tasks to the executor (Thread/Process) pool:

1. `submit(fn,*args, **kwargs)` - It submits tasks to pool and returns lazy **Future** object for each submitted task. It **does not block** execution of the next code. The task gets executed on the process/thread pool and when completed results are updated in **Future** object. We can keep track of **Future** objects and continue to execute the next code until tasks submitted to the pool are completed.
   
2. `map(func, *iterables, timeout=None, chunksize=1)` - It submits tasks to pool and returns results directly. It also **does not block** execution of the next code after its call. It runs asynchronously and returns the generator object immediately. You can loop through the generator object to retrieve results.

### 3.Collect Results:
When tasks are submitted using **submit()** method (which returns **Future** object) then we can call **result()** method on **Future** object to retrieve task result. _The call to **result()** method returns immediately with the result if a task is completed else it'll block waiting for task completion_. We can time it out using **timeout** parameter to prevent blocking while waiting for the result. After timing out, we can try again after some amount of time for result availability. The other useful methods of **Future** object are listed later on.

When we have _submitted tasks using **map()** method, the pool returns the generator object_. We can loop through this generator to retrieve the results of submitted tasks. Please make a _**NOTE** that when looping through the generator, if a task is not completed then execution will block waiting for its completion_. We can raise exception (**TimeoutError**) if task has not completed after specified number of seconds (int/float) by setting **timeout** parameter of **map()**.

### 4.Shutdown Pool after Completion:
Both **ThreadPoolExecutor** and **ProcessPoolExecutor** classes have a method named **shutdown(wait=True, cancel_futures=False)** that can be used to release all resources occupied by the pool once all threads/processes are done with execution.

- The argument **wait** is **True** by default and it'll block execution until all tasks submitted to the pool are completed. You can set it to **False** if you do not want to block code execution. It is useful in situations when the next line of code has no dependency on task completion.
- The argument **cancel_futures** can be used to cancel tasks that are not started yet by setting the parameter value to **True**. The tasks that are started are not interrupted but pending ones are canceled. If both **wait** and **cancel_futures** are **True** then tasks not started will be canceled.
- The developer can use pools as context managers using **with** statement and it'll call **shutdown()** method automatically without the developer calling it explicitly. It's a good practice to use context managers. Python has a library named **[contextlib](https://coderzcolumn.com/tutorials/python/contextlib-useful-context-managers-with-statements-in-python)** that let us easily create context managers.

##### **NOTE:** Any tasks submitted to the pool after a call to **shutdown()** method will raise **RuntimeError**.


```python
import concurrent.futures
import threading

def read_file(filepath):
	with open(filepath,'r') as file:
		text=file.readlines(file)
	return text

files=['f1','f2','f3','f4','f5']

# Naive way
executor= concurrent.futures.ThreadPoolExecutor(max_workers=8) # start pool
futures=[]
for file in files:
	future=executor.submit(read_file,args=(file,))
	futures.append(future)

data=[]
for future in futures:
	data.append(future.result()) # This will block execution is file haven't read

executor.shutdown()


# Better way: use pool as context maanger: as auto delocation of pooled resources
futures=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
	for file in files:
		futures.append(executor.submit(read_file,file))

data=[]
for future in futures:
	data.append(future.result()) # This will block execution is file haven't read


# Using map(): returns a generator that can be used to directly get the data once completed
futures=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
	results=executor.map(read_file,files,timeout=10)

data=[]
try:
	for data in results:
		data.append(data) # This will block execution is file haven't read,will throw TimeoutException if
		#some file takes more time than timeout provided
except Exception as e:
	print("\n{} : {}\n".format(e.__class__.__name__,e))
```


**submit()** method has benefit over **map()** method that it returns future objects. These objects have a few methods that let us check the status of tasks (running, completed, canceled, failed with an exception, etc.). In the case of **map()**, if we start looping through the result generator and the task is not complete then it'll block even though other tasks are submitted after the waiting task is completed.

## Useful Methods of **"Future"** Objects [](https://coderzcolumn.com/tutorials/python/concurrent-futures#8.-Useful-Methods-of-%22Future%22-Objects-)

Below we have listed important methods of **Future** object returned by **submit()** method that can be used for purposes like testing the current status of tasks, canceling them, retrieving results, etc.

1. **cancel()** method cancels task._**Tasks that are not started yet only that can be cancelled. Once the task has started, it can not be  cancelled.**_.
2. **cancelled()** returns **True** or **False** based on whether task is cancelled or not.
3. **running()** method returns **True** or **False** based on whether the task is running or completed/canceled
4. **done()** method returns **True** if task is **completed/cancelled else returns **False**
5. **result(timeout=None)** returns results of task execution when called. It'll block execution if the task is running and not completed yet.
    - The method has an argument named **timeout** which can be given as time in seconds (int/float). The **result()** waits for that many seconds and if the result is still not available then **concurrent.futures.TimeoutError** is raised.
    - The **timeout** parameter can be used to prevent blocking while waiting for task completion.
    - If **Future** is cancelled before completion then **CancelledError** will be raised.
6. **exception(timeout=None)** method returns an error if any raised during the execution of the task. It returns **None** if execution is successful. The **timeout** parameter has same function as **result()**.
7. **add_done_callback(fn)** method accepts a function that will be executed when a task has been completed/cancelled. The exception raised by this function will be logged and ignored. It takes a future object as its argument.


`as_completed():`

Developers can face situations where they need the results of tasks as they complete and **order does not matter**. Just completion of tasks matters. By default, when we submit tasks to the pool, we simply have a list of futures.

We don't have information on which futures have been completed. We can keep looping through futures to check which has been completed but it can be unnecessary CPU usage. Instead we can use a method named **as_completed()** available from **concurrent.futures**.

- **concurrent.futures.as_completed(futures_list, timeout=None)**: We can give list of future objects to this method and it'll return futures in order as they are completed.
This frees us from constantly checking for task completion. We can monitor futures using this method. It'll return future as it completes. Then, we can retrieve results from completed objects.