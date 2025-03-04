
**JAX: The Power of NumPy, with a Twist**

Imagine NumPy, the workhorse of Python for numerical computing, but with superpowers. That's essentially what JAX is.  It's a library that lets you write numerical code just like you would with NumPy, but it adds these key features:

* **Automatic Differentiation:** JAX can automatically calculate derivatives of your functions. This is crucial for training neural networks, where you need to adjust the weights based on how well the network is performing.
* **Just-in-Time (JIT) Compilation:** JAX can compile your Python code to highly optimized machine code, making it run much faster, especially on GPUs.
* **XLA (Accelerated Linear Algebra):** JAX leverages Google's XLA library to efficiently perform linear algebra operations, which are fundamental to deep learning.

**Why Use JAX?**

* **Flexibility:** JAX is highly flexible and allows you to write code in a way that's both readable and performant.
* **Speed:**  JAX's JIT compilation and XLA integration make it incredibly fast, especially for large-scale deep learning tasks.
* **Research-Friendly:** JAX is popular in research because it allows for rapid prototyping and experimentation.


```bash
pip install --upgrade jax
pip install --upgrade jax[cuda]
```

```python
import jax
import jax.numpy as jnp

# 1. Basic Operations:  JAX mirrors NumPy's API closely

arr = jnp.array([1, 2, 3]) 
brr = jnp.arange(10, dtype=jnp.float32) 

print(arr + brr)  # Element-wise addition
print(jnp.dot(arr, brr))  # Dot product
print(jnp.sin(arr))  # Trigonometric functions
print(arr.devices()) 


# 2. Device Arrays:  JAX manages data placement automatically
print(arr.__class__)  # Output: jaxlib.xla_extension.DeviceArray


# 3. Device Management:  Explicitly moving data to CPU or accelerator
cpu_arr = jax.device_get(arr)  # Moves data to CPU
acc_arr = jax.device_put(arr)  # Moves data to accelerator (if available)


# 4. Immutability:  JAX arrays are immutable
arr_new = arr.at[:-1].set(jax.arange(2, 4))  # Creates a new array with updated values

# 5. Automatic Differentiation:  The heart of JAX
def my_function(x):
  return x**2 + 2*x
grad_fn = jax.grad(my_function)  # Get the gradient function
x = 3.0
gradient = grad_fn(x)  # Calculate the gradient at x = 3

print(f"Gradient of my_function at x = {x}: {gradient}")  # Output: 8.0


# 6. JIT Compilation:  Boosting performance
@jax.jit
def my_function(x):
  return x**2 + 2*x


# 7. XLA:  Accelerated Linear Algebra
A = jnp.array([[1, 2], [3, 4]])
B = jnp.array([[5, 6], [7, 8]])

C = jax.lax.dot(A, B)  # Matrix multiplication using XLA
```

**Key Points:**

- **JAX is built on NumPy:** You can use most of the familiar NumPy functions and operations.
- **Device Arrays:** JAX handles data placement seamlessly, but you can control it if needed.
- **Immutability:** JAX's functional approach means you create new arrays instead of modifying existing ones.
- **Automatic Differentiation:** JAX's core feature for training neural networks.
- **JIT Compilation:** Makes your code run faster, especially for complex operations.
- **XLA:** Provides efficient linear algebra operations.

---

**1. `jax.grad`:  Automatic Differentiation**

* **What it does:**  `jax.grad` lets you calculate the derivative of a function automatically.  This is essential for training neural networks, where you need to adjust the weights based on the gradient of the loss function.

* **Example:**

```python
import jax
import jax.numpy as jnp

def my_function(x):
  return x**2 + 2*x

grad_fn = jax.grad(my_function)  # Get the gradient function

x = 3.0
gradient = grad_fn(x)  # Calculate the gradient at x = 3

print(f"Gradient of my_function at x = {x}: {gradient}")  # Output: 8.0
```

* **Key Points:**

    * `jax.grad` takes a function as input and returns a new function that calculates the gradient.
    * It works for functions with scalar or vector inputs.
    * You can use it to find the gradient of any differentiable function.

**2. `jax.jit`:  Just-in-Time Compilation**

* **What it does:**  `jax.jit` compiles your Python code to highly optimized machine code, making it run much faster, especially on GPUs.

* **Example:**

```python
import jax
import jax.numpy as jnp

@jax.jit
def my_function(x):
  return x**2 + 2*x


@jax.jit
def my_function(x, constant):
  return x + constant

x_array = jnp.arange(4).reshape(2, 2)

# Apply my_function with 'constant' as a static argument
results = my_function(x_array, 5)

print(results)  # Output: [[ 5  6] [ 7  8]]
```

* **Key Points:**

    * You decorate a function with `@jax.jit` to enable JIT compilation.
    * JIT compilation can significantly speed up your code, especially for computationally intensive tasks.
    * JAX will automatically determine the best compilation strategy based on the hardware available.

- **When to Use `jax.jit`:**

	* **Performance Boost:**  The primary reason to use `jax.jit` is to speed up your code. It's especially beneficial for:
    * **Large-Scale Computations:**  When you're working with large datasets or complex calculations, JIT compilation can significantly reduce execution time.
    * **GPU Acceleration:**  JAX's JIT compiler is designed to take advantage of GPUs, so it can be a huge performance win for deep learning tasks.
    * **Repeated Calls:**  If a function is called multiple times within a loop or during training, JIT compilation can save you time by compiling it only once.

	* **Simplifying Code:**  `jax.jit` can sometimes make your code more concise and easier to read.  For example, if you have a complex function with many nested loops, JIT compilation can often eliminate the need for manual optimization.

- **When to Avoid `jax.jit`:**

	* **Side Effects:**  `jax.jit` works best with pure functions, which means they should not have any side effects (like modifying global variables or printing output).  If your function has side effects, JIT compilation might not work as expected.
	* **Small Functions:**  For very small functions, the overhead of JIT compilation might outweigh the performance gains.  If a function is only called a few times, it's probably not worth the effort to compile it.
	* **Dynamic Control Flow:**  JIT compilation can be less efficient with functions that have dynamic control flow (like if statements or loops that depend on data).  JAX might not be able to optimize these cases as effectively.
	* **Debugging:**  When you're debugging your code, it's often helpful to disable JIT compilation so you can see the exact values of variables at each step.

**Example of a Function Not Suitable for `jax.jit`:**

```python
import jax
import jax.numpy as jnp

def my_function(x):
  global counter  # Accessing a global variable
  counter += 1
  return x**2 + 2*x

counter = 0

# This function will not work well with jax.jit because of the side effect
# of modifying the global variable 'counter'
```


**3. `jax.vmap`:  Vectorization**

* **What it does:**  `jax.vmap` allows you to apply a function to multiple inputs simultaneously, without writing explicit loops.  Think of it as a vectorized version of your function.

* **Example:**

```python
import jax
import jax.numpy as jnp

def my_function(x):
  return x**2 + 2*x

x_array = jnp.array([1, 2, 3])

# Apply my_function to each element of x_array
results = jax.vmap(my_function)(x_array) 

print(results)  # Output: [3, 8, 15]


# Applying vectorization over paricular axis
def my_function(x, y):
  return x * y

x_array = jnp.arange(4).reshape(2, 2)
y_array = jnp.arange(4, 8).reshape(2, 2)

# Apply my_function along the first axis (axis=0) of both arrays
results = jax.vmap(my_function, in_axes=(0, 0))(x_array, y_array)

print(results)  # Output: [[ 0  4] [12 20]]
```

* **Key Points:**

    * `jax.vmap` takes a function and applies it to each element of an array.
    * It's like a "map" function, but for vectorized operations.
    * It can significantly improve performance by avoiding explicit loops.

**4. `jax.pmap`:  Parallel Mapping**

* **What it does:**  `jax.pmap` is similar to `jax.vmap`, but it distributes the computation across multiple devices (like multiple GPUs) for parallel execution.

* **Example:**

```python
import jax
import jax.numpy as jnp

# Assuming you have multiple GPUs available

@jax.pmap
def my_function(x):
  return x**2 + 2*x

x_array = jnp.array([1, 2, 3])

# Apply my_function to each element of x_array in parallel
results = my_function(x_array) 

print(results)  # Output: [3, 8, 15]

import jax
import jax.numpy as jnp


@jax.pmap
def my_function(x, y):
  return x * y

x_array = jnp.arange(4).reshape(2, 2)
y_array = jnp.arange(4, 8).reshape(2, 2)

# Apply my_function along the first axis (axis=0) of both arrays in parallel (default) else specify
results = my_function(x_array, y_array)

print(results)  # Output: [[ 0  4] [12 20]]
```

* **Key Points:**

    * `jax.pmap` is designed for parallel processing.
    * It splits the input data and distributes it across multiple devices.
    * It can significantly speed up your code, especially for large-scale computations.


---

### **RNG handling in Jax**

In libraries in `numpy or pytorch` whenever we call random it uses a Psudo-random-generation algorithm that changes the value every time you call it.

When ever we set a seed in lib like numpy it consumes the key and move to next for for subsequent generation. To create same number again one need to keep resetting the seed again and again.

```python
import numpy as np

np.random.seed(0) # set seed globally
print("individually:", np.stack([np.random.uniform() for _ in range(3)]))
# individually: [0.5488135  0.71518937 0.60276338]


np.random.seed(0) # reset seed globally again
print("all at once: ", np.random.uniform(size=3))
# all at once:  [0.5488135  0.71518937 0.60276338]

# No matter how u produce random-nums they will 1-by-1 or using size they will always be same
```

JAX’s random number generation differs from NumPy’s in important ways, because NumPy’s PRNG design makes it hard to simultaneously guarantee a number of desirable properties. Specifically, in JAX we want PRNG generation to be:

1. reproducible,
2. parallelizable,
3. vectorisable.

*Explicit random state*

In Jax rather than using a global seed we use explicit random state for each function to ensure same number generation every time.

```python
import jax

key=jax.random.key(42)
print(key)
# Array((), dtype=key<fry>) overlaying: [0,42]

# same key ensure same number generation
print(random.normal(key)) # -0.184717
print(random.normal(key)) # -0.184717
```

Now this don't sound good as when creating random number we want our number to be truly random and differs every time, and to achieve this we can use a different key every time.

**The rule of thumb is: never reuse keys (unless you want identical outputs).**

```python
for i in range(3):
  new_key, subkey = random.split(key)
  del key  # The old key is consumed by split() -- we must never use it again.

  val = random.normal(subkey)
  del subkey  # The subkey is consumed by normal().

  print(f"draw {i}: {val}")
  key = new_key  # new_key is safe to use in the next iteration.
```

```bash
draw 0: 1.369469404220581
draw 1: -0.19947023689746857
draw 2: -2.298278331756592
```

*Lack of sequential equivalence*

As in NumPy, JAX’s random module also allows sampling of vectors of numbers. However, JAX does not provide a sequential equivalence guarantee, because doing so would interfere with the vectorization on SIMD hardware (requirement #3 above).

```python
key = random.key(42)
subkeys = random.split(key, 3)
sequence = np.stack([random.normal(subkey) for subkey in subkeys])
print("individually:", sequence)

key = random.key(42)
print("all at once: ", random.normal(key, shape=(3,)))

# Number will differ as in 1st we are using 3 differ keys for 3 different number but in 2nd we are using same key to sample 3 different num.
```

```bash

individually: [-0.04838832  0.10796154 -1.2226542 ]
all at once:  [ 0.18693547 -1.2806505  -1.5593132 ]
```

---

## Simple Neural Net in Jax


```python
import jax
import jax.numpy as jnp
from typing import List, Tuple

def get_params(f_in: int, f_out: int, key:jnp.random.key) -> Tuple[jnp.ndarray, jnp.ndarray]:
    subkeys = jax.random.split(key, 2)
    weights = jax.random.normal(subkeys[0], (f_in, f_out))
    bias = jax.random.normal(subkeys[1], (1, f_out))
    return weights, bias

def layer(params: Tuple[jnp.ndarray, jnp.ndarray], x: jnp.ndarray) -> jnp.ndarray:
    w, b = params
    return jax.nn.relu(jnp.dot(x, w) + b)  

def forward_pass(params: List[Tuple[jnp.ndarray, jnp.ndarray]], x: jnp.ndarray) -> jnp.ndarray:
    for layer_params in params:
        x = layer(layer_params, x)
    return x

def loss_fn(params: List[Tuple[jnp.ndarray, jnp.ndarray]], x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    preds = forward_pass(params, x)
    return jnp.mean(jnp.square(preds - y))

def build_model(layer_sizes: List[int], seed: int = 42) -> List[Tuple[jnp.ndarray, jnp.ndarray]]:
    key = jax.random.PRNGKey(seed)
    keys = jax.random.split(key, num=len(layer_sizes) - 1)
    params = [get_params(f_in, f_out, key) for f_in, f_out, key in zip(layer_sizes[:-1], layer_sizes[1:], keys)]
    return params

# Example layer sizes and dummy data
layer_sizes = [728, 100, 10]
model = build_model(layer_sizes)

x = jax.random.normal(jax.random.PRNGKey(0), (32, 728))  
y = jax.random.normal(jax.random.PRNGKey(1), (32, 10))   

# Training loop
lr = 1e-3
for epoch in range(100):
    loss, grads = jax.value_and_grad(loss_fn)(model, x, y)
    
    # Update parameters
    model = [(w - lr * dw, b - lr * db) for (w, b), (dw, db) in zip(model, grads)]
    
    print(f"Epoch {epoch + 1}: Loss = {loss}")
```


