_JAX transformation and compilation are designed to work only on Python functions that are functionally pure._

```python
import jax
import jax.numpy as jnp

def impure_print_side_effect(x):
  print("Executing function")  # This is a side-effect
  return x

# The side-effects appear during the first run
print ("First call: ", jit(impure_print_side_effect)(4.))

# Subsequent runs with parameters of same type and shape may not show the side-effect
# This is because JAX now invokes a cached compilation of the function
print ("Second call: ", jit(impure_print_side_effect)(5.))

# JAX re-runs the Python function when the type or shape of the argument changes
print ("Third call, different type: ", jit(impure_print_side_effect)(jnp.array([5.])))
```

```bash
Executing function
First call:  4.0
Second call:  5.0
Executing function
Third call, different type:  [5.]
```


Jax includes no of transformations which operates on Jax functions. These includes:
- `jax.jit():` Just-in-time compilation
- `jax.vmap():` Vectorizing transforms
- `jax.pmap():` Parallelizing transforms
- `jax.grad():` Gradient transforms


## 1. Just-in-time compilation:

JAX accomplishes this by reducing each function into a sequence of primitive operations, each representing one fundamental unit of computation.

```python
import jax
import jax.numpy as jnp

g=0
@jax.jit()
def bad_jit(x):
	print(x) # this won't be traced as impure
	return x+g # g won't be traced as not part of function
print(bad_jit(1)) # 1
g=10
print(bad_jit(1)) # 1: as g is never traced , so changes never detected
print(bad_jit(jax.Array([1]))) # [11]: jit will rerun the function whenever type or shape of arg changes


@jax.jit()
def good_jit(x,g):
	jax.debug.print(x) # this will work but have performance impact
	return x+g # g will be traced 
print(good_jit(1,0)) # 1
print(bad_jit(1,10)) # 11
```

It is _not recommended to use iterators in any JAX function_ you want to `jit` or in any control-flow primitive. The reason is that an iterator is a python object which introduces state to retrieve the next element. Therefore, it is incompatible with JAX functional programming model.

Rather than using iterators use `lax.fori_loop`,`lax.scan` but be careful as can cause unexpected results.

```python
import jax.numpy as jnp
from jax import make_jaxpr

# lax.fori_loop
array = jnp.arange(10)
print(lax.fori_loop(0, 10, lambda i,x: x+array[i], 0)) # expected result 45
iterator = iter(range(10))
print(lax.fori_loop(0, 10, lambda i,x: x+next(iterator), 0)) # unexpected result 0


# lax.scan
def func11(arr, extra):
    ones = jnp.ones(arr.shape)
    def body(carry, aelems):
        ae1, ae2 = aelems
        return (carry + ae1 * ae2 + extra, carry)
    return lax.scan(body, 0., (arr, ones))
make_jaxpr(func11)(jnp.arange(16), 5.)
# make_jaxpr(func11)(iter(range(16)), 5.) # throws error

# lax.cond
array_operand = jnp.array([0.])
lax.cond(True, lambda x: x+1, lambda x: x-1, array_operand)
iter_operand = iter(range(10))
# lax.cond(True, lambda x: next(x)+1, lambda x: next(x)-1, iter_operand) # throws error
```

**JIT and Caching:**

Suppose we define `f = jax.jit(g)`. When we first invoke `f`, it will get compiled, and the resulting XLA code will get cached. Subsequent calls of `f` will reuse the cached code. This is how `jax.jit` makes up for the up-front cost of compilation.

Avoid calling `jax.jit()` on temporary functions defined inside loops or other Python scopes. For most cases, JAX will be able to use the compiled, cached function in subsequent calls to [`jax.jit()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jit.html#jax.jit "jax.jit"). However, because the cache relies on the hash of the function, it becomes problematic when equivalent functions are redefined. This will cause unnecessary compilation each time in the loop:

```python
from functools import partial

def unjitted_loop_body(prev_i):
  return prev_i + 1

def g_inner_jitted_partial(x, n):
  i = 0
  while i < n:
    # Don't do this! each time the partial returns a function with different hash
    i = jax.jit(partial(unjitted_loop_body))(i)
  return x + i

def g_inner_jitted_lambda(x, n):
  i = 0
  while i < n:
    # Don't do this!, lambda will also return a function with a different hash
    i = jax.jit(lambda x: unjitted_loop_body(x))(i)
  return x + i

def g_inner_jitted_normal(x, n):
  i = 0
  while i < n:
    # this is OK, since JAX can find the cached, compiled function
    i = jax.jit(unjitted_loop_body)(i)
  return x + i

print("jit called in a loop with partials:")
%timeit g_inner_jitted_partial(10, 20).block_until_ready()

print("jit called in a loop with lambdas:")
%timeit g_inner_jitted_lambda(10, 20).block_until_ready()

print("jit called in a loop with caching:")
%timeit g_inner_jitted_normal(10, 20).block_until_ready()
```

```bash
jit called in a loop with partials:
243 ms ± 13.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
jit called in a loop with lambdas:
242 ms ± 3.97 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
jit called in a loop with caching:
2.72 ms ± 24.8 μs per loop (mean ± std. dev. of 7 runs, 100 loops each)
```

## 2. Automatic Vectorization:

The primary purpose of `jax.vmap` is to automatically batch a function. For example, if you have a function that takes a single input, `vmap` can make it handle multiple inputs in parallel.

`jax.vmap()` is a powerful function in JAX that lets us vectorize operations without writing explicit loops. When you use `jax.vmap()`, you’re essentially saying, “Apply this function across a batch of inputs in parallel.”

```python
import jax
import jax.numpy as jnp

def convolve(x,w):
	lx,lw=len(x),len(w)
	output=[]
	for i in range():
		output.append(jnp.sum(x[i:i+lw]*w))
	return jnp.Array(output)

x=jnp.arange(1,6)
w=jnp.arange(1,4)
print(convolve(x,w)) # [11,20,29]
```

*1. Manual Vectorization:*

```python
xs=jnp.stack([x,x])
ws=jnp.stack([w,w])

outputs=jnp.stack([convolve(x,w) for x,w in xs,ws])
print(output)
```

```bash
Array([[11., 20., 29.],
       [11., 20., 29.]], dtype=float32)
```

_2. Manual Vectorization and optimization:_

```python
def manually_vectorized_convolve(xs,ws):
	output=[]
	lx,lw=xs.shape[-1],ws.shape[-1]
	for i in range(lx-lw+1):
		output.append(jnp.sum(xs[:,i:i+lw]*ws,axis=1))
	return jnp.stack(output,axis=1)

print(manually_vectorized_convolve(xs,ws))
```

```bash
Array([[11., 20., 29.],
       [11., 20., 29.]], dtype=float32)
```

_3.Automatic Vectorization using `vmap()`:_

```python
auto_batch_convolve=jax.vmap(convolve)
print(auto_batch_convolve(xs,ws))
```

```bash
Array([[11., 20., 29.],
       [11., 20., 29.]], dtype=float32)
```


_`in_axes ` & `out_axes` in vmap():_

The `in_axes` and `out_axes` parameters allow control over how inputs and outputs are batched. Here’s a breakdown:

- **`in_axes`**: Specifies which dimensions of the input arrays should be mapped (or batched) over. Each input to the function can have its own `in_axes` specification.
    
    - `in_axes=0` means we’ll apply `vmap` over the first dimension (batch dimension).
    - `in_axes=None` means the argument won’t be batched and will be used as a constant across all mapped elements.

- **`out_axes`**: Specifies the output's batching behaviour, allowing you to control which axis holds the results after vectorization.

For example, we can implement a matrix-matrix product using a vector dot product:

```python
import jax.numpy as jnp

# we use a,b to indicate vector
vv = lambda x, y: jnp.vdot(x, y)  #  ([a], [a]) -> []
mv = vmap(vv, inaxes=(0, None), out_axes=0)# ([b,a],[a])->[b] (b is the mapped axis)
mm = vmap(mv, in_axes=(None, 1), out_axes=1)# ([b,a],[a,c])->[b,c] (c is the mapped axis)
```

  