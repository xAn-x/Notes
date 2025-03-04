
Flax NNX is a new simplified API that is designed to make it easier to create, inspect, debug, and analyse neural networks in [JAX](https://jax.readthedocs.io/). This allows users to express their models using regular Python objects, which are `modeled as PyGraphs (instead of pytrees)`, enabling reference sharing and mutability.

```bash
! pip install -U flax treescope
```

### Flax NNX Module system

The main difference between the Flax[`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html) and other `Module` systems in [Flax Linen](https://flax-linen.readthedocs.io/en/latest/api_reference/flax.linen/module.html) or [Haiku](https://dm-haiku.readthedocs.io/en/latest/notebooks/basics.html#Built-in-Haiku-nets-and-nested-modules) is that in NNX everything is **explicit**.

1. The [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html) itself holds the state (such as parameters) directly.
2. The [PRNG](https://jax.readthedocs.io/en/latest/random-numbers.html) state is threaded by the user.
3. All shape information must be provided on initialization (no shape inference).

```python
# import flax.linen as nn # old api
from flax import nnx
import jax
import jax.numpy as jnp

class Linear(nn.Module):
	def __init__(self,fin:int,fout:int,*,rngs:nnx.Rngs):
		keys=rngs.params()
		self.w=nnx.Param(jax.random.uniform(key,(fin,fout)))
		self.b=nnx.Param(jax.zeros(fout,))

	def __call__(self,x:jax.Array):
		return jnp.dot(x,self.w)+self.b

linear=Linear(2,5,rngs=nnx.Rngs(params=0)) # Eager Init
y=model(x=jnp.ones((1,2))) 

print(y)
nnx.display(linear) # Help in model viz,generated using treescope lib
```


### Stateful computation

Implementing layers, such as [`nnx.BatchNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm), requires performing state updates during a forward pass. In Flax NNX, you just need to create a [`nnx.Variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Variable) and update its `.value` during the forward pass.

```python
class Count(nnx.Variable): pass

class Counter(nnx.Module):
  def __init__(self):
    self.count = Count(jnp.array(0))

  def __call__(self):
    self.count += 1

counter = Counter()
print(f'{counter.count.value = }')
counter()
print(f'{counter.count.value = }')
```


### Model surgery

Flax [`nnx.Module`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/module.html)s are mutable by default. This means that their structure can be changed at any time, which makes [model surgery](https://flax.readthedocs.io/en/latest/guides/surgery.html) quite easy, as any sub-`Module` attribute can be replaced with anything else, such as new `Module`s, existing shared `Module`s, `Module`s of different types, and so on. Moreover, [`nnx.Variable`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/variables.html#flax.nnx.Variable)s can also be modified or replaced/shared.

```python
# Can create our custom layer
class Linear(nn.Module):
	def __init__(self,fin:int,fout:int,*,rngs:nnx.Rngs):
		keys=rngs.params()
		self.w=nnx.Param(jax.random.uniform(key,(fin,fout)))
		self.b=nnx.Param(jax.zeros(fout,))

	def __call__(self,x:jax.Array):
		return jnp.dot(x,self.w)+self.b

# Nesting layers and modules using sub-classing
class MLP(nnx.Module):
	def __init__(self,fin:int,hidden:int,fout:int,rngs:nnx.Rngs):
		self.linear1=Linear(fin,hidden,rngs=rngs)
		self.dropout=nnx.Dropout(rate=0.1,rngs=rngs)
		self.bn=nn.BatchNorm(hidden,rngs=rngs)
		self.linear2=Linear(hidden,fout,rngs=rngs)

	def __call__(self,x:jax.Array):
		x=nnx.gelu(self.dropout(self.bn(self.linear1(x))))
		return self.linear(x)


# To store stateful contexts,use nnx.Variable if vector
class LoraParam(nnx.Param): pass

# Custom Lora-layer
class LoraLinear(nnx.Module):
  def __init__(self, linear: Linear, rank: int, rngs: nnx.Rngs):
    self.linear = linear
    self.A = LoraParam(jax.random.normal(rngs(), (linear.din, rank)))
    self.B = LoraParam(jax.random.normal(rngs(), (rank, linear.dout)))

  def __call__(self, x: jax.Array):
    return self.linear(x) + x @ self.A @ self.B

rngs = nnx.Rngs(0)
model = MLP(2, 32, 5, rngs=rngs)

# Model surgery.
model.linear1 = LoraLinear(model.linear1, 4, rngs=rngs)
model.linear2 = LoraLinear(model.linear2, 4, rngs=rngs)

y = model(x=jnp.ones((3, 2)))

nnx.display(model)

# optax: contain optimizers 
import optax
optimizer=nnx.Optimizer(model,optax.Adam(1e-3)) # refrence sharing
# The optimizer holds a mutable reference to the model - this relationship is preserved inside the train_step function making it possible to update the model’s parameters using the optimizer alone.


@nnx.jit # auto state management
def train_step(model:MLP,optimizer:nnx.Optimizer,x:jax.Array,y:jax.Array):
	def loss_fn(model:MLP):
		pred=model(x)
		return jnp.mean((y_pred-y)**2)

	loss,grads=jnp.value_and_grad(loss_fn)(model)
	optimize.update(grads)
	return loss

x,y=jnp.ones((5,2)),jnp.ones((5,10))
for epoch in range(100):
	loss=train_step(model,optimizer,x,y)
	print(f"EPOCH:{epoch}:")
	print(f'loss:{loss},optimizer-step-val:{optimizer.step.value = }')
```