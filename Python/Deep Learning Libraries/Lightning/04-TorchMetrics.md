TorchMetrics is a collection of 100+ PyTorch metrics implementations and an easy-to-use API to create custom metrics. It offers:

- A standardized interface to increase reproducibility
    
- Reduces Boilerplate
    
- Distributed-training compatible
    
- Rigorously tested
    
- Automatic accumulation over batches
    
- Automatic synchronization between multiple devices

You can use TorchMetrics in any PyTorch/Lightning model


## *Using TorchMetrics:*

### Functional Metrics:
Similar to [torch.nn](https://pytorch.org/docs/2.4/nn), most metrics have both a class-based and a functional version. The functional versions implement the basic operations required for computing each metric. They are simple python functions that as input take [torch.tensors](https://pytorch.org/docs/2.4/tensors.html) and return the corresponding metric as a `torch.tensor`

```python
import torchmetrics

# simulate a classification problem
preds = torch.randn(10, 5).softmax(dim=-1)
target = torch.randint(5, (10,))

acc = torchmetrics.functional.accuracy(preds, target, task="multiclass", num_classes=5)
```

### Module Metrics:
Nearly all functional metrics have a corresponding class-based metric that calls it a functional counterpart underneath. The class-based metrics are characterized by having one or more internal metrics states (similar to the parameters of the PyTorch module) that allow them to offer additional functionalities:

- Accumulation of multiple batches
    
- Automatic synchronization between multiple devices
    
- Metric arithmetic

```python
import torch
import torchmetrics

# initialize metric
metric = torchmetrics.classification.Accuracy(task="multiclass", num_classes=5)

n_batches = 10
for i in range(n_batches):
    # simulate a classification problem
    preds = torch.randn(10, 5).softmax(dim=-1)
    target = torch.randint(5, (10,))
    # metric on current batch
    acc = metric(preds, target)
    print(f"Accuracy on batch {i}: {acc}")

# metric on all batches using custom accumulation
acc = metric.compute()
print(f"Accuracy on all data: {acc}")

# Resetting internal state such that metric ready for new data
metric.reset()
```

### [All Metrics in TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/all-metrics.html#all-torchmetrics)



