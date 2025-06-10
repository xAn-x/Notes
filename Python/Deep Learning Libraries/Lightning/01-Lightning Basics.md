Lightning is the deep learning framework for professional AI researchers and machine learning engineers who need maximal flexibility without sacrificing performance at scale.

>>> pip install lightning

Lightning organizes PyTorch code to remove boilerplate and unlock scalability.

## *Steps to incorporate lightning:*

```python
# 1. Define lightning module
import lightning as L
import torch
import torch.nn.functional as F
import torch.nn as nn

class LitAutoEncoder(L.LightningModule):
	def __init__(self,encoder,decoder):
		super().__init__()
		self.encoder=encoder
		self.decoder=decoder

	# Define training,validation and test step and lightning will use it automatically
	def common_step(self,x,targets=None):
		x=x.view(x.size(0),-1)
		out=self.encoder(x)
		out=self.decoder(x)
		loss=None
		if targets is not None:
			loss=F.mse_loss(out,targets.view(targets.shape(0),-1))
		return out,loss

	# The respective dls will be auto selected to supply batch while traing and eval
	def training_step(self,batch,batch_idx):
		out,loss=self.common_step(batch[0],batch[1])
		# Logging to TensorBoard (if installed) by default
		self.log("training_loss",loss)
		return loss

	def validation_step(self,batch,batch_idx):
		out,loss=self.common_step(batch[0],batch[1])
		# Logging to TensorBoard (if installed) by default
		self.log("validation_loss",loss)
		return loss

	def test_step(self,batch,batch_idx):
		out,loss=self.common_step(batch[0],batch[1])
		# Logging to TensorBoard (if installed) by default
		self.log("testing_loss",loss)
		return loss

	def predict_step(self,x):
		# -- make pred --
		out,_=self.common_step(x)
		# u can customize to directly serve the output rather than logits
		return out

	# This method will be use to initalize any optimizer or scheduler for the model
	def configure_optimizers(self,lr=3e-4):
		optimizer=torch.nn.optim.Adam(self.parameters(),lr=lr)
		return optimizer

	# U can define ur custom forward and backward pass to
	def backward(self,loss):
		loss.backward()
		
# Creating a instance of lightning Module
autoEncoder=LitAutoEncoder(encoder,decoder)

# Lightning provide with many helpfull callbacks that u can use
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
early_stopping=EarlyStopping(monitor="val_accuracy", min_delta=0.01, patience=3, 	 verbose=False, mode="max",check_val_evey_n_epoch=None,val_check_interval=0.5)

# check validation set after completing 50% of a epochs {i.e 1/2 batches in 1 epoch}


# Getting summary of your model like in keras
from lightning.pytorch.utilitis.model_summary import ModelSummary
summary = ModelSummary(model, max_depth=-1)
print(summary)

# Train the model using Trainer obj of lightning Module
# Trainer obj contains flags to automate a lot of things
trainer=L.Trainer(accelerator="auto",devices=2,accumulates_grad_batches=2,max_epochs=10
			callbacks=[early_stopping])

trainer.fit(model=autoencoder,train_dataloaders=train_dls,validation_dataloaders=val_dls)

# to visualize metrics
%reload_ext tensorboard # this if using Notebook enviorments
%tensorboard --logdir=lightning_logs/

# to load a pretrained model
checkpoint="./lightning_logs/version_0/checkpoints/epoch=0-step=100.ckpt"
autoEncoder=LitAutoEncoder.load_from_checkpoint(checkpoint,encoder=encoder,decoder=decoder)
```


## *Profiling:*
Profiling helps you find bottlenecks in your code by capturing analytics such as how long a function takes or how much memory is used.

## *Measure accelerator usage:*
Another helpful technique to detect bottlenecks is to ensure that you’re using the full capacity of your accelerator (GPU/TPU/HPU). This can be measured with the [`DeviceStatsMonitor`](https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.DeviceStatsMonitor.html#lightning.pytorch.callbacks.DeviceStatsMonitor "lightning.pytorch.callbacks.device_stats_monitor.DeviceStatsMonitor"):

```python
import lightning.pytorch.profiller as profiller
from lightning.pytorch.callbacks import DeviceStatsMonitor

simple_profiler=profiler.SimpleProfiler()
trainer=Trainer(profiler=simple_profiler,callbacks=[DeviceStatsMonitor(gpu_stats=True)])
# or
trainer=Trainer(profiller='advance')

# Once u call fit() u will get a profile report
```