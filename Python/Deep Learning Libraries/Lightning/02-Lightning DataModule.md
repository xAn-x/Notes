
A datamodule is a shareable, reusable class that encapsulates all the steps needed to process data:

A datamodule encapsulates the five steps involved in data processing in PyTorch:

1. Download / tokenize / process.
    
2. Clean and (maybe) save to disk.
    
3. Load inside [`Dataset`](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset "(in PyTorch v2.4)").
    
4. Apply transforms (rotate, tokenize, etc…).
    
5. Wrap inside a [`DataLoader`](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader "(in PyTorch v2.4)").

## Why do I need a DataModule?[](https://lightning.ai/docs/pytorch/stable/data/datamodule.html#why-do-i-need-a-datamodule)

In normal PyTorch code, the data cleaning/preparation is usually scattered across many files. This makes sharing and reusing the exact splits and transforms across projects impossible.

Datamodules are for you if you ever asked the questions:

- what splits did you use?
    
- what transforms did you use?
    
- what normalization did you use?
    
- how did you prepare/tokenize the data?

_It encapsulates training, validation, testing, and prediction dataloaders, as well as any necessary steps for data processing, downloads, and transformations._


```python
import lightning as L

# DataModule:
class MNISTDataModule(L.DataModule):
	def __init__(self,data_dir,batch_size,tfms=None):
		super().__init__()
		self.data_dir=data_dir
		self.batch_size=batch_size
		self.tfms=tfms

	# how to download,tokenize etc...
	def prepare_data(self):
		pass

	# how to split,define_datasets etc...
	def setup(self):
		self.train_ds=MNIST(data_dir,train=True)
		val_ds=MNIST(data_dir,train=False)
		
		# splitting val_ds into val and test ds
		torch.random_seed(42)
		val_size,test_size=int(0.8*len(val_ds)),int(0.2*len(val_ds))
		self.val_ds,self.test_ds=random_split(val_ds,[val_size,test_size],
			generator=torch.Generator().manual_seed(42))

	def train_dataloader(self):
		return DataLoader(self.train_ds,batch_size=self.batch_size,transform=self.tfms)

	def val_dataloader(self):
		return DataLoader(self.val_ds,batch_size=self.batch_size,transform=self.tfms)

	def test_dataloader(self):
		return DataLoader(self.test_ds,batch_size=self.batch_size,transform=self.tfms)

	def predict_dataloader(self):
		# if u have u can return that
		pass

	# There are many other methods like transfer_batch_to_device,etc read about em

trainer=L.Trainer(...)
MNIST=MNISTDataModule(path)
trainer.fit(model=model,datamodule=MNISTDataModule)
```

==Warning==
`prepare_data` is called from the main process. It is not recommended to assign state here (e.g. `self.x = y`) since it is called on a single process and if you assign states here then they won’t be available for other processes.
