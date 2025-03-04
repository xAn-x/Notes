
## Pytorch:

We can entirely replace the custom training loop with fastai’s. That means you can get rid of `train()`, `test()`, and the epoch loop in the original code, and replace it all with just this:

```python
from fastai.imports import *
from migrating_pytorch import * # import all things require for migration

dls=DataLoaders(train_loader,test_loader)
learner=Learner(dls,Model(),loss_func:F.nll_loss,opt_func=Adam,
				metrics=accuracy)
learn.fit_one_cycle(2,lr=1e-2)
```

migrating from pure PyTorch allows you to remove a lot of code, and doesn’t require you to change any of your existing data pipelines, optimizers, loss functions, models, etc.

Once you’ve made this change, you can then benefit from fastai’s rich set of callbacks, transforms, visualizations, and so forth.

### Pytorch to fastai details

1. `Pytorchs Dataloaders to fastai DataLoader:`

```python
import torch,torchvision
from torchvision.transforms import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

tfms=transforms.Compose([
	transforms.Normalize((0.1307,),(0.3081))
])

train_ds=datasets.MNIST("./data",train=True,downloade=True,transforms=tfms)
val_ds=datasets.MNIST("./data",train=False,transforms=tfms)

train_dl=DataLoader(train_ds,batch_size=256,shuffle=True,
					num_workers=2,pin_memory=True)

val_dl=DataLoader(test_ds,batch_size=512,num_workers=2,pin_memory=True)


# Creating fastai DataLoader from torch's
from fastai.data.core import DataLoaders as fs_DataLoader
dls=fs_DataLoader(train_dl,val_dl)
```

2. `Model Building:`
	
```python
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
	def __init__(self,in_chnls, out_chnls, kernel_size=3, stride=1, padding=0):
		self.conv=nn.Conv2d(in_chnls,out_chnls,kernel_size,stride,padding)
		self.actv=nn.ReLU()
		self.pool=nn.MaxPool2d(3,1)
	
	def forward(self,x):
		return self.pool(self.actv(self.conv(x)))



class Net(nn.Module):
	def __init__(self):
		super().__init__()
		self.conv1=ConvBlock(1,32)
		self.conv2(32,64)
		self.flatten=nn.Flatten()
		self.lin1=nn.Linear(inp,128)
		self.actv=nn.ReLU()
		self.dropout=nn.Dropout()
		self.lin2=nn.Linear(128,10)
		
	def forward(self,x):
		x=self.conv1(x)
		x=self.conv2(x)
		x=self.flatten(x)
		x=self.actv(self.lin1(x))
		x=self.dropout(x)
		return self.lin2(x)
```

3. `Pytorch Optimizers in fastai:`

PyTorch optimizers in the fastai framework is made extremely simple thanks to the [`OptimWrapper`](https://docs.fast.ai/optimizer.html#optimwrapper) interface.

Simply write a `partial` function specifying the `opt` as a torch optimizer.

```python
from fastai.optimizer import OptimWrapper
from torch import optim
from functools import partial

opt_func=partial(OptimWrapper,optim.Adam)
```

4. `Training:`

```python
from fastai.learner import Learner
import fastai.callback.schedule # To get `fit_one_cycle`, `lr_find`
from fastai.metrics import accuracy

import torch.nn.functional as F
learner=Learner(dls,Net(),loss_func=F.nll_loss,opt_func=opt_func
				,metrics=accuracy)

learner.fit_one_cycle(2,lr)
# To access any of the above parameters, we look in similarly-named properties such as `learn.dls`, `learn.model`, `learn.loss_func`, and so on.

learn.save('myModel', with_opt=False)
net_dict = torch.load('models/myModel.pth') 
new_net.load_state_dict(net_dict)

with torch.no_grad():
    new_net.cuda()
    tfmd_im = tfmd_im.cuda()
    preds = new_net(transformed_img)

preds.argmax(dim=-1)
```

## Lightning with fastai

```python
from migrating_lightning import *
from fastai.vision.all import *

model = LitModel()
data = DataLoaders(model.train_dataloader(), model.val_dataloader()).cuda()

learn = Learner(data, model, loss_func=F.cross_entropy, opt_func=Adam
				,metrics=accuracy)
learn.fit_one_cycle(1, 0.001)
```

## Taking Benefits of DataBlock API

```python
mnist = DataBlock(blocks=(ImageBlock(cls=PILImageBW), CategoryBlock), 
                  get_items=get_image_files, 
                  splitter=GrandparentSplitter(),
                  get_y=parent_label)

dls = mnist.dataloaders(untar_data(URLs.MNIST_TINY))
dls.show_batch(max_n=9, figsize=(4,4))
```