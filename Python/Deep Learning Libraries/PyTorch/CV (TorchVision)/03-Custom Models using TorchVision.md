
## 1. Object-Detection:

```python
import torch
from torch.utils.data import Dataset,DataLoader

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from torchvision.transforms import v2
from torchvision.model import get_model,get_model_weights
from torchvision import tv_tensor
from torchvision.io import read_image

class MyDataset(Dataset):
	def __init__(self,image_dir,df,transforms=None):
		super().__init__()
		self.root=image_dir
		self.df=df
		self.image_files=df['images'].unqiue().values.tolist()
		self.transforms=transforms
		self.n=len(image_files)

	def __len__(self):
		return self.n

	def __getitem__(self,idx):
		filename=self.image_files[idx]
		try:
			image=read_image(os.path.join(self.root,filename))
			boxes=df[df['image']==filename][['x0','y0','x1','y1']].values
			labels=df[df['image']==filename].values.tolist()
		except:
			raise Exception(f"No file: {filename} exsist in {self.root} directory")

		image=tv_tensor.Image(image)
		target={}
		target["boxes"]=tv_tensor.BoundingBoxes(boxes,format="XYXY",canvas_size=image.shape)
		target["cls_ids"]=labels
		target["idx"]=idx

		if self.transform is not None:
			image,target=self.transform(image,target)
		return image,target


preprocess=v2.Compose([
	v2.Resize((224,224)),
	v2.ToDType(torch.unint8),
	v2.Normalize((0.5,),(0.5,))
])

train_ds=MyDataset("train",train_csv,transforms=preprocess)
val_ds=MyDataset("valid",valid_csv,transforms=preprocess)

train_dl=DataLoader(train_ds,batch_size=16,pin_memory=True,num_workers=2,shuffle=True)
valid_dl=DataLoader(valid_ds,batch_size=32,pin_memory=True,num_workers=2)


# Loading Model
model_name='fasterrcnn_resnet50_fpn"
model_weights=get_model_weights(model_name)
model=get_model(model_name,weights=model_weights,num_classes=10)

def train():
	# --- training-code ----

def evaluate():
	evaluate()


# Training and eval using torchvision helper functions
from torchvision.engine import train_one_epoch,evaluate

# train
for epoch in range(3):
	train_one_epoch(model,optimizer,train_dl,device,epoch,print_freq=10)
	lr_scheduler.step()
	# evaluate
	evaluate(model,valid_dl,device)****
```