## 01-High level API (less flexibility but fast): *Avoid if possible*

```python
from fastai.vision.all import * # everything require for cv: models,helper_funcs etc

# Download Dataset
path=untar_data(url="ds-link",path="where-to-store",archive="where-to-extract")

filepath=path/"images"
img_files=get_image_files(filepath) # only get image_files

# Create DataLoaders
def label_func(fn):
	# return the label

pattern=r"""{{Pattern to extract label}}"""

## The label_func get file_name that can be used to extract the label
dls=ImageDataLoaders.from_name_func(
	path,f_names=img_files,label_func=label_func,
	item_tfms=[],batch_tfms=[],img_cls=PIL.Image, # What type of image format we have
	bs=batch_size,val_bs=validation_batch_size,
	device=device,shuffle=True,seed=42,
	valid_pct=0.2,splitter=*splits
)

# The use re on filename to extract label
dls=ImageDataLoaders.from_name_re(
	path,f_names=img_files,pat=pattern,
	item_tfms=[],batch_tfms=[],img_cls=PIL.Image,
	bs=batch_size,val_bs=validation_batch_size,
	device=device,shuffle=True,seed=42,
	valid_pct=0.2,splitter=*splits
)

# Similar to ImageFolder of torch,subdir names will be used for labels
dls=ImageDataLoaders.from_folder(
	path,train='train_path',valid="valid_path",vocab=['list of subdir to use for label'] # if doesn't match->error
	item_tfms=[],batch_tfms=[],img_cls=PIL.Image,
	bs=batch_size,val_bs=validation_batch_size,
	device=device,shuffle=True,seed=42,
	valid_pct=0.2,splitter=*splits
)

# The label_func get the file_path that can be used to extract the label
dls=ImageDataLoaders.from_path_func(
	path,f_names=img_files,label_func=label_func,...)

# The regular expression get the file_path that can be used to extract the label
dls=ImageDataLoaders.from_path_re(
	path,f_names=filenames,pat=pattern,...)

# Use dataframe cols to get image locns and corress labels
dls=ImageDataLoaders.from_df(
	df,path='.',fn_col=0,label_col=1,folder='path-where-imgs are',
	label_block={{DataBlock type if need to}},valid_col=None,
	valid_pct=0.2,seed=42
)

# Create Learner
learn=vision_learner(dls,model,metrics=[],loss_fn,opt)
learn.model.summary() # TF like summary of model arch
```

## 02-Midlevel API (More Flexibility and intuitive):

```python
from fastai.vision.all import *

# Download Dataset
path=untar_data(url="ds-link",path="where-to-store",archive="where-to-extract")

def get_items(src_path):
	# return filenames or Tuple[[items],[targets]]

splits=RandomSplitter() # -> GrandParentSplitter() | IndexSplitter() | ColSplitter etc...
dblock=DataBlock(
	(ImageDataBLock,CategoryBlock), n_inp=1, # how many block corresponds to input
	get_items=get_items # how to get x and y,it can be filename,df etc..
	get_x=get_image_files, # how to get x from get_items
	get_y=RegexLabeller(pat), # how to get y from get_items
	spliiter=splits # for training , vaidation and test
	tfms=[(PILImage.convert),(Categorize)] # [(x to images),(y to labels)], type_tfms
)

ds=dblock.datasets(path)
dls=dblock.dataloaders(path,bs,val_bs,after_items=[item_tfms],after_batch=[batch_tfms])

# Create learner
learn=Learner(model,dls,loss_fn,metrics,opt)
```

## 03-Low Level API (Direct Pytorch, Max Flexibility):

```python
# Implicit imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.data.utils import Dataset,DataLoader
from torchvision.models.resnet import resnet34

# helper utils
from functools import partial,reduce,compress

# fastai imports
from fastai.data.externals import untar_data,URLs 
from fastai.data.core import Datasets,DataLoaders,show_at
from pathlib import Path
from fastai.xtras import Path # @patched Pathlib.Path,for path.ls() etc..
from fastai.optim import OptimWrapper

from fastai.metrics import accuracy,accuracy_multi
from fastai.losses import BCEWithLogitLossFlat

from fastai.vision.augment import aug_transforms
from fastai.data.transforms import (
	ColReader,
	IntToFloatTensor,
	MultiCategorize,
	Normalize,
	OneHotEncode,
	RandomSplitter
)

from fastai.vision.core import PILImage,PILImageBW
from fastai.vision.learner import vision_learner
from fastai.learner import Learner
from fastai.callback.schedule import Learner # @pathched Learner for lr_find,fit_one_cycle


# ================ Creating fastai DataLoaders ================
# METHOD-1: Create torch Dataset -> torch DataLoader -> fastai DataLoaders
class PlanetDataset(Dataset,transforms=None):
	def __init__(self,transforms=None):
		# Download dataset
		self.src=untar_data(URLs.PLANET_SAMPLE) # returns a path obj
		self.df=pd.read_csv(self.src/"labels.csv")
		self.transforms=transforms
		
	def __getitem__(self,idx):
		img=PILImage.create(df.iloc[idx,0])
		if self.transforms:
			img=self.transforms(img)
		labels=df.iloc[idx,-1].str.split(' ')
		
		return (img,labels)

	def __len__(self):
		return df.shape[0]

ds=PlanetDataset(tfms)
train_size,valid_size=int(0.8*len(ds)),len(ds)-train_size
train_ds,valid_ds=random_split(ds,(train_size,valid_size))

train_dl=DataLoader(
	train_ds,batch_size=64,
	shuffle=True,pin_memory=True,num_workers=2)
	
valid_dl=DataLoader(
	valid_ds,batch_size=128,
	pin_memory=True,num_workers=2)

dls=DataLoaders(train_dl,valid_dl) # fastai's DataLoaders cls


# METHOD-2: Create dataset using Fastai's Datasets -> fastai DataLoader
src=untar_data(URLS.PLANET_SAMPLE)
df=read_csv(src/'labels.csv')

def sep_labels(labels:str):
	return labels.split(' ')

ds=Datasets(
	get_items=df,
	get_x=ColReader('filename',pref=src/'images',suff='.jpg'),get_y=ColReader('labels'),
	tfms=([PILImage.create],[sep_labels,Categorize(vocab),OneHotEncoder()]),
	splitter=RandomSplitter(valid_pct=0.2,seed=42)
)

dls=ds.dataloaders(
	ds,bs=64,val_bs=128,shuffle=True,
	after_item=[CropPad(240),Normalize(),ToTensor()], # item_tfms
	after_batch=[IntToFloatTensor(),*aug_tranform(size=224)] # batch_tfms
)


# METHOD-3: Create Fastai DataBlock -> fastai DataLoaders
src=untar_data(URLS.PLANET_SAMPLE)
df=pd.read_csv(src/'labels.csv')

dblock=DataBlock(
	blocks=(ImageBlock,MultiCategoryBlock),n_inp=1
	get_items=df,
	get_x=ColReader(0,pref=src/'images',suff='.jpg') # './images/{filename}/.jpg'
	get_y=PipeLine([ColReader("labels"),sep_labels,Categorize(vocab),OneHotEncode()])
	tfms=[(),()] # type transforms
	spliiter=RandomSplitter(valid_pct=0.2,seed=42)
)

dls=dblocks.dataloaders(
	bs=64,val_bs=128,shuffle=True,
	after_item=[CropPad(240),Normalize(),ToTensor()], # item_tfms
	after_batch=[IntToFloatTensor(),*aug_tranform(size=224)] # batch_tfms
)

# Creating fastai's optimizer using pytorch's optim
Adam=torch.optim.AdamW
opt=partial(OptimWrapper,opt=Adam)

#================ Creating Learner ================ 
# METHOD-1: Using Learner API: use torch model -> freeze model -> attach new head 
resnet=resnet34(pretrained=True)
# Freeze backbone
for p in model.parameters():
	p.require_grad=False
# attact new head
resnet.fc=nn.Linear(in_channels,num_labels) # new-head
learn=Learner(
	resnet,dls,metrics=partial(accuracy_multi,thres=0.5),
	loss_func=BCEWithLogitLossFlat(),opt_func=opt
)

# METHOD-2: Using vision_learner -> fastai will customizes the model automatically
learn=vision_learn(
	dls,resnet34(pretrained=True),metrics=partial(accuracy_multi,thres=0.5),
	loss_func=BCEWithLogitLossFlat(),opt_func=opt
)

learn.loss_func.threshold=0.5

# Training
learner.lr_find()
learner.fit(2,lr=slice(2e-3)) # train head

# FineTune
learn.unfreeze() 
learn.fine_tune(5,lr=slice(4e-4,3e-5))

# One cycle policy
learn.fit_one_cycle(2,lr_max=2e-3)

```
