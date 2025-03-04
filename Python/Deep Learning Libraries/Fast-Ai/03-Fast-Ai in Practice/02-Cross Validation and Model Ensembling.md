
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from fastai.vision.all import *
from sklearn.model_selection import StratifiedKFold

src=untar_data(URLs.PETS)
files=get_image_files(src/'images')
train_files,test_files=files[:int(0.9*len(files))],files[int(0.9*len(files)):]


skf=StratifiedKFold(n_splits=10,shuffle=True)
temp=np.zeros(len(train_files))
for split in skf.split(temp):
	for train_idx,val_idx in split:
		dblocks=DataBlock(
			blocks=(ImageBlock,MultiCategoryBlock),n_inp=1,
			get_items=train_files,
			get_y=Pipeline([RegexLabeller(pat=r"(.+)_\f+.jpg$"),Categorize()]),
			item_tfms=[Resize(240),CenterCrop(224)],
			batch_tfms=[IntToFloatTensor(),*aug_tfms(size=224,wrap=0),Normalize()],
			bs=64,val_bs=128,
			splitter=IndexSplitter(val_idx) # use these idx to create validation set
		)
		dls=dblocks.dataloaders(src)
		learn=vision_learner(dls,resnet34,metrics=[accuracy])
		learn.fit_one_cycle(1,max_lr=2e-3)
		valid_pcts.append(learn.validate()[1]) # use validation set and get accuracy
```