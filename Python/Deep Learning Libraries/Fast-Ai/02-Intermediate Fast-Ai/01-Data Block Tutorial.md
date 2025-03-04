The data block API takes its name from the way it’s designed: every bit needed to build the [`DataLoaders`](https://docs.fast.ai/data.core.html#dataloaders) object (type of inputs, targets, how to label, split…) is encapsulated in a block, and you can mix and match those blocks

By itself, a [`DataBlock`](https://docs.fast.ai/data.block.html#datablock) is just a blue print on how to assemble your data. It does not do anything until you pass it a source. You can choose to then convert that source into a [`Datasets`](https://docs.fast.ai/data.core.html#datasets) or a [`DataLoaders`](https://docs.fast.ai/data.core.html#dataloaders) by using the [`DataBlock.datasets`](https://docs.fast.ai/data.block.html#datablock.datasets) or [`DataBlock.dataloaders`](https://docs.fast.ai/data.block.html#datablock.dataloaders) method. 

## Building a [`DataBlock`](https://docs.fast.ai/data.block.html#datablock) from scratch

Let’s first build a [`DataBlock`](https://docs.fast.ai/data.block.html#datablock) from scratch on the dogs versus cats problem we saw in the [vision tutorial](https://docs.fast.ai/tutorial.vision.html).

```python
from fastai.data.all import *
from fastai.vision.all import *

path=untar_data(URLs.PETS)
fnames=get_image_files(path/"images")

# directly provide the source files
dblocks=DataBlock() # empty data-block
ds=dblocks.datasets(fnames)

# --- or ---

# tell how and then path
dblock=DataBlock(get_items=get_image_files)
ds=dblocks.datasets(paths/"images")


def label_func(fname):
	return "cat" if fname[0].isupper() else "dog"

# How to get input and outputs
dblock=DataBlock(get_items=get_image_files,get_y=label_func)
ds=dblock.datasets(path/'images')


# Now that our inputs and targets are ready, we can specify types to tell the data block API that our inputs are images and our targets are categories.
dblock=DataBlock(blocks=(ImageBlock,CategoryBlock),
				get_items=get_image_files,get_y=label_func)
ds=dblock.datasets(path/"images")
ds.train[0] # >> (PILImage mode=RGB size=361x500, TensorCategory(1))
```

[`DataBlock`](https://docs.fast.ai/data.block.html#datablock) automatically added the transforms necessary to open the image or how it changed the name “dog” to an index 1 (with a special tensor type TensorCategory(1)). To do this, it created a mapping from categories to the index called “vocab” that we can access this way:

```python
ds.vocab
```

```cmd
['cat','dog']
```

*Note* that you can mix and match any block for input and targets, which is why the API is named data block API. You can also have more than two blocks (if you have multiple inputs and/or targets), you would just need to pass `n_inp` to the [`DataBlock`](https://docs.fast.ai/data.block.html#datablock) to tell the library how many inputs there are (the rest would be targets) and pass a list of functions to `get_x` and/or `get_y` (to explain how to process each item to be ready for his type).

The next step is to control how our validation set is created. We do this by passing a `splitter` to [`DataBlock`](https://docs.fast.ai/data.block.html#datablock).

```python
dblock=DataBlock(blocks=(ImageBlock,CategoryBlock),n_inp=1,
				get_items=get_image_files,get_y=label_func,
				splitter=RandomSplitter,item_tfms=Resize(224))

dl=dblock.dataloaders(path/"images")
dl.show_batch()
```

### DataBlock Creation steps:
The way we usually build the data block in one go is by answering a list of questions:

- what is the types of your inputs/targets? Here images and categories
- where is your data? Here in filenames in subfolders
- does something need to be applied to inputs? Here no
- does something need to be applied to the target? Here the `label_func` function
- how to split the data? Here randomly
- do we need to apply something on formed items? Here a resize
- do we need to apply something on formed batches? Here no

## MNIST image classification using DataBlocks

```python
from fastai.data.all import *
from fastai.vision.all import *

mnist=DataBlock(blocks=(ImageBlock,CategoryBlock),
				get_items=get_image_files,get_y=parent_label,
				splitter=GrandParentSplitter(train_name="train",
											valid_name="valid")
				)
# GranParentSplitter: uses the the folder names in the path to deter the training and validation egs.

# RandomSplitter(valid_pct): randomly chooses images to put in training and validation set

# IndexSplitter(): Uses indexs to determine which index goes to which part 

# many more....

dls=mnist.dataloaders(untar_data(URLS.MNIST_TINY))
dls.show_batch()

# -- or --

dls=mnist.summary(untar_data(URLS.MNIST_TINY)) # to get detail over-view of whats happening under the hood

# create learner and fine_tune in train
```

![[Pasted image 20240625164355.png]]

### PETS classification

The split training/validation is done by using a [`RandomSplitter`](https://docs.fast.ai/data.transforms.html#randomsplitter). The function to get our targets (often called `y`) is a composition of two transforms: we get the name attribute of our `Path` filenames, then apply a regular expression to get the class. To compose those two transforms into one, we use a `Pipeline`.

Finally, We apply a resize at the item level and `aug_transforms()` at the batch level.

```python
pets = DataBlock(blocks=(ImageBlock,CategoryBlock),
				get_items=get_image_files,
				get_y=Pipeline([attrgetter("name"),
				RegexLabeller(pat=r'^(.*)_\d+.jpg$')]),
				item_tfms=Resize(224),
				batch_tfms=aug_transforms(),
				splitter=RandomSplitter())
```

### Pascal (multi-label)

```python
pascal_source=untar_data(URLs.PASCAL_2007)
df=pd.read_csv(pascal_source/"train.csv")

dblock=DataBlock(blocks=(ImageBlock,MultiCategoryBlock),
				get_x=ColReader(0,pref=pascal_source/"train"),
				get_y=ColReader(1,label_delim=' '),
				splitter=RandomSplitter(),
				item_tfms=Resize(224),
				batch_tfms=aug_transforms())
```

Notice how there is one more question compared to before: we wont have to use a `get_items` function here because we already have all our data in one place.But we will need to do something to the raw dataframe to get our inputs, read the first column and add the proper folder before the filename. This is what we pass as `get_x`.

The function to get our inputs (often called `x`) is a [`ColReader`](https://docs.fast.ai/data.transforms.html#colreader) 
on the first column with a prefix, the function to get our targets (often called `y`) is [`ColReader`](https://docs.fast.ai/data.transforms.html#colreader) on the second column, with a space delimiter.

Another way to do this is by directly using functions for `get_x` 
and `get_y`:

```python
pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter(),
                   get_x=lambda x:pascal_source/"train"/f'{x[0]}',
                   get_y=lambda x:x[1].split(' '),
                   item_tfms=Resize(224),
                   batch_tfms=aug_transforms())


# --- or ---

pascal = DataBlock(blocks=(ImageBlock, MultiCategoryBlock),
                   splitter=ColSplitter(),
                   get_x=lambda o:f'{pascal_source}/train/'+o.fname,
                   get_y=lambda o:o.labels.split(),
                   item_tfms=Resize(224),
                   batch_tfms=aug_transforms())


dls = pascal.dataloaders(df)
dls.show_batch()
```

![[Pasted image 20240625183850.png]]

## Image localization[🔗](https://docs.fast.ai/tutorial.datablock.html#image-localization)

There are various problems that fall in the image localization category: image segmentation (which is a task where you have to predict the class of each pixel of an image), coordinate predictions (predict one or several key points on an image) and object detection (draw a box around objects to detect).

### Segmentation:
We will use a small subset of the [CamVid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/) for our example.

```python
path=untar_data(URLs.CAMVID_TINY)

camvid=DataBlock(blocks=(ImageBlock,SegmentationBlock),
				get_items=get_image_files,
				get_y=lambda o:path/'labels'/f'{o.stem}_P{o.suffix}',
				batch_tfms=aug_transforms())	
```

### Text Generation (Lang Modelling):
Since there are no targets here, we only have one block to specify. [`TextBlock`](https://docs.fast.ai/text.data.html#textblock)s are a bit special compared to other [`TransformBlock`](https://docs.fast.ai/data.block.html#transformblock)s: to be able to efficiently tokenize all texts during setup, you need to use the class methods `from_folder` or `from_df`.

```python
imdb_lm=DataBlock(blocks=TextBlock.from_df('text'),is_lm=True,
				 get_x=ColReader('text'),splitter=ColSplitter())

imdb_lm_dls=DataBlock.datalaoders(df,bs=64,seq_len=72)
```

*Note*: the `TestBlock` tokenization process puts tokenized inputs into a column called `text`. The [`ColReader`](https://docs.fast.ai/data.transforms.html#colreader) for `get_x` will always reference `text`, even if the original text inputs were in a column with another name in the dataframe

### Text classification

```python
imdb_clas = DataBlock(blocks=(TextBlock.from_df('text', seq_len=72, 
												vocab=dls.vocab), 
												CategoryBlock),
		                      get_x=ColReader('text'),
		                      get_y=ColReader('label'),
		                      splitter=ColSplitter())
		                      
# ColSplitter(): default to 'is_valid' col to determine the on which part a sentence have to put 
```