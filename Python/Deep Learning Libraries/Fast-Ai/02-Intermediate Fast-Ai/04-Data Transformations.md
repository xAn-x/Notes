Functions for getting, splitting, and labeling data, as well as generic transforms

## Get[🔗](https://docs.fast.ai/data.transforms.html#get)

```python
# unzip data at path and return a path obj
path=untar_data(path) 

# get all files +nt in locn with passed extensions and only in folders if specified
files=get_files(path,extensions=None,recurse=True,folders=None,followLinks=True)

# get all images +nt in locn and only in folders if specified
files=get_image_files(path,recurse=True,folders=None)


```

## Split[🔗](https://docs.fast.ai/data.transforms.html#split)

```python
# Randomly splits data in train and validation split
RandomSplitter(valid_pct=0.2,seed=None)

#Split items into random train and test subsets using sklearn train_test_split utility_
TrainTestSplitter(test_size=test_size,random_state=42,startify=labels)

# Split items so that `val_idx` are in the validation set and the others in the training set
IndexSplitter(valid_idx=[..])

#Create function that splits `items` between train/val with `valid_pct` at the end if `valid_last` else at the start. Useful for ordered data.
EndSplitter(valid_pct=0.2,valid_last=True)

# Split `items` from the grand parent folder names (`train_name` and `valid_name`).
GrandParentSplitter(train_name='train',valid_name='valid')

#Split `items` by result of `func` (`True` for validation, `False` for training set).
FuncSplitter(func)

# Split `items` depending on the value of `mask`(True or False).
MaskSplitter(mask:arr)

# Split `items` by providing file `fname` (contains names of valid items separated by newline).
FileSplitter(fname="valid file-names seprated by '\n'")

# _Split `items` (supposed to be a dataframe) by value in `col`_
ColSplitter(col='is_valid')(df)

# _Take randoms subsets of `splits` with `train_sz` and `valid_sz`_
RandomSubsetSplitter(train_sz,valid_sz,seed=None)
```

## Label[🔗](https://docs.fast.ai/data.transforms.html#label)

```python
# _Label `item` with the parent folder name._
parent_label(o)


# _Label `item` with regex `pat`._
# is a very flexible function since it handles any regex search of the stringified item. Pass `match=True` to use `re.match` (i.e. check only start of string), or `re.search` otherwise (default).
regexLabeller(path,math=False)


# _Read `cols` in `row` with potential `pref` and `suff`_
#`cols` can be a list of column names or a list of indices (or a mix of both). If `label_delim` is passed, the result is split using it.
ColLabeller(pre="",suff="",label_delim=None)
```