## *Images*:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import os
import PIL # For Images
import PIL.Image 
import glob # for pattern match

# Getting Data from url
import pathlib
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
# download data
archive = tf.keras.utils.get_file(origin=dataset_url, extract=True)
data_dir = pathlib.Path(archive).with_suffix('')

# getting all files with extension .jpg
images=data_dir.glob('*/*.jpg')
print(len(images))

# displaying images
roses=data_dir.glob("roses/*")
PIL.Image.open(str(roses[0])) # take image path and then return a PIL image

# Creating Dataset from ImageFolder
ds=tf.keras.preprocessing.image_dataset_from_directory(
	directory=data_dir,
    labels='inferred | None ',
    label_mode='int | categorical | binary | None',
    class_names=None, # if specified,then folder names must be same as given
    color_mode='rgb | grayscale | rgba',
    batch_size=32,
    image_size=(256, 256),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation='bilinear',
    follow_links=False,
    crop_to_aspect_ratio=False,
    pad_to_aspect_ratio=False,
    verbose=True,
    data_format=None # If None uses keras.config.image_data_format() otherwise either 'channel_last' or 'channel_first'.
)

# Data Augmentation
# 1. Using keras preprocessing layers -> can directly insert it in model
tranformation=keras.Sequential([
	keras.layers.RandomFlip("horizontal_and_vertical"),
	keras.layers.RandomRotation(0.2),
	keras.layers.RandomZoom()
])

# Using tf.image -> use ds.map() to apply
def transform(images,labels):
	image=tf.image.randomBrightness(image)
	image=tf.image.rot90(image)
	image=tf.image.centerCrop(image,central_fraction=0.5)
	image=tf.image.flip_left_to_right(image)
	return image,label

ds=ds.batch(32).shuffle().map(transform)

# -- Model-training --
```

## *Text:*

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Getting Data
import pathlib
archive=keras.utils.get_file(origin=url,extract=True,cache_dir='dir-name')
data_dir=pathlib.Path(archive).parent

print(list(data_dir.iterdir())) # iterate inside the directory to give info about files
print(list(data_dir'/train'.iterdir()))
```

This is train-directory structure

```
train/
...csharp/
......1.txt
......2.txt
...java/
......1.txt
......2.txt
...javascript/
......1.txt
......2.txt
...python/
......1.txt
......2.txt
```

```python
raw_train_ds=keras.preprocessing.text_dataset_from_directory(
	dir=train_dir
	labels='inferred | None',
    label_mode='int | Categorical | Binary | None',
    class_names=None,
    batch_size=32,
    max_length=None,
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    follow_links=False, 
    verbose=True
)

for text_batch,label_batch in raw_train_ds.take(1):
	text_batch,label_batch=text_batch.numpy(),label_batch.numpy()
	for i in range(10):
		print("Question: ", text_batch[i])
	    print("Label:", label_batch[i])

for i, label in enumerate(raw_train_ds.class_names):
  print("Label", i, "corresponds to", label)


# Processing Text dataset
# 1.Vectorization
VOCAB_SIZE = 10000
vectorization_layer=keras.layers.Vectorization(
	max_tokens=VOCAB_SIZE,
    standardize='lower_and_strip_punctuation | lower | strip_punchuation | None',
    split='whitespace',
    ngrams=None, #will create ngrams up to that integer, and passing a tuple of integers will create ngrams for the specified values in the tuple
    output_mode='int | multihot | count | tf_idf',
    output_sequence_length=None, # for padding and truncation -> result to (bs,seq_len)
    pad_to_max_tokens=False,
    vocabulary=None,
    idf_weights=None,
    sparse=False,
    ragged=False,
    encoding='utf-8',
    name=None,
    **kwargs
)

# Adapting to some text
text_only_ds=raw_train_ds.map(lambda text,label:text)
vectorization_layer.adapt(text_only_ds) # Use this text to create vocab, will cause the model to build an index of strings to integers

for text_batch,label_batch in raw_train_ds.take(1):
	first_question=text_batch[0].numpy()
	first_label=label_batch[0].numpy()
	
	print("Question:", first_question) 
	print("Label:", first_label)
```

```
>> Question: tf.Tensor(b'"unit testing of setters and getters teacher wanted us to do a comprehensive unit test. for me, this will be the first time that i use junit. i am confused about testing set and get methods. do you think should i test them? if the answer is yes; is this code enough for testing?..  public void testsetandget(){.    int a = 10;.    class firstclass = new class();.    firstclass.setvalue(10);.    int value = firstclass.getvalue();.    assert.asserttrue(""error"", value==a);.  }...in my code, i think if there is an error, we can\'t know that the error is deriving because of setter or getter."\n', shape=(), dtype=string)

>> Label: tf.Tensor(1, shape=(), dtype=int32)
```

```python
print("vectorized question:",
      vectorize_layer(first_question).numpy())
```

>> vectorized question: [1011  773    9 2456    8 1863 2362  690 1267    4  40    5    1 1011 196   12   74   13   72   33    2   98  105   14    3   70 9611    3 34  888  202  773  107    8   41  242   40   58  291   90    3  196 191   10    2  182    6  668    6   13   30 1187   12  773 22   42 1   28    5  140   29 5213   15   29    1   28   51    1    1   1 7   23   30    3  291   10   67    6   32   65  185  166  102   14 2  65    6    1  193    9 2784    45  2410   0    0    0    0    0 0    0 0    0    0    0    0    0    0    0    0    0    0    0 0    0    0   0    0    0    0    0    0    0    0    0    0    0 0    0    0    0   0    0    0    0    0    0    0    0    0    0 0    0    0    0    0   0    0    0    0    0    0    0    0    0 0    0    0    0    0    0    0    0    0    0    0    0    0    0 0    0    0    0    0    0    0   0    0    0    0    0    0    0 0    0    0    0    0    0    0    0   0    0    0    0    0    0 0    0    0    0    0    0    0    0    0   0    0    0    0    0 0    0    0    0    0    0    0    0    0    0   0    0    0    0 0    0    0    0    0    0    0    0    0    0    0   0    0    0 0    0    0    0    0    0    0    0    0    0    0    0]

```python
vocab=vectorizer_layer.get_vocabulary() # int->str mapping
vectorizer_layer.set_vocab(mapping)
vectorizer_layer.finalize_state() # can't make no further change
vectorizer_layer.vocabulary_size()

# -- Train Model --
```