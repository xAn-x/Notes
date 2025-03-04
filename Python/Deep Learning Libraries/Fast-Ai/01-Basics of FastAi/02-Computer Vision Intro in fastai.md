
```python
from fastai.vision.all import *
```

## 1. Single-label Classification:

### 1.Single Class Classification

We will use the example [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) that contains images of cats and dogs of 37 different breeds.

```python
from fastai.vision.all import *

path=untar_data(URLs.PETS) # dwnld and decompress the data frm url
path.ls() # list all files in the path

files=get_image_files(path/"images") # get all img files frm path

# func to distinguish labels
def label_func(x): return x[0].is_upper()


# Create DataLoader
# if using a label_func: ImageDataLoader.from_name_func(func)
# if using a regex: ImageDataLoader.from_name_re(regex)
# and many_more...
dls=ImageDataLoader.from_name_func(path,files,label_func=label_func,
								  item_tfms=Resize(224,224),bs=16)
dls.show_batch() # show a batch of / help in viz



# Create a learner
learner=vision_learner(dls,resnet34,metrics=[error_rate])
learner.fine_tune(2)

# making Preds
label,idx_of_pred_cls,prob_all_cls=learner.predict("new_image_path")

learner.show_results() # randomly choose some images and tell actual and predicted value
```

### 2. Multiclass Classification:
We will use the example [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/) that contains images of cats and dogs of 37 different breeds.

```cmd
'great_pyrenees_173.jpg' -> name of file
```

so the class is everything before the last `_` followed by some digits

```python
pattern=r'^(.*)_\d+.jpg' # extract everything before last _
```

```python
from fastai.vision.all import *

# Create DataLoaders
path=untar(URLs.PETS)
extract_breed=r'^(.*)_\d+.jpg'
dls=ImageDataLoader.from_name_re(path,get_image_files(path/"images")
							label_func=extract_breed,bs=32,
							item_tfms=Resize(460),
							batch_tfms=aug_transformers(size=224))
# batch tfms: apply transformation in whole batch and also do data augumentation
# batch tfms: are applied on GPU and thus faster than item_tfms
dls.show_batch(n_max=8,figsize=(8,8))


# Create a learner
learner=vision_learner(dls,resnet34,metrics=error_rate)
learner.lr_find() # help in finding good val for learning rate
learner.fine_tune(2,lr=3e-3) # 1 epoch for newly added head and then 2 epochs for wholde model with passed lr
learner.show_results() # randomly choose a batch from valid_dls and display over it

# Another thing that is useful is an interpretation object, it can show us where the model made the worse predictions:
intrep=Interpretation.from_learner(learner)
# or
intrep=ClassificationInterpretation.from_learner(learner)

intrep.plot_top_losses(k=10,figsize=(10,8))
```

 [`aug_transforms`](https://docs.fast.ai/vision.augment.html#aug_transforms) is a function that provides a collection of data augmentation transforms with defaults we found that perform well on many datasets. You can customize these transforms by passing appropriate arguments to [`aug_transforms`](https://docs.fast.ai/vision.augment.html#aug_transforms)

## 2. Multi-Label Classification:

For this task, we will use the [Pascal Dataset](http://host.robots.ox.ac.uk/pascal/VOC/) that contains images with different kinds of objects/persons. It’s orginally a dataset for object detection, meaning the task is not only to detect if there is an instance of one class of an image, but to also draw a bounding box around it. Here we will just try to predict all the classes in one given image.

Defers from before in the sense each image does not belong to one category

```python
from fastai.vision.all import *
import pandas as pd

# Create DataLoaders
path=untar_data(URLS.PASCAL_2007) # download data
df=pd.read_csv(path/"train.csv")
df.head() # all the labels are seprated from ' '
```

|     | fname      | labels       | is_valid<br> |
| --- | ---------- | ------------ | ------------ |
| 0   | 000005.jpg | chair        | True         |
| 1   | 000007.jpg | car          | True         |
| 2   | 000009.jpg | horse person | True         |
| 3   | 000012.jpg | car          | False        |
| 4   | 000016.jpg | bicycle      | True         |
|     |            |              |              |
```python
# Create DataLoaders
dls=ImageDataloader.from_df(df,path,folder="train",valid="is_valid",
						   label_delim=' ',item_tfms=Resize(460),
						   batch_tfms=aug_tranforms(size=224))

dls.show_batch(max_n=8)
```

Training a model is as easy as before: the same functions can be applied and ==the fastai library will automatically detect that we are in a multi-label problem==, thus picking the right loss function. The only difference is in the metric we pass: [`error_rate`](https://docs.fast.ai/metrics.html#error_rate) will not work for a multi-label problem, but we can use `accuracy_thresh` and [`F1ScoreMulti`](https://docs.fast.ai/metrics.html#f1scoremulti). We can also change the default name for a metric, for instance, we may want to see F1 scores with `macro` and `samples` averaging.

```python
f1_macro = F1ScoreMulti(thresh=0.5, average='macro')
f1_macro.name = 'F1(macro)' 
f1_samples = F1ScoreMulti(thresh=0.5, average='samples')
f1_samples.name = 'F1(samples)'

# Create Learner
learner = vision_learner(dls, resnet50, metrics=[partial(accuracy_multi, thresh=0.5), f1_macro, f1_samples])

learner.lr_find()
learner.fit(2,2e-3)

learner.show_results()
learner.predict("new_file_path")

# Making Interpretations
intrep=ClassificationInterpretation.from_learner(learner)
intrep.plot_top_losses(k=10)
```

## 3. Segmentation

*Image Segmentation:*
Image segmentation is a computer vision task that involves dividing an image into multiple regions or segments that represent different objects, textures, or other meaningful features. The goal is to extract meaningful information from the image by grouping pixels that belong to the same object or region.

*Loss Functions for Image Segmentation:*
In image segmentation, we use loss functions to measure the difference between the predicted segmentation mask and the ground truth mask. Common loss functions include:

* **Cross-Entropy Loss:** This loss function measures the difference between the predicted probabilities and the ground truth labels. It is often used for binary segmentation tasks.
* **Dice Loss:** This loss function measures the overlap between the predicted and ground truth segmentation masks. It is commonly used for multi-class segmentation tasks.
* **Jaccard Loss (Intersection-over-Union Loss):** This loss function measures the ratio of the intersection between the predicted and ground truth masks to their union. It is similar to the Dice loss but penalizes more heavily for false positives.
* **Focal Loss:** This loss function down-weights easy-to-classify pixels and focuses on hard-to-classify pixels, making it more robust to class imbalance.

***Choice of Loss Function:***
The choice of loss function depends on the specific segmentation task, the data distribution, and the desired performance metrics. For example, cross-entropy loss is suitable for binary segmentation tasks, while Dice loss or Jaccard loss are often preferred for multi-class segmentation tasks. Focal loss is particularly useful when there is a significant class imbalance in the dataset.

#### MODELLING
Segmentation is a problem where we have to predict a category for each pixel of the image. For this task, we will use the [Camvid dataset](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/), a dataset of screenshots from cameras in cars. Each pixel of the image has a label such as “road”, “car” or “pedestrian”.

A traditional CNN won’t work for segmentation, we have to use a special kind of model called a UNet, so we use [`unet_learner`](https://docs.fast.ai/vision.learner.html#unet_learner) to define our [`Learner`](https://docs.fast.ai/learner.html#learner)

```python
from fastai.vision.all import *
path=untar_data(URLs.CAMVID_TINY)
```

The `images` folder contains the images, and the corresponding segmentation masks of labels are in the `labels` folder. The `codes` file contains the corresponding integer to class (the masks have an int value for each pixel).

```python
fnames=get_image_files(path/'images')
# the segmentation masks have the same base names as the images but with an extra `_P`, so we can define a label function

def label_func(fn): return path/"labels"/f"{fn.stem}_P{fn.suffix}"

dls=ImageDataLoader.from_name_func(path,fnames,label_func,
								  bs=32,codes=codes)

dls.show_batch(max_n=6)

# Create a segmentation learner
learner=unet_learner(dls,resnet34)
learner.fine_tune(10)
learner.show_results()

# Making Interpretation
intrep=SegmentationInterpretation.from_learner(learner)
intrep.plot_top_losses(k=4)

```