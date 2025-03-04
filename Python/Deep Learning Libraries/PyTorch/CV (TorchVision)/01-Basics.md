## **Transforming and Augmenting Images:**

Torchvision supports common computer vision transformations in the `torchvision.transforms` and `torchvision.transforms.v2` modules.

```python
from torchvision.transforms import v2 # jit() compatible transforms
from torchvision.io import read_image # read img -> torch.tensor

img=read_image("path")

# Compose-> To apply multiple transforms
tfms=v2.Compose([
	v2.Resize(size=(224,224)),
	v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True), 
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # and many more ...
])

img=tfms(img)
```

## **Detection, Segmentation, Videos:**

We can use Torchvision to create bounding-boxes, segmentation-mask and even working with videos is made simple using a single API `torchvision.tv_tensors`

TVTensors are [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.4)") subclasses. The available TVTensors are [`Image`](https://pytorch.org/vision/stable/generated/torchvision.tv_tensors.Image.html#torchvision.tv_tensors.Image "torchvision.tv_tensors.Image"), [`BoundingBoxes`](https://pytorch.org/vision/stable/generated/torchvision.tv_tensors.BoundingBoxes.html#torchvision.tv_tensors.BoundingBoxes "torchvision.tv_tensors.BoundingBoxes"), [`Mask`](https://pytorch.org/vision/stable/generated/torchvision.tv_tensors.Mask.html#torchvision.tv_tensors.Mask "torchvision.tv_tensors.Mask"), and [`Video`](https://pytorch.org/vision/stable/generated/torchvision.tv_tensors.Video.html#torchvision.tv_tensors.Video "torchvision.tv_tensors.Video").Everything that is supported on a plain [`torch.Tensor`](https://pytorch.org/docs/stable/tensors.html#torch.Tensor "(in PyTorch v2.4)") like `.sum()` or any `torch.*` operator will also work on a TVTensors.

```python
form torchvision import tv_tensors # For everything

# Working with

# 1. Image: torch.Tensor subclass for Image with shape (...,C,H,W)
img=tv_tensors.Image(
		data=(tensor_like,PIL.Image.Image),dtype=Optional[dtype],
		device=Optional[Union([device,str,int]],requires_grad=Optional[bool]
	)

# 2. Bounding Box : Object-Detection
bounding_box=tv_tensors.BoundingBoxes(
		data: Any, *,format: Union[BoundingBoxFormat, str:'XYXY | CXCYWH | XYWH'],
		canvas_size: Tuple[int, int], requires_grad: Optional[bool] = None,
		dtype: Optional[dtype] = None, 
		device: Optional[Union[device, str, int]] = None
	)

# Mask : Image Segmentation
mask=tv_tensors.Mask(
		data: Any, *, dtype: Optional[dtype] = None, 
		device: Optional[Union[device, str, int]] = None,
		requires_grad: Optional[bool] = None
	)

# Video : torch.Tensor subclass for videos with shape [..., T, C, H, W].
video=tv_tensor.Video(
		data: Any, *, dtype: Optional[dtype] = None, 
		device: Optional[Union[device, str, int]] = None, 
		requires_grad: Optional[bool] = None
	)

# U can apply the transforms like this
out_img,out_boxes=tfms(out_img,out_boxes)
# Prefered way as for targets we might need to pass multiple things
out_img,targets=tfms(img,{
	"boxes":boxes,
	"mask":mask,
	"cls_ids":labels,
	"idx":unique_identifier
})
```

# **Dataset compatibility:**
An easy way to force those datasets to return TVTensors and to make them compatible with v2 transforms is to use the [`torchvision.datasets.wrap_dataset_for_transforms_v2()`](https://pytorch.org/vision/stable/generated/torchvision.datasets.wrap_dataset_for_transforms_v2.html#torchvision.datasets.wrap_dataset_for_transforms_v2 "torchvision.datasets.wrap_dataset_for_transforms_v2") function

```python
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

dataset = CocoDetection(..., transforms=my_transforms)
dataset = wrap_dataset_for_transforms_v2(dataset)
# Now the dataset returns TVTensors!
```

## **Some Important Reads:**
* `Read About transformations here:`  [Illustration of transforms](https://pytorch.org/vision/stable/auto_examples/transforms/plot_transforms_illustrations.html#illustration-of-transforms)
* `V2 API refrences:` [Recommended](https://pytorch.org/vision/stable/transforms.html#v2-api-reference-recommended)
* `Auto Augmentation:` is a common Data Augmentation technique that can improve the accuracy of Image Classification models. Though the data augmentation policies are directly linked to their trained dataset, empirical studies show that ImageNet policies provide significant improvements when applied to other datasets. [Auto Augmentation](https://pytorch.org/vision/stable/transforms.html#auto-augmentation) 