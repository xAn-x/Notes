The `torchvision.models` subpackage contains definitions of models for addressing different tasks, including: image classification, pixelwise semantic segmentation, object detection, instance segmentation, person keypoint detection, video classification, and optical flow.

```python
import torchvision 
from torchvision.models import list_models

all_models=list_models()
classification_models=list_models(module=torchvision.models)
detection_models=list_models(module=torchvision.models.detection)

# Loading models and weights
from torchvision.models import resnet50, ResNet50_Weights

# Using pretrained weights:
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet50(weights="IMAGENET1K_V1")

# Using no weights:
resnet50(weights=None)
resnet50()
```

Using the pre-trained models, one must preprocess the image (resize with right resolution/interpolation, apply inference transforms, rescale the values etc). There is no standard way to do this as it depends on how a given model was trained. It can vary across model families, variants or even weight versions. Using the correct preprocessing method is critical and failing to do so may lead to decreased accuracy or incorrect outputs.

To simplify inference, TorchVision bundles the necessary preprocessing transforms into each model weight. These are accessible via the `weight.transforms` attribute

```python
# Initialize the Weight Transforms
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

# Apply it to the input image
img_transformed = preprocess(img)
```

### **Listing and retrieving available models:**
TorchVision offers a new mechanism which allows listing and retrieving models and weights by their names. Here are a few examples on how to use them

```python
# Initialize models
m1 = get_model("mobilenet_v3_large", weights=None)
m2 = get_model("quantized_mobilenet_v3_large", weights="DEFAULT")

# Fetch weights
weights = get_weight("MobileNet_V3_Large_QuantizedWeights.DEFAULT")
assert weights == MobileNet_V3_Large_QuantizedWeights.DEFAULT

# Fetch Weghts usin Model name
weights_enum = get_model_weights("quantized_mobilenet_v3_large")
assert weights_enum == MobileNet_V3_Large_QuantizedWeights

weights_enum2 = get_model_weights(torchvision.models.quantization.mobilenet_v3_large)
assert weights_enum == weights_enum2
```

## **Using models from TorchHub:**

```python
import torch

# Option 1: passing weights param as string
model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")

# Option 2: passing weights param as enum
weights = torch.hub.load("pytorch/vision", "get_weight", weights="ResNet50_Weights.IMAGENET1K_V2")
model = torch.hub.load("pytorch/vision", "resnet50", weights=weights)

weight_enum = torch.hub.load("pytorch/vision", "get_model_weights", name="resnet50")
print([weight for weight in weight_enum])
```