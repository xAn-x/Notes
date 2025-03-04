## 1. Loading Model:

```python
# load model from huggingface or torchvision,torchtext,torchaudio or torchhub models
from torchvision.models import resNet50,ResNet50_Weights

# ways of using pretrained weights
resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
resnet50(weights="IMAGENET1K_V1")

# Using no-weights/random-init
resnet50(weights=None)
resnet50()
```

## 2. Pre-processing / Applying transforms

There is no standard way to do this as it depends on how a given model was trained. It can vary across model families, variants or even weight versions. Using the correct pre-processing method is critical and failing to do so may lead to decreased accuracy or incorrect outputs.

```python
weights=ResNEt50_Weights.DEFAULT
preprocess=weights.transforms()

# apply transform
transformed_img=preprocess(img)
```

### 3. Listing and retrieving available models[](https://pytorch.org/vision/stable/models.html#listing-and-retrieving-available-models)

As of v0.14, TorchVision offers a new mechanism which allows listing and retrieving models and weights by their names. Here are a few examples on how to use them:

```python
from torchvision import list_models,get_model,get_model_weights
# List available models
all_models = list_models()
classification_models = list_models(module=torchvision.models)

# Initialize models
m1 = get_model("mobilenet_v3_large", weights=None)
m2 = get_model("quantized_mobilenet_v3_large", weights="DEFAULT")

# Fetch weights
weights = get_weight("MobileNet_V3_Large_QuantizedWeights.DEFAULT")
assert weights == MobileNet_V3_Large_QuantizedWeights.DEFAULT

weights_enum = get_model_weights("quantized_mobilenet_v3_large")
assert weights_enum == MobileNet_V3_Large_QuantizedWeights

weights_enum2 = get_model_weights(torchvision.models.quantization.mobilenet_v3_large)
assert weights_enum == weights_enum2
```

## 4. Using Models from Hub

```python
import torch

# Option 1: passing weights param as string
model = torch.hub.load("pytorch/vision", "resnet50", weights="IMAGENET1K_V2")

# Option 2: passing weights param as enum
weights = torch.hub.load("pytorch/vision", "get_weight", weights="ResNet50_Weights.IMAGENET1K_V2")
model = torch.hub.load("pytorch/vision", "resnet50", weights=weights)
```

## 5. Finetuning 

```python
from torchvision import list_models,get_model,get_model_weights

models=list_models(torchvision.models)

resnet_weights=get_model_weights("resnet18")
resnet_model=get_model(name="resnet18",weights=resnet_weights.IMAGENET1K_V1)

resnet
```

```cmd
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
```

```python
# Freeze weights
for parameters in resnet_model.parameters():
	parameters.requires_grad=False

trainable_params=sum(torch.numel(p) for p in resnet_model.parameters() if requires_grad)
print(f"No. of trainable params:{trainable_params}") # 0

# Replace head
class ClassificationHead(nn.Module):
	def __init__(self,in_features):
		super().__init__()
		self.my_head=nn.Linear(in_features,10)
	
	def forward(self,x):
		return self.my_head(x)

resnet_model.fc=ClassificationHead(resnet.fc.in_features)

trainable_params=sum(torch.numel(p) for p in resnet_model.parameters() if requires_grad)
print(f"No. of trainable params:{trainable_params}") # 5130

for epoch in range(epochs):
	# --- training loop ---

# make preds
logits=model(input_batch)
probs=F.softmax(logits)
print(f"Prediction:{torch.argmax(probs)}")
```