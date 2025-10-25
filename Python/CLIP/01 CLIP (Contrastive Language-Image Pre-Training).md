## **Abstract Summary**:

_State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labelled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks._

The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training.  _For instance, it match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on._

## **Usage Tips** (Hugging Face)

CLIP is a multi-modal vision and language model. It can be used for image-text similarity and for zero-shot image classification. CLIP uses a ==ViT like transformer to get visual features and a causal language model to get the text features.== _Both the text and visual features are then projected to a latent space with identical dimension. The dot product between the projected image and text features is then used as a similarity score._
To feed images to the Transformer encoder, each image is split into a sequence of fixed-size non-overlapping patches, which are then linearly embedded. A [CLS] token is added to serve as representation of an entire image. The authors also add absolute position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder. The [CLIPImageProcessor](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/clip#transformers.CLIPImageProcessor) can be used to resize (or rescale) and normalize images for the model.

The [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/clip#transformers.CLIPTokenizer) is used to encode the text. The [CLIPProcessor](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/clip#transformers.CLIPProcessor)wraps [CLIPImageProcessor](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/clip#transformers.CLIPImageProcessor) and [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/clip#transformers.CLIPTokenizer) into a single instance to both encode the text and prepare the images. The following example shows how to get the image-text similarity scores using [CLIPProcessor](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/clip#transformers.CLIPProcessor) and [CLIPModel](https://huggingface.co/docs/transformers/v4.39.3/en/model_doc/clip#transformers.CLIPModel).

```python
from PIL import Image
import requests

from transformers import CLIPProcessor,CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
```

## **Usage Tips** (OpenAI)

```python
% pip install git+https://github.com/openai/CLIP.git

import clip
import torch

print(clip.available_model()) # list of all available models
model,preprocess=clip.load("model-name",device=device)


text=["This is an Image of cat","This is an Image of dog"]
img=url

# Tokenize the text
text_tokens = clip.tokenize(text).to(device)

# process image -> reshape(224,224) and center-crop(224,224)
image=preprocess(img).unsqueeze(0).to(device)

# create encodings
with torch.no_grad():
	image_features=model.encode_image(image)
	text_features=model.encode_text(text)

	logits_per_image,logits_per_text=model(image,text)
	probs=F.softmax(logits_per_image,dim=-1)
	print(probs)

# normalize
image_features/=torch.norm(image_features,dim=1,keepdim=True)
text_features/=torch.norm(text_features,dim=1,keepdim=True)

similarity= (image_features @ text_features.T) # similarity (1,2)
```

## **Fine Tune CLIP for Classification**:

_USING OPEN-AI and Pytorch_

```python
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
from clip import clip

# Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the image and text transformations
image_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
])

# Load the custom dataset
custom_dataset = ImageFolder("path/to/custom/dataset", transform=image_transform)
custom_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# Load the pre-trained CLIP model
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# Freeze the CLIP model's parameters
for param in clip_model.parameters():
    param.requires_grad = False

# Add new classification heads for both image and text
num_classes = len(custom_dataset.classes)
new_image_head = torch.nn.Linear(clip_model.visual.output_dim, num_classes)
new_text_head = torch.nn.Linear(clip_model.transformer.width, num_classes)
clip_model.visual.head = new_image_head
clip_model.text_projection = new_text_head

# Move the model to the device
clip_model = clip_model.to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    [
        {"params": clip_model.visual.head.parameters(), "lr": 1e-4},
        {"params": clip_model.text_projection.parameters(), "lr": 1e-4},
    ]
)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in custom_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Encode the images and text
        image_features = clip_model.encode_image(images).to(device)
        text_features = clip_model.encode_text(torch.tensor([
	        f"This is an {custom_dataset.classes[label]}" for label in labels
        ]).to(device))

        # Compute the logits
        image_logits = clip_model.visual.head(image_features)
        text_logits = clip_model.text_projection(text_features)
        logits = (image_logits + text_logits) / 2
		
        # Compute the loss
        loss = criterion(logits, labels)
	    

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the fine-tuned model
torch.save(clip_model.state_dict(), "fine_tuned_clip.pth")
```

_Using Hugging Face_:

```python
from transformers import CLIPModel, CLIPProcessor
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, Resize, ToTensor
import torch.nn as nn
import torch.optim as optim

# Set up the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the image and text transformations
image_transform = Compose([
    Resize((224, 224)),
    ToTensor(),
])

# Load the custom dataset
custom_dataset = ImageFolder("path/to/custom/dataset", transform=image_transform)
custom_loader = DataLoader(custom_dataset, batch_size=32, shuffle=True)

# Load the pre-trained CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Freeze the CLIP model's parameters
for param in clip_model.parameters():
    param.requires_grad = False

# Add new classification heads for both image and text
num_classes = len(custom_dataset.classes)
new_image_head = nn.Linear(clip_model.config.vision_config.projection_dim, num_classes)
new_text_head = nn.Linear(clip_model.config.text_config.projection_dim, num_classes)
clip_model.visual_projection = new_image_head
clip_model.text_projection = new_text_head

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([
    {"params": clip_model.visual_projection.parameters(), "lr": 1e-4},
    {"params": clip_model.text_projection.parameters(), "lr": 1e-4},
])

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for images, labels in custom_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Encode the images and text
        pixel_values = clip_processor(images=images,
	        return_tensors="pt").pixel_values
	        
        input_ids = clip_processor(text=[
	        f"photo of a {custom_dataset.classes[label]}" for label in labels], 
	        return_tensors="pt").input_ids
        image_features = clip_model.get_image_features(pixel_values)
        text_features = clip_model.get_text_features(input_ids)

        # Compute the logits
        image_logits = clip_model.visual_projection(image_features)
        text_logits = clip_model.text_projection(text_features)
        logits = (image_logits + text_logits) / 2 
		 
        # Compute the loss
        loss = criterion(logits, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# Save the fine-tuned model
torch.save(clip_model.state_dict(), "fine_tuned_clip.pth")
```

## **Create Custom VisionTextDualEncoder Models**:

```python
from transformers import (
	VisionTextDualEncoderModel, # For having custom vision & text encoder
	VisionTextDualEncoderProcessor, # For building Processor for dual-encoder
	AutoTokenizer, # Auto selects text-tokenizer according to chkpoint
	AutoImageProcessor, # Auto selects image-processor according to chkpoint
)

model=VisionTextDualEncoderModel.from_vision_text_pretrained("openai/clip-vit-base-patch32", "FacebookAI/roberta-base")

tokenizer=AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
image_processor=AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")

processor=VisionTextDualEncoderProcessor.from_vision_text_pretrained(image_processor,tokenizer)

# save the model and processor
model.save_pretrained("clip-roberta")
processor.save_pretrained("clip-roberta")
```