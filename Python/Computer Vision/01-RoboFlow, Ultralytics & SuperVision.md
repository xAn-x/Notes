
## Ultralytics:

This Module help u train your object detection or segmentation model using YOLO models by providing a simple and straight-forward api using which u can access and train your models.
## YOLO: A Brief History

[YOLO](https://arxiv.org/abs/1506.02640) (You Only Look Once), a popular [object detection](https://www.ultralytics.com/glossary/object-detection) and [image segmentation](https://www.ultralytics.com/glossary/image-segmentation) model, was developed by Joseph Redmon and Ali Farhadi at the University of Washington. Launched in 2015, YOLO quickly gained popularity for its high speed and accuracy.

- [YOLOv2](https://arxiv.org/abs/1612.08242), released in 2016, improved the original model by incorporating batch normalization, anchor boxes, and dimension clusters.
  
- [YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf), launched in 2018, further enhanced the model's performance using a more efficient backbone network, multiple anchors and spatial pyramid pooling.
  
- [YOLOv4](https://arxiv.org/abs/2004.10934) was released in 2020, introducing innovations like Mosaic [data augmentation](https://www.ultralytics.com/glossary/data-augmentation), a new anchor-free detection head, and a new [loss function](https://www.ultralytics.com/glossary/loss-function).
  
- [YOLOv5](https://github.com/ultralytics/yolov5) further improved the model's performance and added new features such as hyperparameter optimization, integrated experiment tracking and automatic export to popular export formats.
  
- [YOLOv6](https://github.com/meituan/YOLOv6) was open-sourced by [Meituan](https://about.meituan.com/)
   in 2022 and is in use in many of the company's autonomous delivery robots.
- [YOLOv7](https://github.com/WongKinYiu/yolov7) added additional tasks such as pose estimation on the COCO keypoints dataset.
  
- [YOLOv8](https://github.com/ultralytics/ultralytics) released in 2023 by Ultralytics. YOLOv8 introduced new features and improvements for enhanced performance, flexibility, and efficiency, supporting a full range of vision AI tasks,
  
- [YOLOv9](https://docs.ultralytics.com/models/yolov9/) introduces innovative methods like Programmable Gradient Information (PGI) and the Generalized Efficient Layer Aggregation Network (GELAN).
  
- [YOLOv10](https://docs.ultralytics.com/models/yolov10/) is created by researchers from [Tsinghua University](https://www.tsinghua.edu.cn/en/) using the [Ultralytics](https://www.ultralytics.com/) [Python package](https://pypi.org/project/ultralytics/). This version provides real-time [object detection](https://docs.ultralytics.com/tasks/detect/) advancements by introducing an End-to-End head that eliminates Non-Maximum Suppression (NMS) requirements.
  
- **[YOLO11](https://docs.ultralytics.com/models/yolo11/) 🚀 NEW**: Ultralytics' latest YOLO models delivering state-of-the-art (SOTA) performance across multiple tasks, including [detection](https://docs.ultralytics.com/tasks/detect/), [segmentation](https://docs.ultralytics.com/tasks/segment/), [pose estimation](https://docs.ultralytics.com/tasks/pose/), [tracking](https://docs.ultralytics.com/modes/track/), and [classification](https://docs.ultralytics.com/tasks/classify/), leverage capabilities across diverse AI applications and domains.


```python
from ultralytics import YOLO

# Load a pre-trained YOLO model (you can choose n, s, m, l, or x versions)
model = YOLO("yolo11n.pt")

# Start training on your custom dataset
model.train(data="path/to/dataset.yaml", epochs=100, imgsz=640)

from ultralytics import YOLO

# Real Time Object-Detection
# Start tracking objects in a video
# You can also use live video streams or webcam input
model.track(source="path/to/video.mp4")


```
## Roboflow:

Roboflow provides everything you need to label, train, and deploy computer vision solutions.

Help you by:
- Providing different computer vision datasets that u can use for your purposes.
- U can create and annotate your own dataset using roboflow and it will automatically convert it to require formats (eg: Pascal-VOC, YOLOv5, etc)
- It also help u in data-augmentation if u don't have a lot of data, and also divide it into training and validation set.
- With Roboflow [Label Assist](https://blog.roboflow.com/announcing-label-assist/), you can use previous versions of your model to annotate future versions. Label Assist uses another model to draw annotations on images for you, which means you can spend less time annotating and get a model ready for production faster than ever.
- Also provides u model that u can download and use directly using infrence api.

### **More of Roboflow:**

#### *roboflow/autodistill:*

Autodistill uses big, slower foundation models to train small, faster supervised models. Using `autodistill`, you can go from unlabeled images to inference on a custom model running at the edge with no human intervention in between.
#### 📚 Basic Concepts[](https://github.com/autodistill/autodistill#-basic-concepts)

To use `autodistill`, you input unlabeled data into a Base Model which uses an Ontology to label a Dataset that is used to train a Target Model which outputs a Distilled Model fine-tuned to perform a specific Task.

![[Pasted image 20241006004554.jpg]]

![[Pasted image 20241006004456.jpg]]

Autodistill defines several basic primitives:

- **Task** - A Task defines what a Target Model will predict. The Task for each component (Base Model, Ontology, and Target Model) of an `autodistill` pipeline must match for them to be compatible with each other. Object Detection and Instance Segmentation are currently supported through the `detection` task. `classification` support will be added soon.
- **Base Model** - A Base Model is a large foundation model that knows a lot about a lot. Base models are often multimodal and can perform many tasks. They're large, slow, and expensive. Examples of Base Models are GroundedSAM and GPT-4's upcoming multimodal variant. We use a Base Model (along with unlabeled input data and an Ontology) to create a Dataset.
- **Ontology** - an Ontology defines how your Base Model is prompted, what your Dataset will describe, and what your Target Model will predict. A simple Ontology is the `CaptionOntology` which prompts a Base Model with text captions and maps them to class names. Other Ontologies may, for instance, use a CLIP vector or example images instead of a text caption.
- **Dataset** - a Dataset is a set of auto-labeled data that can be used to train a Target Model. It is the output generated by a Base Model.
- **Target Model** - a Target Model is a supervised model that consumes a Dataset and outputs a distilled model that is ready for deployment. Target Models are usually small, fast, and fine-tuned to perform a specific task very well (but they don't generalize well beyond the information described in their Dataset). Examples of Target Models are YOLOv8 and DETR.
- **Distilled Model** - a Distilled Model is the final output of the `autodistill` process; it's a set of weights fine-tuned for your task that can be deployed to get predictions.

#### 💡 Theory and Limitations [](https://github.com/autodistill/autodistill#-theory-and-limitations)

Human labeling is one of the biggest barriers to broad adoption of computer vision. It can take thousands of hours to craft a dataset suitable for training a production model. The process of distillation for training supervised models is not new, in fact, traditional human labeling is just another form of distillation from an extremely capable Base Model (the human brain 🧠).

Foundation models know a lot about a lot, but for production we need models that know a lot about a little.

As foundation models get better and better they will increasingly be able to augment or replace humans in the labeling process. We need tools for steering, utilizing, and comparing these models. Additionally, these foundation models are big, expensive, and often gated behind private APIs. For many production use-cases, we need models that can run cheaply and in realtime at the edge

![[Pasted image 20241006004727.jpg]]

Autodistill's Base Models can already create datasets for many common use-cases (and through creative prompting and few-shotting we can expand their utility to many more), but they're not perfect yet. There's still a lot of work to do; this is just the beginning and we'd love your help testing and expanding the capabilities of the system!

```shell
pip install autodistill autodistill-grounded-sam autodistill-yolov8
```

```python
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8

# define an ontology to map class names to our GroundingDINO prompt
# the ontology dictionary has the format {caption: class}
# where caption is the prompt sent to the base model, and class is the label that will
# be saved for that caption in the generated annotations
base_model = GroundedSAM(ontology=CaptionOntology({"shipping container": "container"}))

# label all images in a folder called `context_images`
base_model.label(
  input_folder="./images",
  output_folder="./dataset"
)

# Do changes if require...

target_model = YOLOv8("yolov8n.pt")
target_model.train("./dataset/data.yaml", epochs=200)

# run inference on the new model
pred = target_model.predict("./dataset/valid/your-image.jpg", confidence=0.5)
print(pred)

# optional: upload your model to Roboflow for deployment
from roboflow import Roboflow

rf = Roboflow(api_key="API_KEY")
project = rf.workspace().project("PROJECT_ID")
project.version(DATASET_VERSION).deploy(model_type="yolov8", model_path=f"./runs/detect/train/")
```

#### *roboflow/infrence:*

Roboflow Inference is an open-source platform designed to simplify the deployment of computer vision models. It enables developers to perform object detection, classification, and instance segmentation and utilize foundation models like [CLIP](https://inference.roboflow.com/foundation/clip), [Segment Anything](https://inference.roboflow.com/foundation/sam), and [YOLO-World](https://inference.roboflow.com/foundation/yolo_world) through a Python-native package, a self-hosted inference server, or a fully [managed API](https://docs.roboflow.com/).

```shell
pip install inference
```

Use Inference SDK to run models locally with just a few lines of code. The image input can be a URL, a numpy array (BGR), or a PIL image.

```python
from inference import get_model

model = get_model(model_id="yolov8n-640")

results = model.infer("imagefile")

```

 👉🏻foundational models:

- [CLIP Embeddings](https://inference.roboflow.com/foundation/clip) - generate text and image embeddings that you can use for zero-shot classification or assessing image similarity.
    
    ```python
    from inference.models import Clip
    
    model = Clip()
    
    embeddings_text = clip.embed_text("a football match")
    embeddings_image = model.embed_image("https://media.roboflow.com/inference/soccer.jpg")
    ```
    
- [Segment Anything](https://inference.roboflow.com/foundation/sam) - segment all objects visible in the image or only those associated with selected points or boxes.
    
    ```python
    from inference.models import SegmentAnything
    
    model = SegmentAnything()
    
    result = model.segment_image("https://media.roboflow.com/inference/soccer.jpg")
    ```
    
- [YOLO-World](https://inference.roboflow.com/foundation/yolo_world) - an almost real-time zero-shot detector that enables the detection of any objects without any training.
    
    ```python
    from inference.models import YOLOWorld
    
    model = YOLOWorld(model_id="yolo_world/l")
    
    result = model.infer(
        image="https://media.roboflow.com/inference/dog.jpeg",
        text=["person", "backpack", "dog", "eye", "nose", "ear", "tongue"],
        confidence=0.03
    )
    ```

#### 📟 inference server [](https://github.com/roboflow/inference#-inference-server)

- deploy server
    
The inference server is distributed via Docker. Behind the scenes, inference will download and run the image that is appropriate for your hardware. [Here](https://inference.roboflow.com/quickstart/docker/#advanced-build-a-docker-container-from-scratch), you can learn more about the supported images.
    
```shell
inference server start
```

- run client
  Consume inference server predictions using the HTTP client available in the Inference SDK.
    
```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
api_url="http://localhost:9001",
api_key=<ROBOFLOW_API_KEY>
)
with client.use_model(model_id="soccer-players-5fuqs/1"):
predictions = client.infer("https://media.roboflow.com/inference/soccer.jpg")
    ```
    
`If you're using the hosted API, change the local API URL to 'https://detect.roboflow.com'. Accessing the hosted inference server and/or using any of the fine-tuned models require a 'ROBOFLOW_API_KEY'.`

## 🎥 inference pipeline[](https://github.com/roboflow/inference#-inference-pipeline)

The inference pipeline is an efficient method for processing static video files and streams. Select a model, define the video source, and set a callback action. You can choose from predefined callbacks that allow you to [display results](https://inference.roboflow.com/docs/reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.render_boxes) on the screen or [save them to a file](https://inference.roboflow.com/docs/reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.VideoFileSink).

```python
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

pipeline = InferencePipeline.init(
    model_id="yolov8x-1280",
    video_reference="https://media.roboflow.com/inference/people-walking.mp4",
    on_prediction=render_boxes
)

pipeline.start()
pipeline.join()
```

#### *roboflow/supervision:*

Provide reusable computer vision tools. Whether you need to load your dataset from your hard drive, draw detections on an image or video, or count how many detections are in a zone.

[Must-Read]([Supervision Quickstart - Supervision (roboflow.com)](https://supervision.roboflow.com/develop/notebooks/quickstart/))

