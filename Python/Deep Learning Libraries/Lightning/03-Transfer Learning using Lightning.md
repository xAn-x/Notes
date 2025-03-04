
## *Working with Lightning's Models ⚡:*

```python
class AutoEncoder(L.LightningModule):
	def __init__(self):
		self.encoder=Encoder()
		self.decoder=Decoder()

class CIFAR10Classifier(L.LightningModule):
	def __init__(self):
		# init the pretrained lightning Models
		self.feature_extractor=AutoEncoder()
		# freeze the wts of Lightning model
		self.feature_extractor.freeze()
		# add custom classifier
		self.classifier=nn.Linear(100,10)

	# define forward-pass,training-step,validation-step etc
	def forward(self,x):
		representations=self.feature_extractor(x)
		x=classifier(representations)
		return x
```

## *Working with PyTorch's Models 🔥:*

```python
import torchvision.models as models


class ImagenetTransferLearning(L.LightningModule):
    def __init__(self):
        super().__init__()

        # init a pretrained resnet
        backbone = models.resnet50(weights="DEFAULT")
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        # Freeze the wts of feature_extractor
        self.feature_extractor.eval()

        # use the pretrained model to classify cifar-10 (10 image classes)
        num_target_classes = 10
        self.classifier = nn.Linear(num_filters, num_target_classes)

    def forward(self, x):
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x
```

## *Working with Hugging-Face Models 🤗:*

```python
from transformers import AutoModel,AutoTokenizer
import Lightning as L

class BertFineTuner(L.LightningModule):
	def __init__(self,checkpoint):
		self.checkpoint=checkpoint

	def setup(self):
		self.encoder=AutoModel.from_pretrained(self.checkpoint,output_attentions=True)
		self.encoder.train()
		self.W=nn.Linear(encoder.config.hidden_size,3)
		self.num_classes=3

	def forward(self,input_ids,attention_mask,token_ids):
		h,_,attn=self.bert(input_ids=input_ids,attention_mask=attention_mask,
		 token_type_ids=token_type_ids)
		h_cls=h[:,0]
		logits=self.W(h_cls)
		return logits, attn
```