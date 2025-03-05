Hugging face is a deep-learning lib, that serves open source models, it provide us models and datasets that we can use.

```shell
pip install transformers datasets
```

```python
from transformers import AutoModel,AutoTokenizer,AutoConfig

model_ckpt="bart-base-cased"

# help in converting tokens-to-ids
tokenizer=AutoTokenizer.from_pretrained(model_ckpt)
# model with pretrained wts
model=AutoTokenizer.from_pretrained(model_ckpt)
# config file, containing details about model
config=AutoTokenizer.from_pretrained(model_ckpt)

sents=["Hugging Face 🤗, has democratized deep-learning","I love mathematics and deep-learning."]

inputs=tokenizer(sents,return_tensors='pt',padding=True)
out=model(*inputs)
```

**pipeline():** We can directly use some model from hugging face hub trained for certain task without need to train our own.

```python
from transformers import pipeline

ner_pipeline=pipeline('ner',model="--modelName--")
out=ner_pipeline(["I am Deepanshu Bhatt, I live in Indore","Rahul and Shyam are close friends."])

classification_pipeline=pipeline("classification",model)
out=classification_pipeline("I love watching football.")
```

### Tokenizer 

- Using a pretrained tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased") # Initialize tokenizer from pretrained model

text = "This is a test sentence."
tokens = tokenizer.tokenize(text) # Tokenize text
print(tokens) # Example: ['this', 'is', 'a', 'test', 'sentence', '.']


encoded_input = tokenizer(text,add_special=True,padding='max-length',truncation=True) # Encode text (includes tokenization, attention-mask and other steps like adding special tokens)


print(encoded_input) # Example: {'input_ids': [101, 2023, 2003, 1037, 3231, 6251, 1012, 102], 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1]}

decoded_text = tokenizer.decode(encoded_input["input_ids"]) # Decode token IDs back to text,
print(decoded_text) # Example: '[CLS] this is a test sentence. [SEP]'


# Key Properties and Methods

tokenizer.vocab_size # Vocabulary size
tokenizer.pad_token # Padding token
tokenizer.bos_token # Beginning of sequence token
tokenizer.eos_token # End of sequence token
tokenizer.unk_token # Unknown token
tokenizer.mask_token # Masking token
tokenizer.sep_token # Separator token
tokenizer.cls_token # Classification token

tokenizer.convert_tokens_to_ids(tokens) # Convert tokens to IDs
tokenizer.convert_ids_to_tokens(encoded_input["input_ids"]) # Convert IDs to tokens

# Handling Special Tokens

tokenizer.add_special_tokens({'additional_special_tokens': ['<extra_id_0>', '<extra_id_1>']}) # Add special tokens
tokenizer.encode("A new sentence with special tokens <extra_id_0>", add_special_tokens=True) # Encode with special tokens


# Saving and Loading

tokenizer.save_pretrained("path/to/save") # Save tokenizer locally
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("path/to/save") # Load tokenizer from local path

# Advanced Usage

batch_sentences = ["This is sentence 1.", "This is sentence 2."]
encoded_inputs = tokenizer(batch_sentences, padding=True, truncation=True, return_tensors="pt") # Batch encoding with padding and truncation for PyTorch
print(encoded_inputs)

# Check if the tokenizer is fast
print(f"Is the tokenizer fast? {tokenizer.is_fast}")
```

- Training your own tokenizer from existing one

```python

# Method 1: From Scratch

from tokenizers import Tokenizer
from tokenizers.models import BPE,SentencePeice,Unigram
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer,SentencePeiceTrainer,UnigramTrainer

# Initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

# Set pre-tokenizer
tokenizer.pre_tokenizer = Whitespace()

# Customize training arguments (adjust as needed)
trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], vocab_size=30000, min_frequency=2)

# Example data (replace with your actual data)
data = ["This is the first sentence.", "This is the second sentence."]
files = ["path/to/your/text/file1.txt", "path/to/your/text/file2.txt"] # Or paths to multiple text files


# Train the tokenizer
# if you are using a list of strings
tokenizer.train_from_iterator(data, trainer=trainer)
# if you are using files
# tokenizer.train(files, trainer=trainer)



# Save the trained tokenizer
tokenizer.save("path/to/save/tokenizer.json")  # You can choose your preferred path

# Load the trained tokenizer
from transformers import PreTrainedTokenizerFast

loaded_tokenizer = PreTrainedTokenizerFast(tokenizer_file="path/to/save/tokenizer.json")



# Method 2: Extending an Existing Tokenizer

from transformers import AutoTokenizer

# Load a pre-trained tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Add new tokens (replace with your desired tokens)
new_tokens = ["<new_token_1>", "<new_token_2>"]
tokenizer.add_tokens(new_tokens)

# Resize the model's embedding matrix if using with a model
# Assuming 'model' is your loaded Hugging Face model
# model.resize_token_embeddings(len(tokenizer))

# Save the updated tokenizer
tokenizer.save_pretrained("path/to/save/updated_tokenizer")

# Load the updated tokenizer
updated_tokenizer = AutoTokenizer.from_pretrained("path/to/save/updated_tokenizer")
```

### Models

```python

# Model Usage

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"  
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Example text
text = "This is a test sentence."

# Tokenize input
inputs = tokenizer(text, return_tensors="pt")  # Use "tf" for TensorFlow & "np" for numpy,default is list

# Perform inference
with torch.no_grad(): 
    outputs = model(**inputs)

# Accessing outputs (example for sequence classification)
logits = outputs.logits  # Raw model outputs
predicted_class_id = logits.argmax().item()

# Using pipelines (simplified interface)
classifier = pipeline("sentiment-analysis")
result = classifier(text)
print(result) # Example: [{'label': 'POSITIVE', 'score': 0.9998933672904968}]

# Key Properties and Methods

model.config # Model configuration
model.base_model # Access the underlying base model (e.g., BERT)
model.num_parameters() # Number of model parameters

# Saving and Loading

model.save_pretrained("path/to/save") # Save the model locally
model = AutoModelForSequenceClassification.from_pretrained("path/to/save") # Load the model from local path

```

- Training model using **Trainer** API

```python

from transformers import Trainer, TrainingArguments,AdamW
# for seq-to-seq language modelling
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results", # where to save
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    optimizer=AdamW(lr=4e-4),
    logging_strategy="epoch",
    logging_dir="./logs",
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, # Your training dataset
    eval_dataset=eval_dataset, # Your evaluation dataset
    tokenizer=tokenizer, # Your tokenizer
)

# Start training
trainer.train()

# Evaluation
trainer.evaluate()

# Prediction
predictions = trainer.predict(test_dataset)
```
- Training using **accelerate**: 
	The `accelerate` library simplifies distributed training and inference, managing complexities like multi-GPU and TPU usage. 

*  **Launcher:**  Easily launch scripts across different hardware configurations (single/multi-GPU, TPU, CPU).
* **Abstractions:** Provides high-level APIs to handle data loading, model movement, and gradient accumulation

```python

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from accelerate import Accelerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# Initialize accelerator
accelerator = Accelerator()

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Prepare training arguments
training_args = TrainingArguments(output_dir="./results", num_train_epochs=3, per_device_train_batch_size=16)

# Prepare Trainer (modified to include compute_metrics function) 
def compute_metrics(pred): 
	labels = pred.label_ids 
	preds = pred.predictions.argmax(-1) 
	precision, recall, f1, _ = precision_recall_fscore_support(labels, preds,
		 average='weighted') 
	acc = accuracy_score(labels, preds) 
	return { 'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall }
	
# Prepare Trainer
trainer = Trainer(model=model, args=training_args, 
	train_dataset=train_dataset, eval_dataset=eval_dataset, 
	tokenizer=tokenizer,compute_metrics=compute_metrics
)

# Prepare model, optimizer, and data loaders for training with accelerator
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, trainer.optimizer, trainer.get_train_dataloader(), trainer.get_eval_dataloader()
)




# Training loop
for epoch in range(training_args.num_train_epochs):
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(model): # For gradient accumulation
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()


    # Evaluation loop 
    model.eval()
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            utputs = model(**batch) 
            predictions = outputs.logits.argmax(dim=-1) 
            
            # gather the preds and labels as may be spread across multiple devices
            predictions_gathered = accelerator.gather(predictions) 
            labels_gathered = accelerator.gather(batch["labels"])
	    
	    all_predictions.append(predictions_gathered.cpu().numpy()) 
	    all_labels.append(labels_gathered.cpu().numpy()) # Concatenate predictions and labels for final metric calculation 
	    
	    all_predictions = np.concatenate(all_predictions) 
	    all_labels = np.concatenate(all_labels) # Compute and print metrics 
	    
	    eval_metric = compute_metrics(EvalPrediction(
			    predictions=all_predictions, label_ids=all_labels
		)) 
	    print(f"Epoch {epoch + 1} metrics:", eval_metric)
            
    model.train()

# Save the trained model
accelerator.wait_for_everyone() # Ensure all processes have finished before saving
unwrapped_model = accelerator.unwrap_model(model) # Get the original model
unwrapped_model.save_pretrained("path/to/save", save_function=accelerator.save) # Save using accelerator
```

# Key Methods and Properties

* `accelerator.prepare()`:  Prepares model, optimizer, and dataloaders for distributed training.
* `accelerator.backward()`:  Performs backpropagation in a distributed setting.
* `accelerator.accumulate()`:  Handles gradient accumulation.
* `accelerator.unwrap_model()`:  Retrieves the original model from the wrapped accelerated model.
* `accelerator.save()`:  Saves the model in a distributed environment.
* `accelerator.gather()`: Gather data across various devices

### Datasets 

Datasets library is a place where we can have access to several datasets that are curated and cleaned and can be used freely as per our own requirements. 

We can download the datasets and can use it to train our models. Datasets lib uses **Apache Arrow** which is very efficient when comes to working with big-data.

```python
from datasets import load_dataset, load_from_disk

# Load from the Hugging Face Hub
dataset = load_dataset("imdb")

# Load from local disk
# dataset = load_from_disk("path/to/dataset") # Assumes the dataset has been saved previously

# Exploring Datasets

dataset["train"].features # Dataset features (data types of columns)
dataset["train"][0] # Access the first example in the training split
dataset["train"].num_rows # Number of rows in the training split
dataset.column_names # Column names in the dataset

# Slicing and Dicing

dataset["train"].select([0, 1, 2]) # Select specific rows by index
dataset["train"].filter(lambda example: example["label"] == 1) # Filter examples based on a condition
dataset["train"].sort("text") # Sort the dataset based on a column

# Data Conversion and Manipulation

dataset.map(lambda example: {"text_length": len(example["text"])}) # Add a new column
dataset = dataset.rename_column("text", "movie_review") # Rename a column
dataset = dataset.remove_columns(["label"]) # Remove a column
dataset = dataset.cast_column("label", datasets.ClassLabel(num_classes=2)) # Cast column type

# Concatenation and Merging

dataset_part1 = load_dataset("imdb", split="train[:10%]")
dataset_part2 = load_dataset("imdb", split="train[10%:20%]")
concatenated_dataset = datasets.concatenate_datasets([dataset_part1, dataset_part2]) # Concatenate datasets vertically


# Streaming

for example in load_dataset("imdb", split="train", streaming=True).take(5): # Process only the first 5 examples
    print(example)

# Saving to Disk

dataset.save_to_disk("path/to/save/dataset")


# DatasetDict Methods: {'split:Dataset}

# Mapping
dataset = dataset.map(lambda examples: {'text_length': [len(text) for text in examples['text']]}, batched=True, batch_size=1000)

# Filtering
dataset = dataset.filter(lambda example: example['text_length'] > 100)

# Sorting
dataset = dataset.sort('text_length')

# Shuffling
dataset = dataset.shuffle(seed=42)

# Renaming columns
dataset = dataset.rename_column('old_column_name', 'new_column_name')

# Removing columns
dataset = dataset.remove_columns(['column1', 'column2'])

# Casting columns
dataset = dataset.cast_column('label', datasets.ClassLabel(num_classes=2))

# Train Test Validation Split
dataset = dataset['train'].train_test_split(test_size=0.2, seed=42) # Split the train set into train and validation
# Now you have dataset["train"] and dataset["test"]

# Add a new column using another column example
def add_length_column(example):
    example['length'] = len(example['text'])
    return example

dataset = dataset.map(add_length_column)

# Set the format for specific framework
dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
dataset.reset_format() # back to apache-arrow

# Creaing Dataset from ur own datasets
csv_ds=load_dataset('csv','file-path.csv')
json_ds=DatasetDict('csv',{
	'train':"train-file.json",
	"val":"validation-file.json",
},field="data") # field signifies the key inside which all the data is present, if not provided , use all the unique keys as cols.

# Exporting to different formats
dataset.to_csv('data.csv')
dataset.to_json('data.json')
```

### Data Collators:

Data collators prepare your data for the model by batching and padding.  They're essential for handling variable-length sequences.

**Types of Data Collators**

Hugging Face provides several pre-built collators, each designed for a specific task:

* **`DataCollatorWithPadding`**:  Handles padding of variable-length sequences.  Essential for tasks like text classification and sequence-to-sequence.  It uses a tokenizer's padding token to make all sequences in a batch the same length.
```python


```python
from transformers import DataCollatorWithPadding, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # Example tokenizer

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Example usage:
fake_data = [{"input_ids": [1, 2, 3]}, {"input_ids": [1, 2, 3, 4, 5]}]
padded_batch = data_collator(fake_data) 
# padded_batch will have 'input_ids' padded to the same length, and 'attention_mask' indicating valid tokens.
```

* **`DataCollatorForSeq2Seq`**: Specialized for sequence-to-sequence tasks like translation and summarization.  Handles padding for both input and target sequences. It also dynamically determines the labels for decoder-only models by shifting the labels to the right and replacing the last token with -100 to ignore loss calculation.

```python
from transformers import DataCollatorForSeq2Seq

data_collator = DataCollatorForSeq2Seq(model=model, tokenizer=tokenizer) # Requires model and tokenizer
```

* **`DataCollatorForLanguageModeling`**:  Used for language modeling tasks. Creates input and target sequences dynamically.  It applies techniques like masking to train the model to predict missing words.

```python
from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)  # mlm for Masked Language Modeling, if false then will be used for causal laguage modelling.
```

* **`DataCollatorForTokenClassification`**: Designed for token classification tasks like Named Entity Recognition (NER).  Ensures labels are correctly aligned with padded sequences.

```python
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

* **`DataCollatorForMultipleChoice`**: Used for multiple-choice question answering tasks.

```python
from transformers import DataCollatorForMultipleChoice

data_collator = DataCollatorForMultipleChoice(tokenizer=tokenizer)
```


## Custom Data Collators

You can create custom data collators for specialized needs.  This involves creating a class with a `__call__` method that takes a list of samples and returns a batch suitable for your model.  This is useful for tasks with unique data structures or processing requirements.  See the Hugging Face documentation for examples and details.

```python
from transformers import DataCollatorWithPadding
import torch

# Inherit from DataCollatorWithPadding for basic padding functionality
class CustomDataCollator(DataCollatorWithPadding):
    def __init__(self, tokenizer, max_length=None):
        super().__init__(tokenizer, max_length=max_length)
        # Add any custom attributes here, e.g., for special processing

    def __call__(self, features):
        # Perform basic padding
        batch = super().__call__(features)

        # Custom processing logic
        # Example 1: Add a new key to the batch
        batch['new_key'] = torch.tensor([f['some_feature'] for f in features])

        # Example 2: Modify existing keys
        if 'input_ids' in batch:
            # Example: Truncate sequences longer than a threshold
            threshold = 128  # Example threshold
            batch['input_ids'] = batch['input_ids'][:, :threshold]
            batch['attention_mask'] = batch['attention_mask'][:, :threshold]

        # Example 3: Handle missing keys gracefully
        if 'labels' not in batch:
            # Provide default labels if they're not present in the input features
            batch['labels'] = torch.tensor([-100] * len(features)) # Default labels

        return batch




# Example usage with a dummy dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
data_collator = CustomDataCollator(tokenizer)


# Dummy Features
features = [
    {'input_ids': tokenizer("This is a test sentence.", return_tensors="pt").input_ids[0], 'some_feature': 1},
    {'input_ids': tokenizer("This is a longer test sentence.", return_tensors="pt").input_ids[0], 'some_feature': 0}
]



batch = data_collator(features)

print(batch)

```****