## Train a text classifier from a pretrained model

```python
from fastai.text.all import *

path=untar_data(URLs.IMDB)

dls=TextDataLoader.from_folder(path,valid="/test")
dls.show_batch()

# The library automatically processed all the texts to split then in _tokens_, adding some special tokens like:
# - `xxbos` to indicate the beginning of a text
# - `xxmaj` to indicate the next word was capitalized

learner = text_classifier_learner(dls, AWD_LSTM, drop_mult=0.5, 
								  metrics=accuracy)
learner.fine_tune(4,1e-2)
learner.show_results()

sentiment,idx,prob_vec=learner.predict("I really like that movie")
```

## The ULMFiT approach

**ULMFiT (Universal Language Model Fine-tuning)** is a transfer learning approach for natural language processing (NLP) tasks. It involves training a large language model (LLM) on a massive dataset and then fine-tuning it on a specific downstream task with a much smaller dataset.

**Key Steps of ULMFiT:**

1. **Pre-train a large language model:** Train an LLM, such as GPT-3 or BERT, on a large and diverse text corpus. This LLM learns general language representations and patterns.
2. **Freeze the LLM:** Once the LLM is pre-trained, its weights are frozen, meaning they are no longer updated during training.
3. **Add task-specific layers:** Add a few additional layers on top of the frozen LLM. These layers are task-specific and are responsible for adapting the LLM to the downstream task.
4. **Fine-tune the task-specific layers:** Fine-tune only the task-specific layers on the downstream dataset. This involves updating the weights of these layers to minimize the loss function for the specific task.

**Advantages of ULMFiT:**

* **Fast and efficient:** Fine-tuning is much faster and requires less data than training an LLM from scratch.
* **Improved performance:** ULMFiT often achieves state-of-the-art performance on NLP tasks, even with limited training data.
* **Transferability:** The pre-trained LLM provides a strong foundation for fine-tuning on various downstream tasks.

**Applications of ULMFiT:**

ULMFiT has been successfully applied to a wide range of NLP tasks, including:

* Text classification
* Named entity recognition
* Question answering
* Machine translation
* Summarization

ULMFiT has also been used as a starting point for developing more advanced transfer learning techniques, such as few-shot learning and continual learning. ==You get even better results if you fine tune the (sequence-based) language model prior to fine tuning the classification model.==

![[Pasted image 20240625102854.png]]


### Fine-tuning a language model on IMDb:

```python
from fastai.text.all import *

# Creating DataLoaders for language-modelling
dls_lm=TextDataLoader.from_folder(path,is_lm=True,valid_pct=0.1)
dls_lm.show_batch() # Here the task is to guess the next word, so we can see the targets have all shifted one word to the right. 


# Creating language Modelling learner
lm_learner=language_model_learner(dls_lm,AWD_LSTM,metrics=[accuracy,Perplexity()],path=path,wd=0.1).to_fp16()
# By default, a pretrained learner is in a frozen state, meaning that only the head of the model will train while the body stays frozen. We show you what is behind the fine_tune method here and use a fit_one_cycle method to fit the model
lm_learner.fit_one_cycle(1,1e-2)

lm_learner.save("1epoch")
lm_learner=learner.load("1epoch")
#We can them fine-tune the model after unfreezing:
lm_learner.unfreeze()
lm_learner.fit_one_cycle(10, 1e-3)

#Once this is done, we save all of our model except the final layer that converts activations to probabilities of picking each token in our vocabulary. The model not including the final layer is called the encoder. We can save it with `save_encoder`:
lm_learner.save_encoder('finetuned_lm_encoder')
```

Before using this to fine-tune a classifier on the reviews, we can use our model to generate random reviews: since it’s trained to guess what the next word of the sentence is, we can use it to write new reviews:

```python
TEXT = "I liked this movie because"
N_WORDS = 40
N_SENTENCES = 2

preds = [lm_learner.predict(TEXT, N_WORDS, temperature=0.75) 
         for _ in range(N_SENTENCES)]

print("\n".join(preds))

# >> i liked this movie because of its story and characters . The story line was very strong , very good for a sci - fi film . The main character , Alucard , was very well developed and brought the whole story

# >> i liked this movie because i like the idea of the premise of the movie , the ( very ) convenient virus ( which , when you have to kill a few people , the " evil " machine has to be used to protect
```

### Training a text classifier:

```python
# create dataloaders
dls=TextDataLoader.from_folder(path,valid="test",
							   text_vocab=dls_lm.vocab)
# The main difference is that we have to use the exact same vocabulary as when we were fine-tuning our language model, or the weights learned won’t make any sense. We pass that vocabulary with `text_vocab`

classifier=text_classifier_learner(dls,AWD_LSTM,drop_mult=0.3,
								  metrics=accuracy)

classifier=classifier.load_encoder("finetuned_lm_encoder")
classifier.fit_one_cycle(1,2e-2)

# We can pass `-2` to `freeze_to` to freeze all except the last two parameter groups:
classifier.freeze_to(-2)
classifier.fit_one_cycle(1,2e-2)

#Then we can unfreeze a bit more, and continue training:
classifier.freeze_to(-3)
classifier.fit_one_cycle(2,2e-2)
```