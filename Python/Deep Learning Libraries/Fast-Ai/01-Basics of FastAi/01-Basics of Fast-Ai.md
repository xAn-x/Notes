fastai is a deep learning library which provides practitioners with high-level components that can quickly and easily provide state-of-the-art results in standard deep learning domains, and provides researchers with low-level components that can be mixed and matched to build new approaches. It aims to do both things without substantial compromises in ease of use, flexibility, or performance.

[![[Pasted image 20240624104910.png]]]([]())


## `Quick start:`
fastai’s applications all use the same basic steps and code:

- Create appropriate [`DataLoaders`](https://docs.fast.ai/data.core.html#dataloaders)
- Create a [`Learner`](https://docs.fast.ai/learner.html#learner)
- Call a _fit_ method
- Make predictions or view results.

```python
from fastai.vision.all import * # cv
from fastai.text.all import * # nlp
from fastai.collab import * # collaborative filtering
from fastai.tabular.all import * # tabular data


path=untar_data(URLs.PETS)/"images"

# We r using the pets dataset there all cats files starts with capital letter
def is_cat(x): return x[0].is_upper()

# Create DataLoaders
dl=ImageDataLoader.from_name_func(path,get_items=get_image_files(paths),valid_pct=0.2,seed=42,label_func=is_cat,item_tmfs=Resize(224))

# Create a learner
learner=vision_learner(dls,resnet34,metrics=error_rate)
learner.show_results(max_n=6,figsize=(6,8))

# fine tune
learner.fine_tune(epochs=3) 

# make preds
is_cat,_,probs = learn.predict(img)
print(f"Is this a cat?: {is_cat}.")
print(f"Probability it's a cat: {probs[1].item():.6f}")
# >> Is this a cat?: True.
# >> Probability it's a cat: 0.999722

# Making Interpretation
intrep=ClassificationInterpretation.from_learner(learner)
intrep.plot_top_losses(k=5)
```