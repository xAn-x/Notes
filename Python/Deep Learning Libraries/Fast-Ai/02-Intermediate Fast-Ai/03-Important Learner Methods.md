
## 1. fit() vs fine_tune() vs fit_one_cycle()
### `learner.fine_tune()

The `learner.fine_tune()` method is used to fine-tune a pre-trained model. This is particularly useful ==when you have a large dataset and want to leverage the knowledge learned by a pre-trained model==, rather than training the model from scratch. The fine-tuning process ==involves freezing the lower layers of the model and only training the higher layers==, which can significantly speed up the training process and improve the model's performance.The `learner.fine_tune()` method takes the following arguments:

- `epochs`: The number of epochs to train the model.
- `lr_max`: The maximum learning rate to use during training.
- `wd`: The weight decay to use during training.
- `cbs`: A list of callbacks to use during training.

### learner.fit()

The `learner.fit()` method is the primary method used to train a machine learning model in FastAI. This method takes the following arguments:

- `epochs`: The number of epochs to train the model.
- `lr`: The learning rate to use during training.
- `wd`: The weight decay to use during training.
- `cbs`: A list of callbacks to use during training.

The `learner.fit()` method is a more general method than `learner.fine_tune()`, as it can be used to ==train a model from scratch or to continue training a pre-trained model==.

### learner.fit_one_cycle()

The `learner.fit_one_cycle()` method is a ==variation of the `learner.fit()` method that uses the One Cycle policy for training==. The One Cycle policy is a learning rate scheduling technique that has been shown to improve the performance of deep learning models.The `learner.fit_one_cycle()` method takes the following arguments:

- `epochs`: The number of epochs to train the model.
- `lr_max`: The maximum learning rate to use during training.
- `div_factor`: The ==factor by which to divide the maximum learning rate== to get the minimum learning rate.
- `pct_start`: The ==percentage of the total training time during which the learning rate will increase==.
- `wd`: The weight decay to use during training.
- `cbs`: A list of callbacks to use during training.


The One Cycle policy works by starting with a low learning rate, gradually increasing it to a maximum value, and then gradually decreasing it back to a low value. This can help the model escape local minima and find a better global minimum.

## 2. Saving and Loading a learner

### Learner.save[](https://docs.fast.ai/learner.html#learner.save)

> ```
>  Learner.save (file, with_opt=True, pickle_protocol=2)
> ```

_Save model and optimizer state (if `with_opt`) to `self.path/self.model_dir/file`_

`file` can be a `Path`, a `string` or a buffer. `pickle_protocol` is passed along to `torch.save`.


### Learner.load[](https://docs.fast.ai/learner.html#learner.load)

> ```
>  Learner.load (file, device=None, with_opt=True, strict=True)
> ```

_Load model and optimizer state (if `with_opt`) from `self.path/self.model_dir/file` using `device`_

### Learner.export[](https://docs.fast.ai/learner.html#learner.export)

> ```
>  Learner.export (fname='export.pkl', pickle_module=<module 'pickle' from
>                  '/usr/lib/python3.10/pickle.py'>, pickle_protocol=2)
> ```

_Export the content of `self` without the items and the optimizer state for inference_

## 3.Transfer learning[](https://docs.fast.ai/learner.html#transfer-learning)

---

[source](https://github.com/fastai/fastai/blob/master/fastai/learner.py#L656)

### Learner.unfreeze[](https://docs.fast.ai/learner.html#learner.unfreeze)

> ```
>  Learner.unfreeze ()
> ```

_Unfreeze the entire model_

---

[source](https://github.com/fastai/fastai/blob/master/fastai/learner.py#L653)

### Learner.freeze[](https://docs.fast.ai/learner.html#learner.freeze)

> ```
>  Learner.freeze ()
> ```

_Freeze up to last parameter group_

---

[source](https://github.com/fastai/fastai/blob/master/fastai/learner.py#L647)

### Learner.freeze_to[](https://docs.fast.ai/learner.html#learner.freeze_to)

> ```
>  Learner.freeze_to (n)
> ```

_Freeze parameter groups up to `n`_

## TTA[](https://docs.fast.ai/learner.html#tta)

---

[source](https://github.com/fastai/fastai/blob/master/fastai/learner.py#L665)

### Learner.tta[](https://docs.fast.ai/learner.html#learner.tta)

> ```
>  Learner.tta (ds_idx=1, dl=None, n=4, item_tfms=None, batch_tfms=None,
>               beta=0.25, use_max=False)
> ```

_Return predictions on the `ds_idx` dataset or `dl` using Test Time Augmentation_

In practice, we get the predictions `n` times with the transforms of the training set and average those. The final predictions are `(1-beta)` multiplied by this average + `beta` multiplied by the predictions obtained with the transforms of the dataset. Set `beta` to `None` to get a tuple of the predictions and tta results. You can also use the maximum of all predictions instead of an average by setting `use_max=True`.

If you want to use new transforms, you can pass them with `item_tfms` and `batch_tfms`.

**Test Time Augmentation (TTA)**

**Definition:**

Test Time Augmentation (TTA) is a technique used in machine learning to improve the performance of a model during inference (testing). It ==involves applying additional transformations or augmentations to the input data during testing that were not used during training.==

**Purpose:**

TTA ==aims to increase the model's robustness and generalization capabilities by exposing it to a wider range of input variations.== This can help mitigate overfitting and improve performance on unseen data.

**Methods:**

Common TTA methods include:

* **Horizontal flipping:** Flipping the input image horizontally.
* **Vertical flipping:** Flipping the input image vertically.
* **Rotation:** Rotating the input image by a certain angle.
* **Scaling:** Resizing the input image to different scales.
* **Cropping:** Taking random crops from the input image.
* **Color jittering:** Randomly adjusting the brightness, contrast, saturation, and hue of the input image.

**Benefits:**

TTA can provide several benefits, including:

* **Improved generalization:** Exposing the model to different input variations helps it learn more robust features.
* **Reduced overfitting:** TTA can help prevent the model from memorizing specific training examples.
* **Increased accuracy:** By leveraging multiple augmented inputs, TTA can improve the model's overall accuracy.
* **Ensemble-like performance:** TTA can mimic the behavior of an ensemble of models, where each model is trained on a different augmented dataset.

**Applications:**

TTA is widely used in various machine learning tasks, including:

* Image classification
* Object detection
* Semantic segmentation
* Natural language processing

**Limitations:**

While TTA can be beneficial, it also has some limitations:

* **Increased computational cost:** Augmenting the input data during inference can increase the computational cost.
* **Potential for over-augmentation:** Excessive augmentation can lead to overfitting and reduced performance.
* **Not suitable for all tasks:** TTA may not be effective for all machine learning tasks, particularly those that require precise input information.