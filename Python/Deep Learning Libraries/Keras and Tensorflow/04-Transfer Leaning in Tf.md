### Freezing and Un-Freezing a layer|model

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# How to get trainable weights of a layer 
model=keras.Sequential([
	keras.layers.Input((28,28),name="Input Layer"),
	keras.layers.Conv2d(8,(3,3),padding='same',activation='relu',name="conv1"),
	keras.layers.MaxPool2D(3,name="max_pool_1"),
	keras.layers.Conv2d(16,(3,3),padding='same',activation='relu',name="conv2"),
	keras.layers.MaxPool2D(3,name="max_pool_2"),
	keras.layers.Conv2d(32,(5,5),padding='same',activation='relu',name="conv3"),
	keras.layers.Flatten(name="flatten"),
	keras.layers.BatchNormalization(name="batchnorm")
	keras.layers.Dense(10,name="classification_head")
])
# Build Model first if Input is not specified
for layer in model.layers:
	print(f"{layer.name}:{layer.trainable_weights}")

# Freezing of model-weights of a layer
model.get_layer(name="conv1").trainable=False
model.layers[1].trainable=False

# Freezing weights of whole model
model.trainable=False
```

### Finetune a pretrained model

```python
import tensorflow as tf
from tensorflow import keras

base_model = keras.applications.Xception(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3), # Modify the input layer to accept this shape img
    include_top=False # Do not include the ImageNet classifier at the top.
)  

# Freeze base-model weights
base_model.trainable=False

# add custom head
classifier=nn.Sequential([
	base_model,
	keras.layers.Dense(10)
])

# Compile and fine-tune
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy()])
model.fit(new_dataset, epochs=20, callbacks=..., validation_data=...)
```

### Finetune with custom training loop

```python
# Create base model
base_model = keras.applications.Xception(
    weights='imagenet',
    input_shape=(150, 150, 3),
    include_top=False)
# Freeze base model
base_model.trainable = False

# Create new model on top.
inputs = keras.Input(shape=(150, 150, 3))
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)

loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam()

# Iterate over the batches of a dataset.
for inputs, targets in new_dataset:
    # Open a GradientTape.
    with tf.GradientTape() as tape:
        # Forward pass.
        predictions = model(inputs)
        # Compute the loss value for this batch.
        loss_value = loss_fn(targets, predictions)

    # Get gradients of loss wrt the *trainable* weights.
    gradients = tape.gradient(loss_value, model.trainable_weights)
    # Update the weights of the model.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
```