## Basic Operations

```python
import tensorflow as tf

vec1=tf.constant([1,2,3],dtype=tf.float16) # constant tensor: im-mutable

vec2=tf.Variable([1,2,3],dtype=tf.flaot16,name="variable_tensor") # mutable
# used for making parameters in a NN
vec2.assign([3,4,5])
vec2[-1].assign(-5.3)

# operations of a Variable-tensor
t1.assign_add(t2) # inplace addition
t1.assign_subtract(t2) # inplace addition ...etc read docs


# Operations on tensor: U can do all the operantion that r +nt in numpy,just search docs
tf.reduce_sum(t,axis=None) # find the sum of ele
tf.max(t,axis=None) # finding max
tf.cumm_sum(t,axis=None) # cummalative sum ...etc
```

## Model Building

### 1. *Sequential Modelling*

```python
import tensorflow as tf
import keras

# Method-1
model=keras.Sequential([
	keras.layers.Input(input_shape=(28,28,1)),
	keras.layers.Conv2d(filters,kernel_size,padding="valid",activation="gelu"),
	keras.layers.Conv2d(filters,kernel_size,padding="same",activation="gelu"),
	keras.layers.Flatten(),
	keras.layers.Dense(out_features,activation="sigmoid"),
])


# Method-2
model=keras.Sequential(name="my_sequential_model")
model.add(keras.layers.Input(input_shape=(28,28,1)))
model.add(keras.layers.Conv2d(filters,kernel_size,padding="valid",activation="gelu"))
model.add(keras.layers.Conv2d(filters,kernel_size,padding="same",activation="gelu"))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(out_features,activation="sigmoid"))


model.summary() # provide the structure of NN,can only call if proived input,can use in b/w modelling to determine the output shape
print(model.layers) # all layers in your model
print(model.weights) # all parameters of ur model

# each model and layer has input and output property
print(model.inputs) 
print(model.outputs)
print(model.get_layer(layer_name))
print(model.layers[idx])

# to get the visual repr of model
keras.utils.plot_model(model,"my_model.png",show_shapes=True)

# compile
model.compile(
	loss=keras.loss.SparseCategoricalCrossEntropy(from_logits=False),
	metrics=[keras.metrics.accuracy(threshold=0.5),keras.metrics.topKAccuracy()],
	optimizer=keras.optimizer.Adam(lr=3e-4,momentum=0.2)
)

# fit(): to train
model.fit(
	train_ds,
	val_ds,
	optimizer="rms_prop",
	metrics=["accuracy","f1_score"],
	loss=keras.losses.MSE(lr=0.001),
	callbacks=[]
)


# evaluate: to test
model.evaluate(ds)

# predict(): to make preds
out=model.predict(x)
# or
out=model(x)
```

### 2. *Functional Modelling*

```python
import tensorflow as tf
import keras

inputs=keras.layers.Input((28,28,1))
dense1=keras.layers.Dense(64,"relu")(inputs)
dense2=keras.layers.Dense(64,"relu")(dense1)
out=dense2

model=keras.Model(inputs=[inputs],outputs=[outputs],name="model-name")

# U can use this whenever u want to have mixups of multiple models,or to tkae input or produce multiple outputs

encoder_input = keras.Input(shape=(28, 28, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

autoencoder = keras.Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()
```

### 3. *Model Subclassing*

```python
import tensorflow as tf
import keras

class LinearReg(keras.Model):
	# model initialization
	def __init__(self,in_features,out_features,activation="relu"):
		super().__init__()
		self.in_features=in_features
		self.out=out_features
		self.activation=activation

	# adding trainable params 
	def build(self,input_shape):
		self.Weights=self.add_weights(shape=(self.in_features,self.units)
		,initializer="random_normal"
		,trainable=True)
		
		self.Bias=self.add_weights(shape=(self.in_features,)
		,initializer="all_zeros"
		,trainable=True)

	# similar to 'forward' in pytorch 
	def call(self,x):
		return tf.matmul(x,self.Weights)+self.Bias
	
	def get_config(self):
		# Returns a dictionary containing the configuration used to initialize this
		# model. If the keys differ from the arguments in `__init__()`, then
		# override `from_config(self)` as well. This method is used when saving
		# the layer or a model that contains this model.
		
	def build_from_config(self,config):
		# use to init model from passed config-dict


# Same for making Custom Layers
class CustomLayer(keras.Layer):
	def __init__(self):
		pass
	def build(self,input_shape):
		pass
	def build_rom_config(self,config): 
		pass
	def call(self,x):
		# self.add_loss(loss_val),add_metric(),add_weights()
		pass
	def get_config(self):
		pass


```

### 4.*Preparing Data for training:*

```python
from tensorflow.data import Dataset,TextLineDataset

ds=Dataset.range(start,stop,skip) # create dataset using range-generator
ds=Dataset.from_tensor(1d_tensor) # use individual ele to create  a ds
ds=Dataset.from_tensor_slices(tensor) # Creating ds from tensor/list/dict
ds=TextLineDataset(['file1.txt','file2.txt']) # by reading file lines 
ds=Dataset.list_files("/path/*.txt") # from all files matching pattern
ds=Dataset.from_generator(generator) # from generator function/streaming data


# combining 2 or more tensor-dataset create a dataset
ds=Dataset.from_tensor_slices((feature_ds,target_ds))
ds=Dataset.zip((feature_ds,target_ds))

# Getting a subset of ds
ds.take(n) # take n elements
ds.skip(n) # skip n elements

# Traversing Dataset
# Returns an iterator which converts all elements of the dataset to numpy.
for ele in ds.as_numpy_iterator():
	print(ele)


## Methods to apply tranformation over dataset 
transform_ds=lambda ds: ds.filter(lambda x,y:x>5) # filter accordingly
ds.apply(transform_ds) # apply transformation over whole ds
ds.map(lambda x,y:x*2,y) #  apply transformation over individual elements
ds.shuffle() # shuffle the ds randomly
ds.batch(batch_size,drop_remainder=False,num_parallel_calls=None,
		 deterministic=None,) # batch ds

# we can also chain these operation one-after-another
```

### 5. *Model Training and Evaluation:*

```python
import tensorflow as tf
import tensorflow.keras as keras

model=Model()

# compile model
optimizer=tf.keras.optimizers.adam(lr=3e-4)
loss_fn=tf.keras.losses.SparseCategoricalCrossEntropy(from_logits=True)
metrics=[tf.keras.metrics.Accuracy(),tf.keras.TopKSparseCategoricalLoss(k=5)]

model.compile(
	loss=loss_fn,optimizer=optimizer,metrics=metrics
)

# Callbacks: To add extra functionality
callbacks=[
	keras.callbacks.EarlyStopping(patience=2,monitor="val_loss"),
    keras.callbacks.ModelCheckpoint(filepath='model.{epoch:02d}-{val_loss:.2f}.h5'),
    keras.callbacks.TensorBoard(log_dir='./logs'),
]

# fit model
history=model.fit(x,y,validation_split,validation_data,epochs,initial_epoch,
				  steps_per_epoch,callbacks=callbacks)
# steps_per_epoch: number of steps (batches of samples) before declaring one epoch finished and starting the next epoch.
# initial_epoch: Epoch at which to start training (useful for resuming a previous training run)

print(history.history) # contains info about loss and metrics to look for

model.evaluate(X_test,y_test) # evaluate performance on test data 

out=model.predict(input_batch) # To make prediction over unseen data
```
