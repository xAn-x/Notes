*tf.GradientTape* Api for automatic diffrentiation. Tf "records" relevant operations executed inside the context of a tf.GradientTape onto a *tape*/
Tensorflow then uses that tape to compute gradient of a recordered computation using "reverse mode differentiation".

```python
import tensorflow as tf

# Linear Regression from scratch:
weights=tf.Variable(tf.random.normal((no_of_units,in_features),dtype=tf.float32))
bias=tf.Variable(tf.zeros(no_of_units,),dtype=tf.float32)

model=lambda x,weights,bias: tf.matmul(x,tf.transpose(weights))+ bias
mse_loss=lambda true,predicted:  tf.reduce_mean(tf.square(predicted-true))

epochs,lr=1e3,1e-2
for i in range(epochs):
	with tf.GradientTape() as tape:
	# auto detect tf.Variable() tensor u can manually specify too,then it only looks em
		tape.watch([weights,bias])
		y_pred=model(x,weights,bias)
		loss=mse_loss(y,y_pred)
	
	[dw,db]=tape.gradient(loss,[weights,bias])
	# step
	weights.assign_sub(lr*dw)
	bias.assign_sub(lr*db)



# If want to de for a specefic_layer
# calc grad with resp to every trainable parameter
grad=tape.gradient(loss,layer.trainable_variables)

for var,g in zip(layer.trainable_variables,grad):
	print(f'{var.name}, shape:{g.shape}')
	var.assign_sub(lr*g)
```