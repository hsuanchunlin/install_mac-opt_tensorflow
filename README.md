# Install and Use Mac-optimized TensorFlow and TensorFlow Addons


Recently Apple had provided a Mac optimized TensorFlow which is able to utilize Mac's machine learning framework and GPUs. The link to Apple's blog is [here](https://blog.tensorflow.org/2020/11/accelerating-tensorflow-performance-on-mac.html). Because I am using Anaconda and conda as my major python virtual environment manager, here is my note for installation and testing.

## Set and find the env in Anaconda
For Mac-optimized TensorFlow, we need an environment in python 3.8. You can replace myenv by any preferred name.

```
conda create -n myenv python=3.8
```
After creating the python environment, find your virtual environment folder in
 
```
~/opt/Anaconda3/envs/myenv
```
## Installation

Now it's time to install the Mac-optimized TensorFlow.
Open a terminal window and activate the virtual environment by

```
conda activate myenv
```
Then goto the official github [here](https://github.com/apple/tensorflow_macos)

Please copy the following command from the official website and remove the '%' mark ahead if the command is copied from the official site.

```
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/apple/tensorflow_macos/master/scripts/download_and_install.sh)"

```
After downloading, the program will ask you where to install the tensorflow. Now you can put where your anaconda virtual environment is.

```
~/opt/Anaconda3/envs/myenv
```

## Usage
When use the Mac-optimized TensorFlow, import TensorFlow from tensorflow.compat.v2. Turning off eager execution caused some issues on my CycleGAN code and there are still issues when I used model.fit without eager execution. I have tried using TensorFlow without turning eager execution off on my 15 inch Macbook Pro 2019 with AMD GPU and TensorFlow was still able to utilize AMD GPU to run my training.

```python
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

#Option to disable eager execution
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

#Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='any') 
#Available options are 'cpu', 'gpu', and 'any'.

```

## Test the installation
A jupyter notebook or a python .py file is created and the following code can be copied to execute to test the performance of TensorFlow installation. Or you can find here [easy_test.py](./easy_test.py).

```python
import tensorflow.compat.v2 as tf
tf.enable_v2_behavior()

#Option to disable eager execution
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

#Import mlcompute module to use the optional set_mlc_device API for device selection with ML Compute.
from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='any')
#Available options are 'cpu', 'gpu', and 'any'.

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=1)
```
The sorce code is modified from the tutorial of [TensorFlow site](https://www.tensorflow.org/tutorials/quickstart/beginner).