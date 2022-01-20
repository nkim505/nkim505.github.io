---
layout: single #post도 가능
title:  "Deeplearning.Ai TensorFlow Developer (Course 1)"
---

** [알림] **  <br>
💁‍♀️ 텐서플로우 자격증 취득에 도움이 되는 **코세라** 강의 <br>
💻 ["Deeplearning.Ai TensorFlow Developer"](https://www.coursera.org/professional-certificates/tensorflow-in-practice?trk_ref=globalnav) Course 1 : Introduction을 듣고 강의 내용을 정리하였습니다.<br>
🧠 수업을 들으며 동시에 정리한 내용이어서(필기노트 대용), 의식의 흐름이 강하게 개입되었습니다.<br>
😚 저만의 이해 방법을 풀어 놓아, 강의와 함께 보시는 분께는 작은 도움이 될 수 있을 것 같습니다.<br>

# Week 2.
## Lectures
```python
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
```
<br>

>Using a number is a first step in avoiding bias -- instead of labeling it with words in a specific language and excluding people who don’t speak that language! You can learn more about bias and techniques to avoid it [here.](https://ai.google/responsibilities/responsible-ai-practices/)
<br>

```python
model = keras.Sequantial([
	keras.layers.Flatten(input_shape = (28, 28),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(10, activation = tf.nn.softmax)
])
```

`keras.layers.Dense(units=neuron갯수, activation =, input_shape=)` 하나가 layer 하나의 설정 말한다.   
`keras.layers.Flatten`도 레이어 하나를 뜻한다.   
따라서 위는 three layers이다.   
마지막 레이어는 10개 뉴론이다. 왜냐면 10개의 옷 데이터셋이기때문에. 이건 언제나 맞춰줘야한다.   
첫 레이어는 input layer인데 인풋셰입이 28,28 임을 알려주는 option이 들어가있다. 이걸 flatten 1개의 array로 바꿔서 넣어줌.


### Hello World. A Computer Vision Example   
 

```python
import tensorflow as tf
print(tf.__version__)

mnist = tf.keras.datasets.fashion_mnist

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
```

```python
import matplotlib.pyplot as plt
plt.imghow(training_images[0])
print(training_labels[0])
print(training_images[0])

#normalizing
training_images = training_images/255.0
test_images = test_images/255.0
```

```python
#design the model
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
tf.keras.layers.Dense(128, activation=tf.nn.relu),
tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

#complie the model to find the loss function and the optimizer
model.compile(optimizer = tf.train.AdamOptimizer(), loss = 'sparse_categorical_crossentropy')

#fit the model 
model.fit(training_images, trainin_labels, epochs=5)
```
+ `compiling model`: the goal is to make a guess as to what the relationship is btw the input data and the output data   
and measure how well or how badly it did using the loss function, use the `optimizer` to generate a new guess and repeat.   
+ `fitting model`: trying to fit the training images to the training labels.   

그리고 Test data 로 넘어간다. 여기서 loss 값이 더 나쁘게 나오면 문제가 있다. test set에서 덜 정확한 것.   
이때 신경망의 파라메터를 바꾸거나 epochs를 바꾸거나 방법을 시도할 수 있다. [여기서('Beyond Hello World, A Computer Vision Example')](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W2/ungraded_labs/C1_W2_Lab_1_beyond_hello_world.ipynb) 해당 노트북을 써서 해볼 수 있다.   

>Your job now is to go through the workbook, try the exercises and see by tweaking the parameters on the neural network or changing the epochs   
   

### Using callbacks to control training
> Q. How can I stop training when I reach a point that I want to be at? What do I always have to hard code it to go for a certain number of epochs?    A. The training loop does support callbacks

모든 epoch마다 우리는 코드 함수를 callback 할 수 있다.

```python
model.fit(training_images, training_labels, epochs = 5
```
여기서 model.fit 함수는 트레이닝 루프를 실행한다.

그리고 callback 코드를 파이썬으로 작성하면 아래와 같다.

```python
class myCallback(tf.keras.callbacks.Callback):
def on_epoch_end(self,epoch, logs={}):
if logs.get('loss')<0.4:
print("\n Loss is low so cancelling training!")
self.model.stop_training = True
```

이제 우리가 만든 myCallback 클래스를 가지고 원래 만들어놨던 코드를 수정하면 이렇게 된다.

```python
callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
(training_images, traning_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images/255.0
test_images = test_images / 255.0
model = tf.keras.models.Sequential([
tf.keras.layers.Flatten(input_shape - (28,28)),
tf.keras.layers.Dense(512, activation = tf.nn.relu),
tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs = 5)
```
<br>
시작점에서 `callback = myCallback()` 넣어주고, 마지막에 model.fit 안에서 파라메터로 `callbacks = [callbacks]`을 넣어준다.
<br>
<br>
> See how to implement Callbacks (Lab 2)   Experiment with using Callbacks in [this notebook](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W2/ungraded_labs/C1_W2_Lab_2_callbacks.ipynb) -- work through it to see how they perform!
<br>

```python
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

callbacks = myCallback()

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
```
<br>
<br>

**exercise tip** : nomalization하니까 5번째 epoch에서 acc>0.99 성공함.
nomalization 안했을 때는 10번째에서도 0.95 정도였음.




# Week 3. <br> Enhancing Vision with Convolutional Neural Networks
## Lectures
### What are convolutions and pooling?

>  there's a lot of wasted space in each image. While there are only 784 pixels, it will be interesting to see if there was a way that we could condense the image down to the important features that distinguish what makes it a shoe, or a handbag, or a shirt. That's where convolutions come in

convolution과 pooling은 이미 Andrew 강의에서 공부했으므로 자세한 노트하지 않음.
Convolution과 pooling을 함께 쓰면 매우 powerful해진다.
Convolution은 ` they narrow down the content of the image to focus on specific, distinct, details.`하는 것이다.

> * [Conv2D 설명사이트](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D)   * [MaxPooling2D 설명 사이트](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D) 

### Implementing Convolutional layers (코딩으로 구현하기)

```puthon
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

기본적으로 위의 코드에서 아래로 바뀌게 된다.


```puthon
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
```

* `tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28,28,1))` : 64개의 필터를 만든다. 필터들은  3x3 이고 활성함수는 relu (즉, 음수 값은 모두 날려버리게). input_shape은 28 x 28 이다.  28 x 28 x 1 을 쓴 이유는  color depth로 single byte를 쓴다는 것을 보여준다. (이 데이터는 회색scale임)
* `64개의 필터` : 'you might wonder what the 64 filters are. It's a little beyond the scope of this class to define them, but they aren't random. They start with a set of known good filters in a similar way to the pattern fitting that you saw earlier, and the ones that work from that set are learned over time.' 이 필터들은 무작위는 아니고 좋은 필터들의 세트로 시작해서 시간이 지남에 따레 계속 학습됨.
* 컨볼루션 뉴럴 네트워크 학습 플레이리스트 : [http://bit.ly/2UGa7uH](http://bit.ly/2UGa7uH)


### Implementing pooling layers
* `tf.keras.layers.MaxPooling2D(2,2)` : 2x2pool로 모든 4개의 픽셀마다 가장 큰 1개의 픽셀만 취하게 된다. 

*`tf.keras.layers.Conv2D(64, (3,3), activation='relu'),   
  tf.keras.layers.MaxPooling2D(2,2)` : 그리고 다른 Cov 레이어와 Pool 레이어가 또 따라온다. 그래서 다른 컨브 학습 필터를 거치고 풀링을 통해 사이즈가 줄어들게 된다.

그래서 이미 Dense 레이어로 넘어갈 때 데이터는 아주아주 simplified 된다.

```python
model.summary()
```
: 모델의 레이어들을 점검하게 해준다. 그리고 컨볼루션하는 이미지들의 여정을 결과 모양과 함께 직접 눈으로 확인할 수 있다. It's important to keep an eye on the output shape column. When you first look at this, it can be a little bit confusing and feel like a bug. After all, isn't the data 28 by 28, so y is the output, 26 by 26. --> 가장 가장자리의 픽셀들은 3x3을 쓸 수 없기 때문에 날아간다. 그렇게 위 아래 옆으로 한 줄씩 못쓰고 없어지기 때문에 28 by 28이 아니라 26 by 26이 된다.


### Improving the Fashion classifier with convolutions

### Try it for yourself (Lab 1)   
>[Here](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W3/ungraded_labs/C1_W3_Lab_1_improving_accuracy_using_convolutions.ipynb)’s the notebook that Laurence was using in that screencast. To make it work quicker, go to the ‘Runtime’ menu, and select ‘Change runtime type’. Then select GPU as the hardware accelerator! <br> Work through it, and try some of the exercises at the bottom! It's really worth spending a bit of time on these because, as before, they'll really help you by seeing the impact of small changes to various parameters in the code. You should spend at least 1 hour on this today!  <br>   Once you’re done, go onto the next video and take a look at some code to build a convolution yourself to visualize how it works!   

### Visualizing the Convolutions and Pooling
```python
print(test_labels[:100]
import matplotlib.pyplot as plt
f, axarr = plt.subplots(3,4)
FIRST_IMAGE=0
SECOND_IMAGE=7
THIRD_IMAGE=26
CONVOLUTION_NUMBER = 1
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)
```

**here are some exercises to consider by tweaking codes**

1. Try editing the convolutions(below). Change the 32s to either 16 or 64. What impact will this have on accuracy and/or training time.
--> accuracy는 16에서 0.9825, 64에서 0.9850으로 올라갔다. fillter가 32일 때 제일 높게 나왔다. 계산시간은 16에서가 제일 빠를 것 같은데, 정확히 재본 것이 아니라서 모르겠다.
2. Remove the final Convolution. What impact will this have on accuracy or training time?
--> Convolution이 2개일 때는 test acc = 0.9922 인데 1개로 줄어들면 test acc 도 조금 떨어진다.
3. How about adding more Convolutions? What impact do you think this will have? Experiment with it.
--> 오히려 test acc 가 조금 줄어들었다. (아래 표3 참고)
4. Remove all Convolutions but the first. What impact do you think this will have? Experiment with it.
5. In the previous lesson you implemented a callback to check on the loss function and to cancel training once it hit a certain amount. See if you can implement that here! 
--> ok. got it.

```python
#The convolution
import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10)
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
```

**표로 결과 정리**
1. change the number of fillter   

|Conv 갯수|Conv fillter 갯수|epochs|traim acc|test acc|
|------|---|---|---|---|
|2|16|10|0.9968|0.9899|
|2|32|10|<span style="color:red">0.9977</span>|<span style="color:red">0.9922</span>|
|2|64|10|0.9972|0.9916|

2. delete final conv   

|Conv 갯수|Conv fillter 갯수|epochs|traim acc|test acc|
|------|---|---|---|---|
|1|32|10|<span style="color:red">0.9980</span>|0.9876|
|2|32|10|0.9977|<span style="color:red">0.9922</span>|

3. add more conv layer   

|Conv 갯수|Conv fillter 갯수|epochs|traim acc|test acc|
|------|---|---|---|---|
|1|32|10|<span style="color:red">0.9980</span>|0.9876|
|2|32|10|0.9977|<span style="color:red">0.9922</span>|
|3|32|10|0.9925|0.9851|

5. 
```python
import tensorflow as tf
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('accuracy') > 0.995:
      print("\nArrived to 99.5% accuracy so cancelling training!")
      self.model.stop_training = True

mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0

callbacks = myCallback()

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=10, callbacks = [callbacks])
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(test_acc)
```
### Experiment with filters and pools (Lab 2)

>To try this notebook for yourself, and play with some convolutions, here’s the notebook. Let us know if you come up with any interesting filters of your own! 

[link](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W3/ungraded_labs/C1_W3_Lab_2_exploring_convolutions.ipynb)

>As before, spend a little time playing with this notebook. Try different filters, and research different filter types. There's some fun information about them here: https://lodev.org/cgtutor/filtering.html

## Quiz

1. What is a Convolution?

A technique to make images smaller
A technique to filter out unwanted images
A technique to make images bigger
**A technique to isolate features in images**

2. What is a Pooling?

A technique to isolate features in images
**A technique to reduce the information in an image while maintaining features**
A technique to combine pictures
A technique to make images sharper

3. How do Convolutions improve image recognition?

They make the image smaller
They make processing of images faster
**They isolate features in images**
They make the image clearer

4. After passing a 3x3 filter over a 28x28 image, how big will the output be?

28x28
**26x26**
25x25
31x31

5. After max pooling a 26x26 image with a 2x2 filter, how big will the output be?

28x28
**13x13**
26x26
56x56

6. Applying Convolutions on top of our Deep neural network will make training:

**It depends on many factors. It might make your training faster or slower, and a poorly designed Convolutional layer may even be less efficient than a plain DNN!**
Slower
Faster
Stay the same


# Week 4.
## lectures
### Understanding ImageGenerator   

```python
from tensorflow.keras.preprocessing.image
import ImageDataGenerator

#<train generator>
#It's a common mistake that people point the generator at the sub-directory.
#It will fail in that circumstance. You should always point it at the directory
#that contains sub-directories that contain your images.

train_dagagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
train_dir, #subdirectory 아니고 directory로 제대로 설정
target_size=(300,300), #트레이닝을 위해 input data가 모두 같은 size여야 한다. 그래서 이 옵션 넣어서 로드할 때 알아서 맞춰줄 수 있다.
batch_size = 128, #이미지들은 트레이닝과 벨리데이션에서 batch로 제공된다.
class_mode = 'binary')

#<validation generator>
#training dategen과 동일하지만 한 가지! test images들이 들어있는 subdirectory를 지니고 있는 validation directory를 points at한다.
test_dagagen = ImageDataGenerator(rescale=1./255)
validation_generator = test_datagen.flow_from_directory(
validation_dir,  #이 부분에서 차이!! (역시나 subdirectory아니고 directory로 설정해주어야한다.)
target_size=(300,300),
batch_size=32,
class_mode ='binary')
```

### Defining a ConvNet to use complex images

```python
# Here is the code
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)), #298,298,16,448(param)
  tf.keras.layers.MaxPooling2D(2, 2), #149,149,16,0
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'), #147,147,32,4640
  tf.keras.layers.MaxPooling2D(2,2), #73,73,32,0
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'), #71,71,64,18496
  tf.keras.layers.MaxPooling2D(2,2), #35,35,64,0
  tf.keras.layers.Flatten(), #78400,,,0
  tf.keras.layers.Dense(512, activation='relu'), #512,,,40141312
  tf.keras.layers.Dense(1, activation='sigmoid') #binary에서 쓰기 편리하다. #1,513
])

#total params: 40,165,409
#trainable params:40,165,409
#nontrainable params:0
```

### Training the ConvNet with fit_generator

```python
from tensorflow.keras.optimizers import RMSprop

model.compile(loss = 'binary_crossentropy',
    optimizer = RMSprop(lr=0.001),
    metrics = ['acc'])
```

* now we gonna use `model.fit_generator`, not model.fit : 때문에 우리는 데이터셋 대신에 a generator를 사용할 것이다. (앞서서 본 이미지 제너레이터)

```python
history = model.fit_generator(
    train_generator, #training direc에서 이미지들을 끌어온다.
    steps_per_epoch = 8, #전체 batch 묶음이 몇 개 인가. 1024개의 train 이미지를 128개씩 묶어서 1 batch로 가져오니까 8개 묶음이다.
    epochs = 15, #트레이닝 epochs
    validation_data = validation_generator, #validation set도 어디서 가져오는지 정한다.
    validation_steps = 8, #256개의 validation이미지고 32개씩 묶어서 1 batch를 가져와서 8개 묶음으로 만든다.
                          #'handle them in batches of 32 so we will do 8 steps'
    verbose =2 # how much to display while training is going on
)
```


* Once the model is trained, you will, of course, want to do some predictions on the model.

```python
import numpy as np
from google.colab import files
from keras.preprocessing import image

uploaded = files.upload()

for fn in uploaded.keys():

    #predicting images
    path= '/content/' + fn
    img = image.load_img(path,target_size=(300,300)) #인풋이미지 사이즈가 트레니이할 때와 같게 맞추기
    x=image.img_to_array(img)
    x=np.expand_dims(x,axis=0)

    images=np.vstack([x])
    classes = model.predict(image, batch_size=10)
    print(classes[0])
    if classes[0]>0.5:
        print(fn+"is a human")
    else:
        print(fn + "is a horse")
```

### Walking through developing a ConvNet

### Experiment with the horse or human classifier (Lab 1) 
[**Lab link**](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W4/ungraded_labs/C1_W4_Lab_1_image_generator_no_validation.ipynb)

### Adding automatic validation to test accuracy

### Get hands-on and use validation (Lab 2)
[**Lab link**](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W4/ungraded_labs/C1_W4_Lab_2_image_generator_with_validation.ipynb)

### Exploring the impact of compressing images
* 이미지를 더 작게 줄였을 때의 영향 확인하기
>So we had quite a few convolutional layers to reduce the images down to condensed features. Now, this of course can slow down the training. So let's take a look at what would happen if we change it to a 150 by a 150 for the images to have a quarter of the overall data and to see what the impact would be.

뒷모습의 파란드레스 여자를 150 by 150 픽셀로 줄인 사진에서는 말로 인식하는 오류가 나타남.

> But now, she isn't. This is a great example  of the importance of measuring your training data against a large validation set, inspecting where it got it wrong and seeing what you can do to fix it. Using this smaller set is much cheaper to train, but then errors like this woman with her back turned and her legs obscured by the dress will happen, because we don't have that data in the training set. That's a nice hint about how to edit your dataset for the best effect in training.

즉 train 세트를 더 늘려줘야한다는 이야기인가??


### Get Hands-on with compacted images (Lab 3)
[**Lab link**](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C1/W4/ungraded_labs/C1_W4_Lab_3_compacted_images.ipynb)

## Quiz

1. Using Image Generator, how do you label images?   
It’s based on the directory the image is contained in
2. What method on the Image Generator is used to normalize the image?   
rescale
3. How did we specify the training size for the images?   
The target_size parameter on the training generator
4. When we specify the input_shape to be (300, 300, 3), what does that mean?   
Every Image will be 300x300 pixels, with 3 bytes to define color
5. If your training data is close to 1.000 accuracy, but your validation data isn’t, what’s the risk here?   
You’re overfitting on your training data
6. Convolutional Neural Networks are better for classifying images like horses and humans because   
(1) In these images, the features may be in different parts of the frame
(2) There’s a wide variety of horses
(3) There’s a wide variety of humans
**(4) All of the above**
7. After reducing the size of the images, the training results were different. Why?   
(1)There was more condensed information in the images
(2)The training was faster
(3)There was less information in the images
**(4)We removed some convolutions to handle the smaller images***
