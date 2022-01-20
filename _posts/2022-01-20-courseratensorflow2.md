---
layout: single
title: "Deeplearning.Ai TensorFlow Developer (Course 2)"
---

** [알림] **  <br>
💁‍♀️ 텐서플로우 자격증 취득에 도움이 되는 **코세라** 강의 <br>
💻 ["Deeplearning.Ai TensorFlow Developer"](https://www.coursera.org/professional-certificates/tensorflow-in-practice?trk_ref=globalnav)를 듣고 강의 내용을 정리하였습니다.<br>
🧠 수업을 들으며 동시에 정리한 내용이어서(필기노트 대용), 의식의 흐름이 강하게 개입되었습니다.<br>
😚 저만의 이해 방법을 풀어 놓아, 강의와 함께 보시는 분께는 작은 도움이 될 수 있을 것 같습니다.<br>

# Week 1.
## Lectures
### Looking at the notebook (Lab 1)
[Link](https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W1/ungraded_lab/C2_W1_Lab_1_cats_vs_dogs.ipynb)

### Looking at accuracy and loss
```python
history = model.fit_generator(
...
verbose = 2)

acc = history.history['acc']
val_acc = history.history['val_acc']

epochs = range(len(acc))

plt.plot(epochs, acc)
plt.plot(epochs, val_acc)
plt.title('Training and Validation accuracy')

plt.figure()

plt.plot(epochs, loss)
plt.plot(epochs, val_loss)
plt.title('Training and Validation loss')
```
> 마지막 비디오의 끝에서 훈련 히스토리를 탐색하는 방법을 보았고 흥미로운 현상을 발견했습니다. 훈련 데이터 세트의 정확도가 매우 높았지만 몇 에포크 후에 검증 세트가 평준화되는 것을 보았습니다. 이것은 우리가 다시 과대적합되었다는 분명한 신호입니다. 더 많은 데이터를 사용하면 이에 도움이 되지만 더 작은 데이터 세트에도 사용할 수 있는 몇 가지 다른 기술이 있습니다. 그리고 우리는 다음 주 수업에서 그들을 볼 것입니다!

## Quiz
1. What does flow_from_directory give you on the ImageGenerator?   
The ability to easily load images for training   
The ability to pick the size of training images   
The ability to automatically label images based on their directory name   
All of the above   
2. If my Image is sized 150x150, and I pass a 3x3 Convolution over it, what size is the resulting image?   
148*148
3. If my data is sized 150x150, and I use Pooling of size 2x2, what size will the resulting image be?   
75x75
4. If I want to view the history of my training, how can I access it?   
Create a variable ‘history’ and assign it to the return of model.fit or model.fit_generator   
5. What’s the name of the API that allows you to inspect the impact of convolutions on the images?    
The model.layers API
6. When exploring the graphs, the loss leveled out at about .75 after 2 epochs, but the accuracy climbed close to 1.0 after 15 epochs. What's the significance of this?   
There was no point training after 2 epochs, as we overfit to the training data
7. Why is the validation accuracy a better indicator of model performance than training accuracy?   
The validation accuracy is based on images that the model hasn't been trained with, and thus a better indicator of how the model will perform with new images.
8. Why is overfitting more likely to occur on smaller datasets?   
Because there's less likelihood of all possible features being encountered in the training process.

# Week 2.
## Lectures
* https://github.com/keras-team/keras-preprocessing
* Image data preprocessing function들 설명 사이트: https://keras.io/api/preprocessing/image/
### Coding augmentation with ImageDataGenerator

```python
train_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40, #랜덤으로 로테이트
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2, #줌인 비율
      horizontal_flip=True, #가로로 뒤집기
      fill_mode='nearest') 
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

validation_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
validation_generator = validation_datagen.flow_from_directory(VALIDATION_DIR,
                                                              batch_size=100,
                                                              class_mode='binary',
                                                              target_size=(150, 150))

```
### Lab 1: cats_v_dogs_augmentation
link: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W2/ungraded_labs/C2_W2_Lab_1_cats_v_dogs_augmentation.ipynb

### Demonstrating overfitting in cats vs. dogs
> Meanwhile, the validation topped out at around 70 percent, and that's overfitting clearly been demonstrated. In other words, the neural network was terrific at finding a correlation between the images and labels of cats versus dogs for the 2,000 images that it was trained on, but once it tried to predict the images that it previously hadn't seen, it was about 70 percent accurate. It's a little bit like the example of the shoes we spoke about earlier. 

### The impact of augmentation on Cats vs. Dogs
> Now that we’ve seen it overfitting, let’s next look at how, with a simple code modification, we can add Augmentation to the same Convolutional Neural Network to see how it gives us better training data that overfits less!
이런 트레이닝데이터의 오버피팅 문제를 이미지 augmentation으로 어느 정도 완화 가능하다.

### Adding augmentation to cats vs. dogs
augmentation으로 트레이닝데이터의 acc이 조금 떨어졌지만 vali set의 결과는 전보다 나아져서 오버피팅의 문제를 어느정도 해소했다는 사인을 볼 수 있다.

### Lab2 :horses_v_humans_augmentation
link: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W2/ungraded_labs/C2_W2_Lab_2_horses_v_humans_augmentation.ipynb

### Exploring augmentation with horses vs. humans
> So by the time the training has completed, we can see the same pattern. The training accuracy is trending towards 100 percent, but the validation is fluctuating in the 60s and 70s. Let's plot this, we can see that the training accuracy climbs steadily in the way that we would want, but the validation fluctuated like crazy. So what we can learn from this is that the image augmentation introduces a random element to the training images but if the validation set doesn't have the same randomness, then its results can fluctuate like this. **So bear in mind that you don't just need a broad set of images for training, you also need them for testing or the image augmentation won't help you very much.**

training 에 augmentation줄 때 test에도 동일하게 해줘야 효력이 있다.

## Quiz
3. When training with augmentation, you noticed that the training is a little slower. Why?   
Because the image processing takes cycles
4. What does the fill_mode parameter do?   
It attempts to recreate lost information after a transformation like a shear
6. How does Image Augmentation help solve overfitting?   
It manipulates the training set to generate more scenarios for features in the images
8. Using Image Augmentation effectively simulates having a larger data set for training.   
True   




# Week 3.
## Lectures
> you'll learn how to implement transfer learning and use it to get your models to not only train faster but also get higher accuracy. 

### Understanding transfer learning: the concepts
전이학습의 개념: 훨씬 더 많은 데이터로 학습된 기존 모델을 가져와서 해당 모델이 학습한 기능을 사용하는 것.   
기존 이미지에 너무 specialized 되어 있을수 있으므로 일부 하위 항목도 다시 훈련하도록 선택할 수 있음.   
올바른 조합을 찾는데는 어느정도 some trial and error 가 필요함.   

### Lab1: Transfer learning
Link: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W3/ungraded_lab/C2_W3_Lab_1_transfer_learning.ipynb
> For more on how to freeze/lock layers, explore the documentation, which includes an example using MobileNet architecture: [https://www.tensorflow.org/tutorials/images/transfer_learning](https://www.tensorflow.org/tutorials/images/transfer_learning)

### Coding transfer learning from the inception mode
* 기존의 모델에서 어떻게 the layers를 가져오고 그 레이어들이 내 모델 안에서 다시 재훈련되지 않도록 freeze하는 법
```python
import os

from tensorflow.keras import layers # to pick at the layers and understand which ones we want to use and which ones we want to retrain
from tensorflow.keras import Model

# A copy of the pretrained weights for the inception neural network is saved at this URL
!wget --no-check-certificate \
    https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
    -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5

from tensorflow.keras.applications.inception_v3 import InceptionV3

local_weights_file = '/tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), #내 데이터의 shape
                                include_top = False, # inception_v3는 fully-connected layer가 맨 위에 있는데 False해서 그걸 무시하고 곧바로 convolutions으로 접근한다고 하는 것. fully-connected layer: 한 층의 모든 뉴런이 다음 층의 모든 뉴런과 연결되어 있는 layer (Dense layer라고도 함)
                                weights = None) # built-in weights은 쓰지 않겠다.

pre_trained_model.load_weights(local_weights_file)

# 핵심: Lock them with the 2 lines
for layer in pre_trained_model.layers: #layers들을 iterate해서 부르는데 얘네들을 다 trainable하지 않게 선언(?)
  layer.trainable = False
# pre_trained_model.summary() #print summary해 볼 수 있음.
```

### Adding your DNN
* 이제 위에는 기존 Conv레이어들을 freeze시켰다면, 그 아래로 own DNN 을 추가해서 내 데이터에 retrain할 수 있게 하는 법을 보자.

### Coding your own model with transferred features
```python
last_layer = pre_trained_model.get_layer('mixed7') #중간에 있는 'mixed7' 레이어를 콕 찝어서 그 레이어를 마지막 레이어로 삼을 수 있다. 
print('last layer output shape: ', last_layer.output_shape) # 'mixed7'의 모양 확인해주고.
last_output = last_layer.output # 'mixed7를 output으로 지정함.

from tensorflow.keras.optimizers import RMSprop

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output) # 'mixed7'에서 이미 배운 Dense model으로 넘어가게 됨.
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense  (1, activation='sigmoid')(x)           

model = Model(pre_trained_model.input, x) 

model.compile(optimizer = RMSprop(learning_rate=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
```

### Using dropouts!
Another useful tool to explore at this point is the Dropout. 
> The idea behind Dropouts is that they remove a random number of neurons in your neural network. This works very well for two reasons: The first is that neighboring neurons often end up with similar weights, which can lead to overfitting, so dropping some out at random can remove this. The second is that often a neuron can over-weigh the input from a neuron in the previous layer, and can over specialize as a result. Thus, dropping out can break the neural network out of this potential bad habit!    Check out Andrew's terrific video explaining dropouts here:   https://www.youtube.com/watch?v=ARq74QuavAo

* 전이학습된 모델로 고양이 개 분류기 코드를 마저 만들어서 model.fit_generator까지 수행해보자.

```python
# 배운대로 데이터를 어떻게 생성할지 만들어서
train_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40, 
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True, 
      fill_mode='nearest') 

#데이터를 가져와서
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
#만들어논 모델로 fit한다.
history = model.fit_generator(
                    train_generator,
                    validation_data=validation_generator,
                    #등등
                    verbose = 2)
```

근데, 이렇게 코드를 실행하면 overfitting되어 vali set 결과는 낮은 값으로 계속 분산하는 걸 볼 수 있다.
이미 전이학습된 모델을 사용했는데 과적합이 일어났다. 이런 과적합을 
이것을 해결하기 위한 방법 중 하나가, 신경망에서 임의의 수의 뉴런을 제거하는 **dropout**
> dropout 작동 이유 <br> 1. 이웃한 뉴런이 비슷한 가중치로 끝나면서 좌적합되는데 무작위로 일부 제가하면 이를 피할 수 있음. <br> 2. 뉴런이 이전 계층의 뉴런에서 입력된 값을 과도하게 사욯하여 결과적으로 과도하게 specialized 될 수 있는데 이를 완화함.

### Exploring dropouts
위에서 전이모델을 사용하고, 또 augmentation도 사용했다. Despite that, 과적합이 나타났는데 dropout으로 완화하자.

```python
x = layers.Flatten()(last_output) # 'mixed7'에서 이미 배운 Dense model으로 넘어가게 됨.
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x) #dropping out 20% of neurons
x = layers.Dense  (1, activation='sigmoid')(x)  
```

### Lab 1. Applying Transfer Learning to Cats v Dogs (Lab 1)
https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C2/C2/W3/ungraded_labs/C2_W3_Lab_1_transfer_learning.ipynb#scrollTo=BMXb913pbvFg

### What we've learnt here
*Transfer Learning을 보고 기존 모델을 사용하여 많은 레이어를 고정하여 해당 모델이 재훈련되지 않도록 하고 이미지에 맞게 훈련된 컨볼루션을 효과적으로 '기억'하는 방법을 보았습니다.

*그런 다음 이 아래에 고유한 DNN을 추가하여 다른 모델의 컨볼루션을 사용하여 이미지를 다시 학습할 수 있습니다.

*네트워크를 더 효율적으로 만들기 위해 드롭아웃을 사용하는 정규화에 대해 배웠습니다.


## Quiz
1. If I put a dropout parameter of 0.2, how many nodes will I lose?
-> 20% of them

1. Why is transfer learning useful?
-> Because I can use the features that were learned from large datasets that I may not have access to

1. How did you lock or freeze a layer from retraining?
-> layer.trainable = false

1. How do you c1hange the number of classes the model can classify when using transfer learning? (i.e. the original model handled 1000 classes, but yours handles just 2)
-> When you add your DNN at the bottom of the network, you specify your output layer with the number of classes you want

1. Can you use Image Augmentation with Transfer Learning Models?
-> Yes, because you are adding new layers at the bottom of the network, and you can use image augmentation when training these

1. Why do dropouts help avoid overfitting?
-> Because neighbor neurons can have similar weights, and thus can skew the final training

1. What would the symptom of a Dropout rate being set too high?
-> The network would lose specialization to the effect that it would be inefficient or ineffective at learning, driving accuracy down

1. Which is the correct line of code for adding Dropout of 20% of neurons using TensorFlow
-> tf.keras.layers.Dropout(0.2),

### Error Found
* `Callback error: '>' not supported between instances of 'NsoneType' and 'float'`   
->  use logs.get('accuracy') inplace of logs.get('acc')

# Week 4.
## Lectures
> So in this week, you get to play with this cool new data set and use it to practice implementing building multi-class classifiers in TensorFlow. Please dive in.

### Introducing the Rock-Paper-Scissors dataset
http://www.laurencemoroney.com/rock-paper-scissors-dataset/
>Rock Paper Scissors is a dataset containing 2,892 images of diverse hands in Rock/Paper/Scissors poses. It is licensed CC By 2.0 and available for all purposes, but it’s intent is primarily for learning and research.

>Rock Paper Scissors contains images from a variety of different hands,  from different races, ages and genders, posed into Rock / Paper or Scissors and labelled as such. You can download the training set here, and the test set here. These images have all been generated using CGI techniques as an experiment in determining if a CGI-based dataset can be used for classification against real images. I also generated a few images that you can use for predictions. You can find them here.

>Note that all of this data is posed against a white background.   Each image is 300×300 pixels in 24-bit color

### Explore multi-class with Rock Paper Scissors dataset
```python
# 1st change
train_datagen = ImageDataGenerator(rescale=1./255) 
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(300, 300),
                                                    batch_size=128,
                                                    class_mode='categorical')

# Next change
model =tf.keras.models.Sequential([

    tf.keras.layers.Dense(3, activation = 'softmax')])
 
# activation = 'softmax' -> 하나의 사진이 3가지 경우에 대해서 0-1 사이의 확률로 표시된다.
# 예) Rock: 0.001, Paper: 0.647, Scissors: 0.352


# Final change
from tensorflow.keras.optimizers import RMSprop

model.compile(loss ='categorical_crossentropy', #loss func을 바꿔준다. scarse categoricall crossentropy도 있다.
              optimizer=RMSprop(lr=0.001),
              metrics = ['acc'] )
```

### Lab1: multi_class_classifier
link: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C2/W4/ungraded_lab/C2_W4_Lab_1_multi_class_classifier.ipynb

```python

# 먼저, get the data
# rps training set
!gdown --id 1DYVMuV2I_fA6A3er-mgTavrzKuxwpvKV
  
# rps testing set
!gdown --id 1RaodrRK1K03J_dGiLu8raeUynwmIbUaM

# import 라이브러리들 & 데이터 zip 풀기
import os
import zipfile

local_zip = './rps.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/rps-train')
zip_ref.close()

local_zip = './rps-test-set.zip'
zip_ref = zipfile.ZipFile(local_zip, 'r')
zip_ref.extractall('tmp/rps-test')
zip_ref.close()

# 하위 파일들 보기
base_dir = 'tmp/rps-train/rps'

rock_dir = os.path.join(base_dir, 'rock')
paper_dir = os.path.join(base_dir, 'paper')
scissors_dir = os.path.join(base_dir, 'scissors')

print('total training rock images:', len(os.listdir(rock_dir)))
print('total training paper images:', len(os.listdir(paper_dir)))
print('total training scissors images:', len(os.listdir(scissors_dir)))

rock_files = os.listdir(rock_dir)
print(rock_files[:10])

# 플롯으로 어떻게 생겼는지 보기
%matplotlib inline

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

pic_index = 2

next_rock = [os.path.join(rock_dir, fname) 
                for fname in rock_files[pic_index-2:pic_index]]
next_paper = [os.path.join(paper_dir, fname) 
                for fname in paper_files[pic_index-2:pic_index]]
next_scissors = [os.path.join(scissors_dir, fname) 
                for fname in scissors_files[pic_index-2:pic_index]]

for i, img_path in enumerate(next_rock+next_paper+next_scissors):
  #print(img_path)
  img = mpimg.imread(img_path)
  plt.imshow(img)
  plt.axis('Off')
  plt.show()

# 모델 만들어서 돌리기
import tensorflow as tf
import keras_preprocessing
from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator

TRAINING_DIR = "tmp/rps-train/rps"
training_datagen = ImageDataGenerator(
      rescale = 1./255,
	    rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

VALIDATION_DIR = "tmp/rps-test/rps-test-set"
validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = training_datagen.flow_from_directory(
	TRAINING_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

validation_generator = validation_datagen.flow_from_directory(
	VALIDATION_DIR,
	target_size=(150,150),
	class_mode='categorical',
  batch_size=126
)

model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 150x150 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])


model.summary()

model.compile(loss = 'categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

history = model.fit(train_generator, epochs=25, steps_per_epoch=20, validation_data = validation_generator, verbose = 1, validation_steps=3)

model.save("rps.h5")

# acc결과 그래프로 그리기
import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()

paper_files = os.listdir(paper_dir)
print(paper_files[:10])

scissors_files = os.listdir(scissors_dir)
print(scissors_files[:10])

```

### What have we seen so far?
> You're coming to the end of Course 2, and you've come a long way! From first principles in understanding how ML works, to using a DNN to do basic computer vision, and then beyond into Convolutions.   With Convolutions, you then saw how to extract features from an image, and you saw the tools in TensorFlow and Keras to build with Convolutions and Pooling as well as handling complex, multi-sized images.   Through this you saw how overfitting can have an impact on your classifiers, and explored some strategies to avoid it, including Image Augmentation, Dropouts, Transfer Learning and more. To wrap things up, this week you've looked at the considerations in your code that you need for moving towards multi-class classification! 

## Quiz
1. Thediagram for traditional programming had Rules and Data In, but what came out?
-> Answers

1. Why does the DNN for Fashion MNIST have 10 output neurons?
-> The dataset has 10 classes

1. What is a Convolution?
-> A technique to extract features from an image

1. Applying Convolutions on top of a DNN will have what impact on training?
-> It depends on many factors. It might make your training faster or slower, and a poorly designed Convolutional layer may even be less efficient than a plain DNN!

1. What method on an ImageGenerator is used to normalize the image?
-> rescale

1. When using Image Augmentation with the ImageDataGenerator, what happens to your raw image data on-disk.
-> Nothing

1. Can you use Image augmentation with Transfer Learning?
-> Yes. It's pre-trained layers that are frozen. So you can augment your images as you train the bottom layers of the DNN with them

1. When training for multiple classes what is the Class Mode for Image Augmentation?
-> class_mode='categorical'
