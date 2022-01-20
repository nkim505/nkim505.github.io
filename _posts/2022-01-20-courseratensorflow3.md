---
layout: single #post도 가능
title:  "Deeplearning.Ai TensorFlow Developer (Course 3)"
---

** [알림] **  <br>
💁‍♀️ 텐서플로우 자격증 취득에 도움이 되는 **코세라** 강의 <br>
💻 ["Deeplearning.Ai TensorFlow Developer"](https://www.coursera.org/professional-certificates/tensorflow-in-practice?trk_ref=globalnav) - Course 3 : Natural Language Processing을 듣고 강의 내용을 정리하였습니다.<br>
🧠 수업을 들으며 동시에 정리한 내용이어서(필기노트 대용), 의식의 흐름이 강하게 개입되었습니다.<br>
😚 저만의 이해 방법을 풀어 놓아, 강의와 함께 보시는 분께는 작은 도움이 될 수 있을 것 같습니다.<br>

# Week 1. Sentiment in Text
## Lectures
### Using APIs for NLP

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!'
]

tokenizer = Tokenizer(num_words = 100)
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

#결과: {'i':3, ....., 'dog':4}
```

### Lab 1. tokenize_basic
>  how to tokenize the words and sentences, building up a dictionary of all the words to make a corpus(말뭉치)
* link: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W1/ungraded_labs/C3_W1_Lab_1_tokenize_basic.ipynb

* `sequences=tokenizer.texts_to_sequences(sentences)` 하면, 문장이 encoding된 모양으로 나온다. [1,2,4,5] 이렇게.
근데 이제 fit_on_texts()에서 indexing하지 않은 단어는 안나오게 된다.
* 이로써 알 수 있는 것.
1. 많은 단어량을 확보하기 위해서 아주 큰 training data 가 필요하다.
2. 보지 못한 단어가 나온다면 그냥 무시하는 것이 아니라 어떤 특정 값을 매겨주는 것이 좋아보인다.

### Lab 2. sequences_basic
* Link: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W1/ungraded_labs/C3_W1_Lab_2_sequences_basic.ipynb

```python
import tensorflow as tf
from tensorflow import keras


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences #길이가 다른 문장들을 이용할 때, 
# 다 같은 길이의 문장이 되도록 짧은 것은 padding하거나 긴 것은 truncate 해주는 기능.


sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>") # oov_token: word index에서 인식 안되는 단어를 위해 special token도 만들겠다
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

# 그리고 이제 문장 내의 단어들을 sequences of tokens으로 전환하기 by calling 'texts_to_sequences()' method
sequences = tokenizer.texts_to_sequences(sentences)

padded = pad_sequences(sequences, maxlen=5) # max length를 5 words로, 그러면 4 words 문장은 pre-padded되어 0 * * * * 됨.
# 6 words 문장은 앞에 한 단어가 erase 된다.
print("\nWord Index = " , word_index)
print("\nSequences = " , sequences)
print("\nPadded Sequences:")
print(padded)


# Try with words that the tokenizer wasn't fit to
test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]

test_seq = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequence = ", test_seq)

padded = pad_sequences(test_seq, maxlen=10)
print("\nPadded Test Sequence: ")
print(padded)

```

### Sarcasm, really?

```python
#json 파일을 파이썬 구조로 만드는 라이브러리
import json # allows you to load data in JSON format and automatically create a Python data structure from it.

with open("/sarcasm.json", 'r') as f:
    datastore = json.load(f)

sentences = [] 
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
```
### lab 3. sarcasm
* link: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W1/ungraded_labs/C3_W1_Lab_3_sarcasm.ipynb

```python
word_index = tokenizer.word_index
print(len(word_index)) #인덱스가 몇 개나 만들어졌는지
print(word_index) # order of commonality로 숫자가 매겨진다.
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape) #26709 sentences in the dataset and they are 40 characters long
```
## Quiz

1. What is the name of the object used to tokenize sentences?
-> Tokenizer

1. What is the name of the method used to tokenize a list of sentences?
-> fit_on_texts(sentences)

1. Once you have the corpus tokenized, what’s the method used to encode a list of sentences to use those tokens?
-> texts_to_sequences(sentences)

1. When initializing the tokenizer, how to you specify a token to use for unknown words?
-> oov_token=<Token>

1. If you don’t use a token for out of vocabulary words, what happens at encoding?
-> The word isn’t encoded, and is skipped in the sequence

1. If you have a number of sequences of different lengths, how do you ensure that they are understood when fed into a neural network?
-> Make sure that they are all the same length using the pad_sequences method of the tokenizer

1. If you have a number of sequences of different length, and call pad_sequences on them, what’s the default result?
-> They’ll get padded to the length of the longest sequence by adding zeros to the beginning of shorter ones

1. When padding sequences, if you want the padding to be at the end of the sequence, how do you do it?
-> Pass padding=’post’ to pad_sequences when initializing it

## Optional Assignment- Explore the BBC news archive (해보기)
> For this exercise you’ll get the BBC text archive. Your job will be to tokenize the dataset, removing common stopwords. A great source of these stop words can be found here.
* [BBC text archive](http://mlg.ucd.ie/datasets/bbc.html)
* [A great source of these stop words](https://github.com/Yoast/YoastSEO.js/blob/develop/src/config/stopwords.js)
* [Solution](https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C3/C3/W1/assignment/C3_W1_Assignment_Solution.ipynb)

# Week 2. Word Embedding
> you'll take that to the next step using something called **Embeddings**, that takes these numbers and starts to establish sentiment(정서, 감정) from them, so that you can begin to classify and then later predict texts.
## Lectures
### IMBD review dataset
> You will find here 50,000 movie reviews which are classified as positive of negative. 
* http://ai.stanford.edu/~amaas/data/sentiment/

### Lab1. 
* link: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W2/ungraded_labs/C3_W2_Lab_1_imdb.ipynb

```python
!pip install tensorflow==2.5.0

import tensorflow as tf
print(tf.__version__)

#!pip install -q tensorflow-datasets

import tensorflow_datasets as tfds
imdb, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True) #imdb_reviews가져오기

import numpy as np

train_data, test_data = imdb['train'], imdb['test'] # 각각 나눠줬음

training_sentences = []
training_labels = []

testing_sentences = []
testing_labels = []

# str(s.tonumpy()) is needed in Python3 instead of just s.numpy()
for s,l in train_data:
  training_sentences.append(s.numpy().decode('utf8'))
  training_labels.append(l.numpy())
  
for s,l in test_data:
  testing_sentences.append(s.numpy().decode('utf8'))
  testing_labels.append(l.numpy())
  
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

vocab_size = 10000
embedding_dim = 16
max_length = 120
trunc_type='post'
oov_tok = "<OOV>"


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[3]))
print(training_sentences[3])

### How can we use vectors?
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.Flatten(),
    #tf.keras.layers.GlobalAveragePooling1D(), #도 flatten 대신 사용 가능하다. output shape = 16 (simple하고 조금 더 빠름)
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()


### More into the details
num_epochs = 10
model.fit(padded,
    training_labels_final, 
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels_final))

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape) # shape: (vocab_size, embedding_dim)

#write the vector and metadata auto files.
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()

try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')

# and go to the tensorflow rending projector on projector.tensorflow.org

out_m.close()

```
### Lab2. sarcasm_classifier

* Link: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/20_sep_2021_fixes/C3/W2/ungraded_labs/C3_W2_Lab_2_sarcasm_classifier.ipynb#scrollTo=2HYfBKXjkmU8
<br> <br>  
* TensorFlow Dataset Catalog : https://www.tensorflow.org/datasets/catalog/overview
* TensorFlow Dataset : https://github.com/tensorflow/datasets/tree/master/docs/catalog
* IMDB review Dataset : https://github.com/tensorflow/datasets/blob/master/docs/catalog/imdb_reviews.md
* Subword text Encoder : https://www.tensorflow.org/datasets/api_docs/python/tfds/deprecated/text/SubwordTextEncoder


### Diving into the code (part 2)
subwords 들은 한데 묶어서 순서대로 나열되어야 의미있어지니까 (semantic), 이제까지는 그 위치(ordering)는 중요하게 여기지 않았기 때문에 거기서 나타나는 문제 때문에 loss 가 늘어난다...
이 문제를 해결하기 위해서 주 3차에 recurrent Neural Network 배울 것

### lab3. imdb_subwords
* link: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W2/ungraded_labs/C3_W3_Lab_3_imdb_subwords.ipynb

## Optional Assignment- BBC news archive (해보기2)
* link: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C3/C3/W2/assignment/C3_W2_Assignment.ipynb
* Solution link: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C3/C3/W2/assignment/C3_W2_Assignment_Solution.ipynb

# Week 3. Sequence model
## Lectures
### RNN - Introduction
### RNN 개념 학습 영상
* 앤드류의 Sequence modeling : https://www.coursera.org/lecture/nlp-sequence-models/deep-rnns-ehs0S

### LSTM(Long Short Term Memory)
Today has a beautiful <>
<> might be sky
> In addition to the context being PaaSed as it is in RNNs, LSTMs have an additional pipeline of contexts called cell state. This can pass through the network to impact it. This helps keep context from earlier tokens relevance in later ones so issues like the one that we just discussed can be avoided

> Cell states can also be bidirectional. So later contexts can impact earlier ones as we'll see when we look at the code. 

* LSTM 강의: https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay

### Implementing LSTMs in code (lab1, lab2)
* lab1_single_layer: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W3/ungraded_labs/C3_W3_Lab_1_single_layer_LSTM.ipynb
* lab2_multiple_layer: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W3/ungraded_labs/C3_W3_Lab_2_multiple_layer_LSTM.ipynb

```python
# ONE LAYER SLTM
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']
tokenizer = info.features['text'].encoder
BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_dataset))
test_dataset = test_dataset.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_dataset))
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)), # it will make my cell state go in both directions
# 혹시 LSTM을 두 겹 이상으로 쌓게 되면 첫번째 LSTM 라인에 "return_sequences = True" 해줘야한다. This ensures that the outputs of the LSTM match the desired inputs of the next one.(LSTM의 출력값 모양이 다음 LSTM의 입력모양이랑 서로 맞게 해줌)
 
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
NUM_EPOCHS = 10
history = model.fit(train_dataset, epochs=NUM_EPOCHS, validation_data=test_dataset)
import matplotlib.pyplot as plt

# TWO LAYERS SLTM
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(tokenizer.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences = True))
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```

### Accuracy and loss
> Let's look at the impact of using an LSTM on the model that we looked at in the last module, where we had subword tokens.

>  the comparison of accuracies between the one layer LSTM and the two layer one over 10 epochs. There's not much of a difference except **the nosedive(급강하) of the validation accuracy.**

>  I found from training networks **that jaggedness(들쭉날쭉함) can be an indication that your model needs improvement**, and the single LSTM that you can see here is not the smoothest.

### A word from Laurence
> In this video, you'll see some other options of RNN including convolutions, Gated Recurrent Units also called GRUs, and more on how you can write the code for them. You'll investigate the impact that they have on training

* 실험
> But we can experiment with the layers that bridge the embedding and the dense by removing the flatten and puling from here, and replacing them with an LSTM like this.
* 결과
> Again, this shows some over fitting in the LSTM network. While the accuracy of the prediction increased, the confidence in it decreased. So you should be careful to adjust your training parameters when you use different network types, it's not just a straight drop-in like I did here

### embedding 후에 Conv network 넣기

> ???? :  If we go back to the model and explore the parameters, we'll see that we have 128 filters each for 5 words. And an exploration of the model will show these dimensions. As the size of the input was 120 words, and a filter that is 5 words long will shave off 2 words from the front and back, leaving us with 116. The 128 filters that we specified will show up here as part of the convolutional layer.

### lab 3. multiple_layer_GRU
* link: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W3/ungraded_labs/C3_W3_Lab_3_multiple_layer_GRU.ipynb

일반 방법, RNN- LSTM 방법, RNN-GRU 방법, Conv(+pooling1d) 방법 써봄
대부분 overfitting 문제가 나타남


### lab 4. imdb_reviews_with_GRU_LSTM_Conv1D.
*link: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W3/ungraded_labs/C3_W3_Lab_4_imdb_reviews_with_GRU_LSTM_Conv1D.ipynb

### 오버피팅을 피하기 위한 tips from Laurance
> Try them out for yourself, check on the time, check on the results, and see **what techniques you can figure out to avoid some of the overfitting.** 

> Remember that with text, you'll probably get a bit more overfitting than you would have done with images. Not least because you'll almost always have out of vocabulary words in the validation data set. That is words in the validation dataset that weren't present in the training, naturally leading to overfitting. These words can't be classified and, of course, you're going to have these overfitting issues, but see what you can do to avoid them.

### lab 5 - lab 6
> We've created a number of notebooks for you to explore the different types of sequence model.   Spend some time going through these to see how they work, and what the impact of different layer types have on training for classification.

* Sarcasm with Bidirectional LSTM (lab 5) : https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W3/ungraded_labs/C3_W3_Lab_5_sarcasm_with_bi_LSTM.ipynb

* Sarcasm with 1D Convolutional Layer (lab 6) :  https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W3/ungraded_labs/C3_W3_Lab_6_sarcasm_with_1D_convolutional.ipynb

## Quiz

1. Why does sequence make a large difference when determining semantics of language?
-> Because the order in which words appear dictate their meaning

1. How do Recurrent Neural Networks help you understand the impact of sequence on meaning?
-> They carry meaning from one cell to the next

1. How does an LSTM help understand meaning when words that qualify each other aren’t necessarily beside each other in a sentence?
-> Values from earlier words can be carried to later ones via a cell state

1. What keras layer type allows LSTMs to look forward and backward in a sentence?
-> Bidirectional

1. What’s the output shape of a bidirectional LSTM layer with 64 units?
-> (None, 128)

1. When stacking LSTMs, how do you instruct an LSTM to feed the next one in the sequence?
-> Ensure that return_sequences is set to True only on units that feed to another LSTM

1. If a sentence has 120 tokens in it, and a Conv1D with 128 filters with a Kernal size of 5 is passed over it, what’s the output shape?   
**-> (None, 116, 128)***

1. What’s the best way to avoid overfitting in NLP datasets?
-> None of the above (Use LSTMs, Use GRUs, Use Conv1D)
## Wrap up - week 3.
> You’ve been experimenting with NLP for text classification over the last few weeks. Next week you’ll switch gears -- and take a look at using the tools that you’ve learned to predict text, which ultimately means you can create text. By learning sequences of words you can predict the most common word that comes next in the sequence, and thus, when starting from a new sequence of words you can create a model that builds on them. You’ll take different training sets -- like traditional Irish songs, or Shakespeare poetry, and learn how to create new sets of words using their embeddings!

(해석: 지난 몇 주 동안 텍스트 분류를 위해 NLP를 실험해 왔습니다. 다음 주에는 기어를 바꿔서 학습한 도구를 사용하여 텍스트를 예측하는 방법을 살펴보겠습니다. 이는 궁극적으로 텍스트를 생성할 수 있음을 의미합니다. 단어 시퀀스를 학습하면 시퀀스에서 다음에 오는 가장 일반적인 단어를 예측할 수 있으므로 새로운 단어 시퀀스에서 시작할 때 이를 기반으로 하는 모델을 만들 수 있습니다. 전통적인 아일랜드 노래나 셰익스피어 시와 같은 다양한 훈련 세트를 수강하고 임베딩을 사용하여 새로운 단어 세트를 만드는 방법을 배웁니다!)


## Optional Assignment: Exploring overfitting in NLP (해보기)

* When looking at a number of different types of layer for text classification this week you saw many examples of overfitting -- with one of the major reasons for the overfitting being that your training dataset was quite small, and with a small number of words. Embeddings derived from this may be over generalized also. So for this week’s exercise you’re going to train on a large dataset, as well as using transfer learning of an existing set of embeddings.

* The dataset is from:  https://www.kaggle.com/kazanova/sentiment140. I’ve cleaned it up a little, in particular to make the file encoding work with Python CSV reader. 

* The embeddings that you will transfer learn from are called the GloVe, also known as Global Vectors for Word Representation, available at: https://nlp.stanford.edu/projects/glove/

* Link assignment: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C3/C3/W3/assignment/C3_W3_Assignment.ipynb
* Link Solution : https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C3/C3/W3/assignment/C3_W3_Assignment_Solution.ipynb

# Week 4. Sequence Model and Literature
## Lectures
### Lab 1. Intro for 'Create Texts exercise'
* lab.1 link: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C3/C3/W4/ungraded_labs/C3_W4_Lab_1.ipynb

lil.ai enter
### 로렌스 시 노트북
* 링크: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W4/misc/Laurences_generated_poetry.txt

### 아이리시 가사 노트북
* 링크: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C3/W4/ungraded_labs/C3_W4_Lab_2_irish_lyrics.ipynb

### 문자(character) 기반한 RNN (비교: 단어기반 RNN) : Link to generating text using a character-based RNN
* 링크: https://www.tensorflow.org/text/tutorials/text_generation

## Quiz
1. What is the name of the method used to tokenize a list of sentences?
-> fit_on_texts(sentences)

1. If a sentence has 120 tokens in it, and a Conv1D with 128 filters with a Kernal size of 5 is passed over it, what’s the output shape?
-> (None, 116, 128)

1. What is the purpose of the embedding dimension?
-> It is the number of dimensions for the vector representing the word encoding

1. IMDB Reviews are either positive or negative. What type of loss function should be used in this scenario?
-> Binary crossentropy

1. If you have a number of sequences of different lengths, how do you ensure that they are understood when fed into a neural network?
-> Use the pad_sequences object from the tensorflow.keras.preprocessing.sequence namespace

1. When predicting words to generate poetry, the more words predicted the more likely it will end up gibberish. Why?
-> Because the probability that each word matches an existing phrase goes down the more words you create

1. What is a major drawback of word-based training for text generation instead of character-based generation?
-> Because there are far more words in a typical corpus than characters, it is much more memory intensive

1. How does an LSTM help understand meaning when words that qualify each other aren’t necessarily beside each other in a sentence?
-> Values from earlier words can be carried to later ones via a cell state

## Assignment 
> Optional Assignment - Using LSTMs, see if you can write Shakespeare!    In this course you’ve done a lot of NLP and text processing. This week you trained with a dataset of Irish songs to create traditional-sounding poetry. For this week’s exercise, you’ll take a corpus of Shakespeare sonnets, and use them to train a model. Then, see if that model can create poetry!

* link: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C3/C3/W4/assignment/C3_W4_Assignment.ipynb
* solution: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C3/C3/W4/assignment/C3_W4_Assignment_Solution.ipynb

## Wrap up - week 4.
> Over the last four weeks you've gotten a grounding in how to do Natural Language processing with TensorFlow and Keras. You went from first principles -- basic Tokenization and Padding of text to produce data structures that could be used in a Neural Network.   You then learned about embeddings, and how words could be mapped to vectors, and words of similar semantics given vectors pointing in a similar direction, giving you a mathematical model for their meaning, which could then be fed into a deep neural network for classification.   From there you started learning about sequence models, and how they help deepen your understanding of sentiment in text by not just looking at words in isolation, but also how their meanings change when they qualify one another.   You wrapped up by taking everything you learned and using it to build a poetry generator!   This is just a beginning in using TensorFlow for natural language processing. I hope it was a good start for you, and you feel equipped to go to the next level!

* 해석: 지난 4주 동안 TensorFlow 및 Keras를 사용하여 자연어 처리를 수행하는 방법에 대한 기초를 얻었습니다. 신경망에서 사용할 수 있는 데이터 구조를 생성하기 위해 텍스트의 기본 토큰화 및 패딩이라는 첫 번째 원칙에서 출발했습니다.

그런 다음 임베딩에 대해 배웠고 단어가 벡터에 매핑되는 방법과 유사한 방향을 가리키는 벡터가 제공되는 유사한 의미의 단어를 통해 의미에 대한 수학적 모델을 제공하고 분류를 위해 심층 신경망에 입력할 수 있습니다.

거기에서 시퀀스 모델에 대해 배우기 시작했고, 단어를 개별적으로 살펴보는 것이 아니라 단어가 서로를 수식할 때 의미가 어떻게 변하는지 살펴봄으로써 텍스트의 감정에 대한 이해를 심화하는 데 도움이 되는 방법을 배우기 시작했습니다.

배운 모든 것을 시 생성기를 구축하는 데 사용하여 마무리했습니다!

이것은 자연어 처리를 위해 TensorFlow를 사용하는 시작에 불과합니다. 좋은 시작이 되었기를 바라며 다음 단계로 나아갈 준비가 되었다고 생각합니다! 
