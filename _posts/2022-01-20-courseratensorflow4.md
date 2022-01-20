---
layout: single #post도 가능
title:  "Deeplearning.Ai TensorFlow Developer (Course 4)"
---

**<<알림>>**<br>
💁‍♀️ 텐서플로우 자격증 취득에 도움이 되는 **코세라** 강의 <br>
💻 ["Deeplearning.Ai TensorFlow Developer"](https://www.coursera.org/professional-certificates/tensorflow-in-practice?trk_ref=globalnav) - Course 4 : Sequences, Time Series and Prediction을 듣고 강의 내용을 정리하였습니다.<br>
🧠 수업을 들으며 동시에 정리한 내용이어서(필기노트 대용), 의식의 흐름이 강하게 개입되었습니다.<br>
😚 저만의 이해 방법을 풀어 놓아, 강의와 함께 보시는 분께는 작은 도움이 될 수 있을 것 같습니다.<br>



# Week 1. Sequences and Prediction
## Lectures
### Train, validation and test sets
* Fixed partitioning: We typically want to split the time series into a training period, a validation period and a test period.

- Here's where you can experiment to find the right architecture for training. And work on it and your hyper parameters, until you get the desired performance, measured using the validation set. Often, once you've done that, you can retrain using both the training and validation data. And then test on the test period to see if your model will perform just as well. And if it does, then you could take the unusual step of retraining again, using also the test data. But why would you do that? Well, it's because the test data is the closest data you have to the current point in time. And as such it's often the strongest signal in determining future values. If your model is not trained using that data, too, then it may not be optimal. 

* roll-forward partitioning: We start with a short training period, and we gradually increase it, say by one day at a time, or by one week at a time. At each iteration, we train the model on a training period. And we use it to forecast the following day, or the following week, in the validation period. 

### Metrics for evaluating performance

```python
errors = forecasts - actual
mse = np.square(errors).mean() # mean squared error
rmse = np.sqrt(mse) # root mean squared error
mae = np.abs(error).mean() # mean absolute error / also called the main absolute deviation or mad / 절대값이 큰 값에게 그렇게 많은 페널라이즈를 부여하지 않는다(마치 mean squared error 가 하는 것 처럼)
mape = # mean absolute percentage error  / mean ratio between 에러의 절대값 and value의 절대값

```
* 업무에 따라서 MAE나 MSE를 쓰는 것을 선호할 수도 있다.
- 예를 들어서 큰 에러가 위험할 가능성이 있다고 하고 그런 큰 에러들이 cost하게 되면 MSE를 쓰는 게 선호될 수 있다.
- 하지만 나의 gain과 loss가 에러 사이즈에 단지 비례한다고 하면, MAE가 낫다.
* mape: value들과 비교한 에러의 사이즈에 대한 인사이트를 준다. (this gives an idea of the size of the errors compared to the values.)

### Moving average and differencing

* 이동 평균(MA) : 설정한 Averaging Window (예를 들어 30일, 1년) 안에서 평균값을 취한다. 

* 차분: series(t) - series(t-365) 해서 추세와 트렌드를 사라지게 만듬. 이후에 Moving average를 구함. 이것은 오리지널 시계열 데이터의 MA가 아닌 차분된 시계열 데이터의 이동평균임.
* To get the final forecasts for the original time series, we just need to add back the value at time T minus 365 : 오리지널 시계열 데이터 추정을 하기 위해서 마지막으로 A_(t-365) 값을 다시 더해줘야한다.
* ** Forcast = moving average of defferenced series + series(t-365)**

* 다시 뺐던 값을 넣어주니까 어느 정도 노이즈는 다시 생겼다. 그러면,   Where does that noise come from?   Well, that's coming from the past values that we added back into our forecasts
-->  we can improve these forecasts by also removing the past noise using a moving average on that. If we do that, we get much smoother forecasts. 
* 이 예측을 더 낫게 만들기 위해서 과거 노이즈를 지움으로써 더 스무스한 예측선을 만들어낼 수 있다.

* ** Forcast = trailing MA of differenced series + centered MA of past series(t-365)**

>  In fact, since the series is generated, we can compute that a perfect model will give a mean absolute error of about four due to the noise. Apparently, with this approach, we're not too far from the optimal. Keep this in mind before you rush into deep learning. Simple approaches sometimes can work just fine.

### Trailing versus centered windows (후행창과 중앙창?)
> Note that when we use the trailing window when computing the moving average of present values from t minus 32, t minus one. But when we use a centered window to compute the moving average of past values from one year ago, that's t minus one year minus five days, to t minus one year plus five days. Then moving averages using centered windows can be more accurate than using trailing windows. But we can't use centered windows to smooth present values since we don't know future values. However, to smooth past values we can afford to use centered windows. 


## Quiz
1. What is an example of a Univariate time series?    hour by hour temperature
1. What is an example of a Multivariate time series?    Hour by hour weather 
1. What is imputed data?    A projection of unknown (usually past or missing) data
1. A sound wave is a good example of time series data    T
1. What is Seasonality?    A regular change in shape of the data
1. What is a trend?    An overall direction for data regardless of direction
1. In the context of time series, what is noise?    
1. What is autocorrelation?    Data that follows a predictable shape, even if the scale is different
1. What is a non-stationary time series?    One that has a disruptive event breaking trend and seasonality 

## Optional Assignment - Create and predict synthetic data
* link: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C4/C4/W1/assignment/C4_W1_Assignment.ipynb
* solution: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C4/C4/W1/assignment/C4_W1_Assignment_Solution.ipynb

# Week 2. Deep Neural Network for Time Series
## Lectures
### Sequence bias
> Sequence bias is when the order of things can impact the selection of things. For example, if I were to ask you your favorite TV show, and listed "Game of Thrones", "Killing Eve", "Travellers" and "Doctor Who" in that order, you're probably more likely to select 'Game of Thrones' as you are familiar with it, and it's the first thing you see. Even if it is equal to the other TV shows. So, when training data in a dataset, we don't want the sequence to impact the training in a similar way, so it's good to shuffle them up. 
한국어로, 시퀀스 편향??? 인가

## Quiz
1.
질문 1
What is a windowed dataset?
- A fixed-size subset of a time series.

2. What does ‘drop_remainder=true’ do?
- It ensures that all rows in the data window are the same length by cropping data

3. What’s the correct line of code to split an n column window into n-1 columns for features and 1 column for a label
- dataset = dataset.map(lambda window: (window[:-1], window[-1:]))


4. What does MSE stand for?
- Mean Squared error
5. What does MAE stand for?
- Mean Absolute Error
6. If time values are in time[], series values are in series[] and we want to split the series into training and validation at time 1000, what is the correct code?

time_train = time[:split_time]

x_train = series[:split_time]

time_valid = time[split_time:]

x_valid = series[split_time:]
7. If you want to inspect the learned parameters in a layer after training, what’s a good technique to use?
- Assign a variable to the layer and add it to the model using that variable. Inspect its properties after training
8. How do you set the learning rate of the SGD optimizer? 
- Use the lr property
9. If you want to amend the learning rate of the optimizer on the fly, after each epoch, what do you do?
- Use a LearningRateScheduler object in the callbacks namespace and assign that to the callback 

## Wrap up
> You've now explored time series data, and seen how to split it into training and validation sets for training a DNN.
> But sequence data tends to work better with RNNs, so next week you're going to look at training a model using RNNs and LSTMs on this data to see what happens!

## Optional Assignment - Predict with a DNN (해보기)
> In class you saw how to split a dataset, and how to start training a DNN using it. For this exercise you’ll create your own synthetic dataset -- I’ve plotted a chart for what it should look like, see if you can figure out the parameters that get this series.   Once you have your series, you’ll create a DNN to predict values for that series

* link: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C4/C4/W2/assignment/C4_W2_Assignment.ipynb
* solution: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C4/C4/W2/assignment/C4_W2_Assignment_Solution.ipynb

# Week 3. Recurrent Neural Networks for Time Series
## Lectures
### LSTM lecture link
* https://www.coursera.org/lecture/nlp-sequence-models/long-short-term-memory-lstm-KXoay


## Quiz
1. If X is the standard notation for the input to an RNN, what are the standard notations for the outputs?
* Y(hat) and H
1. What is a sequence to vector if an RNN has 30 cells numbered 0 to 29
* The Y(hat) for the last cell
1. What does a Lambda layer in a neural network do?
* Allows you to execute arbitrary code while training
1. What does the axis parameter of tf.expand_dims do?
* Defines the dimension index at which you will expand the shape of the tensor 
1. A new loss function was introduced in this module, named after a famous statistician. What is it called?
* Huber loss
1. What’s the primary difference between a simple RNN and an LSTM
* In addition to the H output, LSTMs have a cell state that runs across all cells 
1. If you want to clear out all temporary variables that tensorflow might have from previous sessions, what code do you run?
* tf.keras.backend.clear_session()  
1. What happens if you define a neural network with these two layers?
* Your model will fail because you need return_sequences=True after the first LSTM layer

## Wrap up
> Now that you've built on your DNNs with RNNs and LSTMs, it's time to put it all together using CNNs and some real world data.  Next week you'll do that, exploring prediction of sunspot data from a couple of hundred years worth of measurements!
RNN과 LSTM으로 DNN을 구축했으미로 이제 CNN과 실제 데이터로 모든 것을 통합할 것이다.   다음 주에서는 흑점 데이터로 예측해보면서 배우겠다.

## Assignment -Mean Absolute Error (해보기)
> In class you learned about RNNs and LSTMs for prediction, as well as a simple methodology to pick a decent learning rate for the stochastic gradient descent optimizer. In this exercise you’ll take a synthetic data set and write the code to pick the learning rate and then train on it to get an MAE of < 3

* link: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/25_august_2021_fixes/C4/W3/assignment/C4_W3_Assignment.ipynb
* solution: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/25_august_2021_fixes/C4/W3/assignment/C4_W3_Assignment_Solution.ipynb

# Week 4. Real-world time series data
## Lectures
### Convolutional neural networks course
* link: https://www.coursera.org/learn/convolutional-neural-networks/home/welcome
### More on batch sizing (Mini Batch Gradient Descent(C2W2L01))
* link: https://www.youtube.com/watch?v=4qJaSmvhxi8
### lab1. LSTM
* link: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W4/ungraded_labs/C4_W4_Lab_1_LSTM.ipynb
### Train and tune the model
#### 모델을 훈련시키고 다시 조정하면서 거치는 과정
* `window_size 변경`: 20->132 주기가 11년 정도 되어보여서 주기를 커버하는 정도로 윈도우 사이즈를 변경해 본다.   <br> -> 결과: MAE 더 나빠짐. 효과 없었다. <br> -> 이유? : 윈도우에 시즌 하나를 꽉 채워서 넣을 필요는 없다. (we dont need a full season in our window). 데이터 기간을 더 확대해서 보면 안에도 시즌 1개 안에도 무수히 많은 노이즈가 있고, 이 노이즈의 길이가 한 20 정도 되어 보임
* `window_size 변경2` + `split_time 늘임` : 20 -> 30, 1000 -> 3500. 보니까 train 데이터가 1000 이고 validation이 더 많아서 충분한 train data 가 없었던 것 같다.  <br>  -> 결과: MAE가 더 작게 나와서 효과가 있었다.  <br>  -> 더 향상시킬 수 있을까?
* `뉴런네트워크 디자인을 수정`: 3 layers of 10, 10, and 1 neurons -> 30, 15, 1   <br> ->  결과: 약간 이 전의 상황으로 돌아갔다. MAE가 도리어 올라감.   <br> -> 다시 10, 10, 1로 돌려놓음
* `learning_rate 수정(tweak tweak tweak~)`: MAE getting a little smaller

### Prediction
### Lab 2, lab 3
* Sunspot notebook : https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/adding_C4/C4/W4/ungraded_labs/C4_W4_Lab_2_Sunspots.ipynb#scrollTo=PrktQX3hKYex
*  a version of the notebook that uses only DNN: https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W4/ungraded_labs/C4_W4_Lab_3_DNN_only.ipynb

## Quiz
1. How do you add a 1 dimensional convolution to your model for predicting time series data?
* Use a Conv1D layer type
1. What’s the input shape for a univariate time series to a Conv1D? 
* [None, 1]
1. You used a sunspots dataset that was stored in CSV. What’s the name of the Python library used to read CSVs?
* CSV
1. If your CSV file has a header that you don’t want to read into your dataset, what do you execute before iterating through the file using a ‘reader’ object?
* next(reader)
1. When you read a row from a reader and want to cast column 2 to another data type, for example, a float, what’s the correct syntax?
* float(row[2]) 
1. What was the sunspot seasonality?
* **11 or 22 years depending on who you ask**
1. After studying this course, what neural network type do you think is best for predicting time series like our sunspots dataset?
* A combination of all of the above
1. Why is MAE a good analytic for measuring accuracy of predictions for time series?
* It doesn’t heavily punish larger errors like square errors do

## Optional Assignment - Sunspots
> This week you moved away from synthetic data to do a real-world prediction -- sunspots. You loaded data from CSV and built models to use it. For this week’s exercise, you’ll use a dataset from Jason Brownlee, author of the amazing MachineLearningMastery.com site and who has shared lots of datasets at https://github.com/jbrownlee/Datasets. It’s a dataset of daily minimum temperatures in the city of Melbourne, Australia measured from 1981 to 1990.  Your task is to download the dataset, parse the CSV, create a time series and build a prediction model from it. Your model should have an MAE of less than 2, and as you can see in the output, mine had 1.78. I’m sure you can beat that! :)

* link: https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/25_august_2021_fixes/C4/W4/assignment/C4_W4_Assignment.ipynb
* solution: https://colab.sandbox.google.com/github/https-deeplearning-ai/tensorflow-1-public/blob/25_august_2021_fixes/C4/W4/assignment/C4_W4_Assignment_Solution.ipynb


