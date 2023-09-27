# FitAI-ESE3600
FitAI: A tool that classifies the intensity and type of workouts using heart rate data
Created by: Lars Finlayson and Cheryl Lim

Introduction:

Heart rate data has proven pivotal in many applications, especially in the healthcare industry. For instance, machine learning algorithms can be trained on large datasets of patient health records to predict heart rates and detect abnormalities. These predictions can help doctors diagnose and treat conditions such as arrhythmia, heart failure, and hypertension. It can also be used as a remote monitoring device to monitor patients with chronic heart conditions and alert healthcare providers for abnormalities.

Our machine learning project instead focuses on the athletics industry. Our project aims to classify the types and intensity of athletes' workouts based on the user's heart rate. This system can provide athletes with valuable insights into their cardiovascular health, fitness level, and performance, which can help them optimize their training regimens and achieve their desired outcomes more efficiently and effectively.

We collect data from athletes conducting workouts using a Polar H10 heart rate sensor to train the machine learning model. Data collected belongs to one of the three different workouts – hard workout, interval workout, and steady workout. Once the machine learning model is trained with these data, it can inform the user on the type of workout he/she is doing so the user is aware of their actual exertion compared to their intended level of exertion.
This report will discuss data collection and preprocessing, model architecture, quantization, and deployment.

Motivation:

Heart rate monitoring is an essential tool for athletes in training as it provides valuable insights into the body's response to exercise. When an athlete engages in physical activity, the heart rate increases to supply the necessary oxygen and nutrients to the working muscles. Monitoring the heart rate during exercise helps the athlete maintain a steady and safe level of intensity and prevent overexertion, which can lead to injury or burnout. Moreover, by monitoring their heart rate during workouts, athletes can determine their maximum heart rate and establish their optimal training zones. These zones represent the specific range of heart rate at which an athlete can exercise to achieve specific training goals, such as endurance, strength, or speed. Thus, our project leverages this need by using historical heart rate trends to classify the types of workouts so that users can be informed of their exertions and potential overexertion beforehand. This way, they can adjust their training intensity and prevent the harm caused by overtraining. Overtraining can lead to decreased performance and increased risk of injury. By informing users on potential overexertion before the harm has been made, we are able to prevent overtraining and ensure that each training session is beneficial. 

Overall, the heart rate classification in real time is exciting because it has the potential to revolutionize the way athletes train and optimize their performance. By using machine learning to classify types of workouts based on heart rates, athletes are able to keep track of their exertion without bias from mood, weather, and other external conditions that can disrupt the user’s accurate perception of the workout. Hence, the user can train smarter and achieve their desired outcomes more efficiently and effectively.


Dataset Details:

Data Collection
The data collection process for our machine learning project was paramount, as the data quality and diversity would directly impact our classification model's performance. To ensure the most accurate and representative dataset, we utilized the Polar Flow app in conjunction with the Polar H10 heart rate monitor, allowing us to collect comprehensive heart rate data from various workouts seamlessly.

The initial data collection phase involved consistently wearing the Polar H10 heart rate monitor during various physical activities. The Polar H10's advanced technology provided highly accurate and reliable heart rate measurements, capturing even the subtlest fluctuations in real time. As we participated in different workouts, such as steady state, interval, and hard workouts, the Polar H10 continuously tracked our heart rate, creating a rich and diverse dataset for our project.

Once the workouts were completed, the Polar Flow app synchronized with the Polar H10, automatically storing and organizing the heart rate data for each exercise session. The app's user-friendly interface made reviewing and analyzing the collected data easy, enabling us to identify and select the most representative workouts from each category. These selected workouts formed the basis of our dataset, providing a balanced and comprehensive foundation for our machine-learning model.

The data collection process was a critical component of our project, and the use of Polar Flow and the Polar H10 heart rate monitor proved invaluable in obtaining high-quality heart rate data across various workout categories. By carefully collecting and selecting data that accurately represented each workout type, we ensured that our machine-learning model would effectively classify workouts based on heart rate patterns.



Data Preprocessing:

Interval:

![image](https://github.com/Shasta4208/FitAI-ESE3600/assets/123327124/8e675d59-8bda-4932-87f8-7ace77acc35c)

Steady State:

![image](https://github.com/Shasta4208/FitAI-ESE3600/assets/123327124/603796fb-8a90-46a2-8c68-e8c1f5b88d93)

Hard:

![image](https://github.com/Shasta4208/FitAI-ESE3600/assets/123327124/759af4bb-aaf9-477e-8117-be40c9063953)

Rest (yellow portion only):

![image](https://github.com/Shasta4208/FitAI-ESE3600/assets/123327124/a5b6202b-ff26-4976-b10a-bf8f14b492f1)


The four graphs above give a good representation of the heart rate trends used to train the model to identify the type of workout. We noted that each type of heart rate trend has distinct features. For example, interval workouts have heart rate trends that peak and valley in short periods of time; hard workouts are mostly in the 200 beats per minute range and with sharp increases from low heart rates to 200 beats per minute; steady workouts usually have the heart rate staying at 150 beats per minute with very slight increases across long periods; resting heart rate is represented by the drastic decrease in heart rate from a high heart rate value. 

However, we cannot obtain data from a single workout session that fits perfectly with one category. Each initial data set most likely contains heart rate trends belonging to more than one workout as more than one type of workout is done. Since we were dealing with raw data from daily training regimens, data preprocessing was essential to our project. 

First, we plotted the graphs of all the heart rate data obtained and manually separated and labeled the different sections. An example is shown below.

![image](https://github.com/Shasta4208/FitAI-ESE3600/assets/123327124/9cba67c5-fb46-4d9d-99f9-9020074929bc)
 
Yellow: Interval
Pink: Rest
Green: Hard

By manually splitting workout data to separate resting heart rate and warm-up heart rate, we keep only relevant data in the dataset of each category. This helps the model to learn the characteristics of each category more effectively.

The next part of preprocessing is to ensure that all data in the dataset has the same length, n. This is because we want our final product to be able to predict the type of workout every n seconds and then give a percentage of the type of workouts done in that single training session instead of just providing one prediction for the entire workout. Since the user may do several types of workouts in each training session, we want the model to be able to capture that and give a more useful measure of exertion. 

The length (n) chosen also plays an important role in model accuracy, this is because there is a tradeoff between using a small value of n, which ensures that the system predicts at a more frequent rate, and also allows for lower dimensionality data; versus a larger n value which is able to capture more features of the workout that distinguishes between the classes. After experimenting with different lengths, we found that the length of n = 80 gives the best model accuracy. Hence, each set of data is split up into lengths of 80 samples. Through this step, we changed the dataset from having 165 sets of data of varying lengths of samples to 724 sets of data with 80 samples each. Next, we split the 724 data sets into training, validation, and test sets, with 70% as training, 15% as validation, and 15% as test. 



Model Choice/Design

Our final model has an architecture of:

model = Sequential()

model.add(Input(shape=(data.shape[1], 1)))

model.add(LSTM(units=32,))

model.add(Dropout(rate=0.2))

model.add(Dense(units=16, activation='relu'))

model.add(Dense(4, activation='softmax'))

An important feature of this model is the use of an LSTM layer. We chose to use an LSTM model because our project aims to classify a sequence of time series data, in which the dependencies between data points are crucial information for the model to pick up. LSTMs are able to learn long-term dependencies between time steps of data and hence explain the significantly higher accuracy achieved compared to other models. We included the dropout layer to prevent overfitting. Due to the large dimensionality of our input data, we have to train the model for many epochs. The dropout layer prevents the model from overtraining. The two dense layers follow the dropout layer, which helps the model find relationships between the data values output by the LSTM layer. The output of the model is an array of prediction probabilities of all four classes, in which the class with the highest probability is the model’s prediction.

The initial version of this model consists of the same layers but with a much larger number of units in a layer. The LSTM and Dense layers comprised 128 units, and the dropout parameter was 0.5. This model also had approximately the same accuracy but was too large to be quantized and deployed. We realized that lowering the number of units gives a significant decrease in model size while the model maintains good accuracy. Hence, in order to obtain a model as small as possible, we decreased the number of units per layer until the model no longer performs well.

Other versions of models that were tested include a 1-dimensional convolution model, a single-layer RNN model, and models consisting of only dense layers. We tested these models because they are much smaller and easier to quantize. However, these models cannot learn the data well, and training accuracy will only stop increasing at about 50%. Hence, despite the small size of these models, the accuracies achieved were not sufficiently high.


Model Training and Evaluation Results 

We trained our final model for 1000 epochs, with a learning rate of 0.001 and ‘adam’ as the optimizer. We aimed to train the model until the training and validation accuracies were somewhat equal to each other to ensure optimal training without overfitting. Hence, 1000 is the ideal number of epochs.

The training and validation accuracy is consistently at least 80% or more. The testing accuracy is 75%. The graph for training and validation accuracy is shown below. 

![image](https://github.com/Shasta4208/FitAI-ESE3600/assets/123327124/090cc7ab-510d-4dc6-b90e-e6c7103548dd)

To better understand the classification accuracies in each class, we referred to the confusion matrix below.

![image](https://github.com/Shasta4208/FitAI-ESE3600/assets/123327124/7207ef92-70b4-4615-acc6-069be5b5cf13)

In our project application, we are not too particular about having 100% accuracy in predicting a specific class. For example, interval workouts with slightly longer intervals can also be very similar to a hard workout in intensity. The most crucial aspect that we are focusing on is the ability of the model to distinguish a steady workout from an interval and a hard workout. From the confusion matrix, we observe that the model is able to do that decently well. Now that the model is able to predict 80-second workouts individually, we moved on to test the model according to its real-life functionality.

Our system aims to take in a full arbitrary long workout’s heart rate data, and when feeding it into the system, the model will run inference on every 80 seconds of heart rate data. The user then can get an update on their exertion level by knowing their predicted workout type every 80 seconds. The number of each class predicted is tracked by the system, and the percentage of the classes is then reported to the user. Hence, at the end of the workout, the user will get a summary of how much of a certain class of workout they did in order to determine their level of exertion. 
To test this functionality, we input a 10-minute steady workout. As shown in the picture below, the model ran an inference every 80 seconds, predicted the workout as steady throughout the 10 minutes, and correctly predicted the last 80 seconds as rest because the workout had ended before that.
 
![image](https://github.com/Shasta4208/FitAI-ESE3600/assets/123327124/984ed27f-6101-40a9-a179-8726c7f77f58)

Deployment/Hardware Details

The deployment of our machine learning project, which focused on classifying workouts into different categories based on heart rate data, involved integrating various hardware components to create a portable, low-memory, and efficient system. Our chosen platform for deploying the classification model was the Arduino Nano 33 BLE, a microcontroller board with built-in Bluetooth Low Energy (BLE) capabilities. The Arduino Nano 33 BLE's compatibility with a wide range of sensors and ability to run complex models made it an ideal choice for our project, as it allowed us to interface with the Polar H10 heart rate monitor and execute the trained machine-learning model efficiently. We managed to quantize the model down from 879834 bytes to 17552 bytes, which is small enough for deployment in Arduino Nano 33 BLE. 
We relied on the Polar H10 monitor to collect the heart rate data, which measures heart rate during various physical activities. This device was used to transmit the collected data wirelessly to our Arduino Nano 33 BLE system, which was then processed in real time by the deployed machine learning model. The BLE communication protocol facilitated data transmission between the devices, ensuring minimal latency and power consumption.

Furthermore, we utilized the Polar Flow app to manage and organize the historical heart rate data and synchronize the data with the Polar H10 heart rate monitor. The app's extensive data representation and user-friendly interface streamlined our data management process, enabling us to access and analyze the heart rate data as needed quickly.

The successful deployment of our machine learning project onto the Arduino Nano 33 BLE, in conjunction with the Polar H10 heart rate monitor and the Polar Flow app, resulted in a portable and effective system for classifying workouts based on heart rate data. 

Challenges/ Future Work

Throughout the development and deployment of our machine learning project, which aimed to classify workouts into different categories based on heart rate data, we faced several challenges that tested our problem-solving abilities and perseverance. One of the primary hurdles was the unstable Bluetooth connection between the Polar H10 heart rate monitor and the Arduino Nano 33 BLE. Although the devices could sometimes establish a connection, it was often for a short period, lasting only a few seconds before disconnecting and being unable to be found again for some time. This issue impacted the real-time processing of heart rate data and the overall functionality of our system.

Another significant challenge was compressing our LSTM (Long Short-Term Memory) model to fit onto the Arduino Nano 33 BLE. The limited storage capacity of the microcontroller board required us to optimize the size of our model without compromising its performance. This process involved meticulous experimentation with different compression techniques and hyperparameter tuning to find the most suitable configuration.

Moving forward, we have identified some promising paths forward for future work to overcome these challenges and improve the performance of our project. One potential solution is to develop a dedicated mobile app for running the machine learning model, which could provide a more stable connection to the heart rate monitor and facilitate real-time data processing. Alternatively, we could explore using a wired heart rate monitor that directly connects to the Arduino, thereby ensuring a consistent and uninterrupted data flow to the model.

By addressing these challenges and further refining our system, our machine learning project has the potential to impact fitness and workout personalization significantly. By providing users with a reliable and accurate method for classifying workouts based on heart rate data, we can contribute to the optimization of individual fitness regimens and overall well-being.

Link to code on Colab can be found here
https://colab.research.google.com/drive/1P4koZnSms98NK1HSYhbVrSF6MWYDnSrI?usp=sharing

Full Dataset can be found here
https://drive.google.com/drive/folders/1-FwGB54RY1K-Z3c_LfaJBtYFOuXHH4v_?usp=sharing
