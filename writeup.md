#*Traffic Sign Recognition*

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/exploratory_of_the_data_set.jpg "Visualization"
[image2]: ./output_images/the_RGB_normarized_train_data_set.jpg "Grayscaling"
[image3]: ./output_images/the_RGB_normarized_train_data_set.jpg "Random Noise"
[image9]: ./output_images/test_data_set.jpg

[image4]: ./output_images/Ahead_only.jpg "Traffic Sign 1"
[image5]: ./output_images/Children_crossing.jpg "Traffic Sign 2"
[image6]: ./output_images/General_caution.jpg "Traffic Sign 3"
[image7]: ./output_images/roundabout_mandatory.jpg "Traffic Sign 4"
[image8]: ./output_images/Traffic_signals.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how I addressed each one. This is a template as a guide for writing the report.

Here is a link to my [project code](./P4.ipynb)

### Data Set Summary & Exploration

#### 1. Traffic signe data set summary
I provide a basic summary of the data set. Below is the data set summary oft the Traffic signe signature data set. In the code, the analysis was done using python.
I used the pandas library to calculate summary statistics of the traffic
signs data set:

| Data set Summary   		        | details      			| 
|:---------------------------------:|:---------------------:| 
| The size of training set          | 34799          		|
| The size of the validation set    | 4410                	|
| The size of test set	            | 12630					|
| The shape of a traffic sign image | 32x32x3               |
| The number of unique classes in the data set | 43         |

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It looks like some pictures have different brightness vale.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Pre-Process
I describe how I preprocessed the image data. 

As pre-processing method, I used the normalization techniques. Onece I tried to convert RGB to Gray scale but I don't want to reduce the dimention of feature vector for classification.

So, as a preprocess, I decided to normarize the images without gray scale conversion.
Here is an example of a traffic sign image after normarization.

![alt text][image2]


#### 2. Model Architecture
I describe what my final model architecture looks like including model type, layers, layer sizes, connectivity, etc.
I used LeNet for classification and my final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride,same padding,outputs 28x28x6(32x32x64)|
| RELU					| -												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6(16x16x64)    	|
| Convolution 3x3	    | 1x1 stride,same padding,outputs 14x14x6		|
| RELU					| -												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16               	    |
| Fully connected(fc1)	| outputs 120								    |
| RELU					| -												|
| Drop out		        | keep probability = 0.5						|
| Fully connected(fc2)  | outputs 84							    	|
| RELU					| -												|
| Drop out				| keep probability = 0.5						|
| Fully connected(fc3)  | outputs 10							    	|
| Softmax				| output = 5 (The number of unique classes in the data set) |
|						|												|
|						|												|
 

#### 3. Model parameters
I describe how I trained my model. This discussion includes the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
To train the model, I used LeNet and the parameters in the table below

| Prameters             |     Description	        				    | 
|:---------------------:|:---------------------------------------------:| 
| optimizer       		| AdamOptimizer					                | 
| the batch size     	| 64                                            |
| number of epochs		| 10											|
| learning rate	      	| 0.01                                         	|

#### 4. Model Training Result
This step code is contained in `Train, Validate and Test the Model` section  in `P3.ipynb`.

At first, I used just `LeNet` and the accuracy of validation data set is not good. This seems to be `over fitting` to train data set.  To avoid `over fitting`, I add `Dropout` to the layers.

My final model results are:

| Result                  |     Description         				    | 
|:-----------------------:|:-------------------------------------------:| 
| validation set accuracy |0.95                                         |
| test set accuracy		  |0.939										|

### Test a Model on New Images

#### 1. Overview New Test Images
I choose five German traffic signs found on the web and provide them in this report.

Here are five German traffic signs that I found on the web:

![alt text][image9]

The first image might be difficult to classify because it's looks like fifth image, which has same color (only red, black and white).
In this project, I use collor image as input image and if color is the same, I think my DNN misunderestanding the images.


#### 2. Prediction result for New Images
This section discusses the model's predictions on these new traffic signs and compare the results to predicting on the test set. 

Here are the results of the prediction:

| Image			        |     Prediction	    | Probability		| 
|:---------------------:|:---------------------:|------------------:| 
| Ahead_only      		| Go straight or left   |0.99				| 
| Children_crossing		| Children_crossing 	|0.99				|
| General_caution		| General_caution		|1.0				|
| roundabout_mandatory	| roundabout_mandatory	|0.99		 		|
| Traffic_signals		| No entry      		|0.42			    |

The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. This result is worse compared to that on the test data set of 93.9%.


#### 3. Analysis for the predictions
I describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. 

Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in `Output Top 5 Softmax Probabilities For Each Image Found on the Web` of `P3.ipynb`.

#### For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

#### In 5 German traffic signals, the top five soft max probabilities are:

![alt text][image4]

| Probability   | No  	|     Prediction	        				| 
|:-------------:|:-----:|:-----------------------------------------:| 
| .99         	|18		| General caution 							| 
| .00     		|37		| Go straight or left						|
| .00			|04		| Speed limit (70km/h)						|
| .00	      	|26		| Traffic signals			 				|
| .00			|33	    | Turn right ahead 							|


For the second image ... 

![alt text][image5]

| Probability   | No  	|     Prediction	        				| 
|:-------------:|:-----:|:-----------------------------------------:| 
| .99         	|28		| Children crossing 						| 
| .00     		|29		| Bicycles crossing			    			|
| .00			|30		| Beware of ice/snow						|
| .00	      	|20		| Dangerous curve to the right 				|
| .00			|24	    | Road narrows on the right					|

For the third image ... 

![alt text][image6]

| Probability   | No  	|     Prediction	        				| 
|:-------------:|:-----:|:-----------------------------------------:| 
| 1.0         	|18		| General caution 						    | 
| .00     		|27		| Pedestrians			    			    |
| .00			|24	    | Road narrows on the right					|
| .00	      	|26		| Traffic signals 				            |
| .00			|11	    | Right-of-way at the next intersection		|

For the forth image ... 

![alt text][image7]

| Probability   | No  	|     Prediction	        				| 
|:-------------:|:-----:|:-----------------------------------------:| 
| .99         	|40		| Roundabout mandatory 						    | 
| .00     		|42		| End of no passing by vehicles over 3.5 metric tons			    			    |
| .00			|41	    | End of no passing					        |
| .00	      	|38		| Keep right 				                |
| .00			|13	    | Yield		                                |

For the fifth image ... 

![alt text][image8]

| Probability   | No  	|     Prediction	        				| 
|:-------------:|:-----:|:-----------------------------------------:| 
| .42         	|14		| Stop 						                | 
| .41     		|17		| No entry			    			        |
| .08			|29	    | Bicycles crossing					        |
| .02	      	|26		| Traffic signals 				            |
| .00			|12	    | Priority road		                        |

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


