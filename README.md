# **Report: Traffic Sign Classification**

---

The goals / steps of this project are the following:
* Design a model structure for traffic sign classification on German traffic sign dataset with high accuracy.
* Test the trained model on new images found on the web.

### Reflection

### 1. Dataset Exploration

The designed model is trained and evaluated on a German traffic sign dataset. It has 43 classes of signs and more than 50,000 images in total (34799 training samples, 4410 validation samples, and 12630 test samples). The image below shows some example images from the dataset.
<p float="left">
  <img src="/Results/dataset_visual.png" width="400" title="Fig. 1 dataset visualization"/>
</p>

### 2. Design and Test a Model Architecture

#### Preprocessing

For model training and test on the given data, no extra preprocessing is employed. For model test on the new images found on the web, the images are processed through the following pipeline:

Convertion (Image.convert( ) returns a Image object) --> Resize (Image.resize( ) and Numpy.reshape ( ) to make the image fit the input of trained network) 

#### Model Architecture

I deploy the classic LeNet structure for this project. It is comprised of five layers: two convolutional layers and three fully connected layers. The network structure is summarized in the table below:
<p float="left">
  <img src="/Results/network_structure.png" width="800" title="Fig. 2 network structure"/>
</p>

#### Model Training and Solution Approach

I have tried network training with different hyperparameters. The figure below shows accuracy performance on the validation set:
<p float="left">
  <img src="/Results/performance_comp.jpg" width="800" title="Fig. 3 model performance"/>
</p>
where "lr" means "learning rate". I have tested with different dropout rates while 0.75 gives me the best performance. To accelerate convergence and stabilize training, I add batch normalization to the two convolutional layers. To avoid dead ReLU, I use ELU instead. These two techniques help to boost validation accuracy to 0.936, which outperforms than that of other configurations. The complete list of hyperparameters is as follows:  

Batch size: 128  
Epoch: 35  
Learning rate: 1e-3  
Optimizer: Adam Optimizer  
  
### 3. Model Test on New Images

I test the model on five images found on the web. The images, test results, and ground truth ("GT") are shown in the following figure:
<p float="left">
  <img src="/Results/test_images.png" width="800" title="Fig. 4 test images and results"/>
</p>
As an observation, when the test images are similar to that in the training dataset (the sign occupies the major area of the image, such as image 1, 2, and 4), the probability of being correctly classified is high. When the sign occupies less area in the image, as shown in image 3 and 5, the probability of misclassification increases. The background information acts as noise and affects the classification. To avoid the background noise and boost the accuracy on new images, I think an object segmentation can be deployed first to extract the sign area, then the sign area is resized and sent to the classifier. 

The accuracy on the new images is 60%, which is relatively lower than the accuracy of 94.5% on the given test data. The top five softmax probabilities on the new images are outputted, as shown below:
<p float="left">
  <img src="/Results/top5_softmax_prob.png" width="800" title="Fig. 5 top five softmax probabilities"/>
</p>
As can be seen from the table above, for the correctly classified signs, the model is very certain of its prediction. For misclassified signs, the model is less certain. 

In addition, I visualize the feature maps of the first convolutional layer (conv1) and second convolutional layer (conv2) when input the image 1.  
Feature maps of conv1:
<p float="left">
  <img src="/Results/conv1.png" width="800" title="Fig. 6 feature maps of conv1"/>
</p>  

Feature maps of conv2:
<p float="left">
  <img src="/Results/conv2.png" width="800" title="Fig. 7 feature maps of conv2"/>
</p>  
As seen in the feature maps of conv1, the feature maps react with high activation to the sign's boundary outline and to the contrast in the sign's painted symbol. The feature maps of conv2 are harder to interpret.  

### 4. Discussion

When tested on other new images (in total I tested 20 images, the accuracy is around 50%), the model accuracy is still not as high as that (94.5%) on the given test data. I attribute this accuracy loss to insufficient rotation and tilt variations in the training data. I have two thoughts to tackle this issue. First, add more rotation and tilt to augment the training data. Second, transform the extracted sign area into "bird's-eye view" through perspective transformation to get rid of the impact due to rotation and tilt in the captured sign images.
