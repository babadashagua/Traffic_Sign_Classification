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
where lr means "learning rate". I have tested with different dropout rates while 0.75 gives me the best performance. To accelerate convergence and stabilize training, I add batch normalization to the two convolutional layers. To avoid dead ReLU, I use ELU instead. These two techniques help to boost validation accuracy to 0.936, which outperforms than that of other configurations. The complete list of hyperparameters is as follows:

Batch size: 128
Epoch: 35
Learning rate: 1e-3
Optimizer: Adam Optimizer

### 3. Model Test on New Images
