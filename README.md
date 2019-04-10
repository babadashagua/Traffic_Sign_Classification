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

Convertion (Image.convert() returns a Image object) --> Resize (Image.resize() and Numpy.reshape () to make the image fit the input of trained network) 

#### Model Architecture



#### Model Training

#### Solution Approach

### 3. Model Test on New Images
