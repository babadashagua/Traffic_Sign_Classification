# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 16:04:04 2019

@author: yu
"""

import tensorflow as tf
import pickle
import numpy as np
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt
import os
from PIL import Image
import csv

MODEL_SAVE = "models/epoch35/model.ckpt"
IMG_SIZE = 32

# training, validation, and test data
training_data = 'data/train.p'
valid_data = 'data/valid.p'
test_data = 'data/test.p'

with open(training_data, mode='rb') as f:
    train = pickle.load(f)
    
with open(valid_data, mode='rb') as f:
    valid = pickle.load(f)
    
with open(test_data, mode='rb') as f:
    test = pickle.load(f)

x_train, y_train = train['features'], train['labels']
x_valid, y_valid = valid['features'], valid['labels']
x_test, y_test = test['features'], test['labels']

# check if the numbers of features and labels match
assert(len(x_train) == len(y_train))
assert(len(x_valid) == len(y_valid))
assert(len(x_test) == len(y_test))


print("Image Shape: {}".format(x_train[0].shape))
print("Training Set:   {} samples".format(len(x_train)))
print("Validation Set: {} samples".format(len(x_valid)))
print("Test Set:       {} samples".format(len(x_test)))


# display one image
index = random.randint(0, len(x_train))
image = x_train[index].squeeze()

plt.figure(figsize=(1,1))
plt.imshow(image)
print(y_train[index])

# visualize feature map
def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    activation = tf_activation.eval(session=sess, feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
        plt.tight_layout()
# define a LeNet convolution network, x is the input training data
def LeNet(x, n_classes, dropout):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    w_1 = tf.Variable(tf.truncated_normal([5,5,3,6], mean=mu, stddev=sigma))
    b_1 = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, w_1, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.contrib.layers.batch_norm(conv1)
    conv1 = tf.nn.bias_add(conv1, b_1)
    # Activation.
    conv1 = tf.nn.elu(conv1)
    
    # Pooling. Input = 28x28x6. Output = 14x14x6.
    ksize_1 = [1,2,2,1]
    strides_1 = [1,2,2,1]
    conv1 = tf.nn.max_pool(conv1, ksize_1, strides_1, padding='VALID')
    # Layer 2: Convolutional. Output = 10x10x16.
    w_2 = tf.Variable(tf.truncated_normal([5,5,6,16], mean=mu, stddev=sigma))
    b_2 = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, w_2, strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.contrib.layers.batch_norm(conv2)
    conv2 = tf.nn.bias_add(conv2, b_2)
    # Activation.
    conv2 = tf.nn.elu(conv2)
    
    # Pooling. Input = 10x10x16. Output = 5x5x16.
    ksize_2 = [1,2,2,1]
    strides_2 = [1,2,2,1]
    conv2 = tf.nn.max_pool(conv2, ksize_2, strides_2, 'VALID')  
    
    # Flatten. Input = 5x5x16. Output = 400.
    fc0 = tf.contrib.layers.flatten(conv2)
#     Layer 3: Fully Connected. Input = 400. Output = 120.
    w_4 = tf.Variable(tf.truncated_normal([400, 120], mean=mu, stddev=sigma))
    b_4 = tf.Variable(tf.zeros(120))
    fc1 = tf.add(tf.matmul(fc0, w_4), b_4)
    # Activation.
    fc1 = tf.nn.elu(fc1)
    
    # Layer 4: Fully Connected. Input = 120. Output = 84.
    w_5 = tf.Variable(tf.truncated_normal([120, 84], mean=mu, stddev=sigma))
    b_5 = tf.Variable(tf.zeros(84))
    fc2 = tf.add(tf.matmul(fc1, w_5), b_5)
    # Activation.
    fc2 = tf.nn.elu(fc2)
    fc2 = tf.nn.dropout(fc2, dropout)
    
    # Layer 5: Fully Connected. Input = 84. Output = 10.
    w_6 = tf.Variable(tf.truncated_normal([84, n_classes], mean=mu, stddev=sigma))
    b_6 = tf.Variable(tf.zeros(n_classes))
    logits = tf.add(tf.matmul(fc2, w_6), b_6) 
    
    # return conv1 and conv2 to visualize learned feature maps
    return conv1, conv2, logits

# parameters
learning_rate = 1e-3
epochs = 35
batch_size = 128

n_classes = len(np.unique(y_train))
    
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
dropout = tf.placeholder_with_default(1.0, shape=())
one_hot_y = tf.one_hot(y, n_classes)

# training pipeline
conv1, conv2, logits = LeNet(x, n_classes, dropout)
logits_soft = tf.nn.softmax(logits)
predictions = tf.argmax(logits, 1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_operation = optimizer.minimize(loss_operation)

# model evaluation
correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(one_hot_y,1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(x_data, y_data):
    num_examples = len(x_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, batch_size):
        batch_x, batch_y = x_data[offset:offset+batch_size], y_data[offset:offset+batch_size]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(x_train)
    
    print("Training")
    for i in range(epochs):
        x_train, y_train = shuffle(x_train, y_train)
        for offset in range(0, num_examples, batch_size):
            end = offset + batch_size
            batch_x, batch_y = x_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, dropout: 0.75})
        
        validation_accuracy = evaluate(x_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        
    saver.save(sess, MODEL_SAVE)
    print("Model saved")


# evaluate the model on test dataset
with tf.Session() as sess:

    saver.restore(sess, MODEL_SAVE)
    
    test_accuracy = evaluate(x_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
    
# test on new images
with tf.Session() as sess:
    saver.restore(sess, MODEL_SAVE)    
    
    image_folder = "visualize"
    images = os.listdir(image_folder)
    imgs = []
    for image_file in images:
        image_path = image_folder + '/' + image_file
        img = Image.open(image_path)
        img = img.convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)
        img = np.array(list(img.getdata()), dtype='uint8')
        img = np.reshape(img, (IMG_SIZE, IMG_SIZE, 3))
        
        imgs.append(img)   
    
    pred = sess.run(predictions, feed_dict={x: imgs})
    log_soft = sess.run(logits_soft, feed_dict={x: imgs})
    outputFeatureMap(imgs, conv2)
    
    # build the label map 
    label_map = {}
    with open('signnames.csv') as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            label, label_descrip = int(row[0]), row[1]
            label_map[label] = label_descrip        
    
    # print out top 5 softmax probabilities with corresponding sign category
    final_pred = [label_map[i] for i in pred]
    for i in range(len(imgs)):
        index = np.argpartition(log_soft[i], -5)[-5:]
        ind_sort = index[np.argsort(log_soft[i][index])]
        ind_sort = ind_sort[::-1]
        top5_labels = [label_map[j] for j in ind_sort]
        print('%s --> %s --> %s -->%s' % (images[i], final_pred[i], log_soft[i][ind_sort], top5_labels))
        print('\n')
        
