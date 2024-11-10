# Summary - Paper

## Introduction 

- Challenges of using EMG signals for smooth control: 
    1. Detect the exact hand movement 
    2. Choosing the right combination of features
    3. Having enough data (since we are working with a DNN)
- 8 basic movements and 6 combined movements

# Movements
![image.png](attachment:image.png)

## Preprocessing

- 3/5 of each movements repetitions were added to the training set, remaining in test set
- 15 time and frequency features were used to create 4 feature sets
- Outlier removal and scaling using mean and standard deviation 
- Classifiers
    - K-Nearest Neighbours with 40 neighbours and Euclidian distance metric 
    - Support Vector Machines with linear kernel and regularization parameter equal to one 
    - Multilayer Perceptron with 300 neurons in the single hidden layer, tanh as activation function and 0.00001 learning rate
    - LDA (Linear Discriminant Analysis) as reference classifiers

## Deep Learning methods

- Two baseline CNNs are proposed in the article and both can be divided into 2 parts
- First part is an inter-connected network of convolutional blocks working as a "feature extractor" and the second part consists of few fully connected layers serving as the "classfier"
- Implemented using Keras (We will do it in pytorch)
- Each classifier’s inputs are configured as a 10 × 512 matrix (number of channels × data points in one window)
- Activation function is randomized rectified linear unit (RReLU)
- Dropout with (just going to use) 0.3 are eliminated from hidden layers
- Batch Normalization (BN) was aimed to solve the need for low learning rate and careful parameter init. in the training of the DNN 

## Deep learning architecture 1: Cnet2D

- 3 consecutive conv. blocks constructing the feature extractor part
- followed by 2 fully connected blocks as the classifiers part 
- ully connected blocks as the classifier part (Figure 3). Each convolutional block consists of a convolution layer with a 2D filter shape, BN, RReLU activation layer, max-pooling, and dropout
- Filter sizes of convolutional layers are (3,13), (3,9), (3,5), respectively
- The first fully connected block includes a dense layer, BN, RReLU, and dropout, while the second fully connected block does not include dropout
- In the end, a Softmax layer is added to create the output of the classifier
- Adam optimizer is being used as an optimization method
- During training, the model with minimum validation loss (20% of training data is randomly selected as validation set) is saved and used for testing

## Deep learning architecture 2: Cnet1D

- similar to Cnet2D
- filter’s shape is such that it does not exploit the relations between channels in the feature extraction part
- Filter sizes of convolutional layers are (1,13), (1,9), (1,5), respectively
- max-pooling layer parameters are selected differently
- The max-pooling size for Nearlab is (1,4), while for Ninapro is (1,3)

## Learning Parameters

![image-3.png](attachment:image-3.png)

# Implementation: 

# Reading in the File

Brief Description: Nearlab dataset is a sEMG hand/wrist movement dataset, which
includes 8 basic movements along with their combinations (6 combined
movements), acquired from 11 subjects. Nearlab dataset is specifically useful to
investigate sEMG data transferability 1. between complex movements and their
associated basic movements 2. between subjects. It is our hope that this dataset
will become a useful tool to compare different task-transfer strategies. Moreover,
by considering 3 hand orientations when executing each hand gesture, EMG
variability due to starting hand orientation was accounted for in this database.

### Data format: Data is available in .csv format.
#### Basic movement data:

- Each row is one window (512 sample) extracted from the movements performed in a round. The segments are shuffled. 
- Each row has 5122 columns (512 samples*10 channels+ 2). The first 5120 is related to 512 samples of 10 channels of EMG sensors (first 512 samples for the first channel then second and so on).
- Column 5121 shows label of movement class related to that segment (1 Flexion, 2 Extension, 3 Supination, 4 Pronation, 5 Open, 6 Pinch, 7 Lateral pinch, 8 Fist).
- Column 5122 shows which movement number (out of 40) this row belongs to. (In each round subject performs 5 repetitions of each of the 8 movements, so each round includes 40 movements.) 

- To clarify why this labeling is important please consider the following example:

    For example, window one could be related to class 1 and movement 20 and window
    ten could be related to class 1 but movement 32. Hence, although these segments
    are related to same class but are not from the same repetition of the movement.
    When we want to separate the repetitions of one class to generate train and test
    subsets, it is important to include all segments of the same repetition in one subset
    and do not distribute the segments related to class 1 movement 32 into train and
    test subsets. Otherwise, the classifier will be trained and tested on the same
    repetition and falsely indicate generalizability. The last column providing
    movement number helps the user to separate the shuffled data in the correct
    format.