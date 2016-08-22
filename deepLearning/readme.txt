https://github.com/lisa-lab/DeepLearningTutorials
http://deeplearning.net/tutorial/lenet.html
http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
http://colah.github.io/posts/2014-07-Understanding-Convolutions/



//===
The Full Model: LeNet
Sparse, convolutional layers and max-pooling are at the heart of the LeNet family of models. 
While the exact details of the model will vary greatly, 

//=== max-pooling is a form of non-linear down-sampling.
Max-pooling partitions the input image into a set of non-overlapping rectangles and, 
for each such sub-region, outputs the maximum value.

Max-pooling is useful in vision for two reasons:
1. By eliminating non-maximal values, it reduces computation for upper layers.

2. It provides a form of translation invariance. 
Imagine cascading a max-pooling layer with a convolutional layer. 
There are 8 directions in which one can translate the input image by a single pixel. 
If max-pooling is done over a 2x2 region, 
3 out of these 8 possible configurations will produce exactly the same output at the convolutional layer. 
For max-pooling over a 3x3 window, this jumps to 5/8.

Since it provides additional robustness to position, max-pooling is a “smart” way of reducing the dimensionality of intermediate representations.

Max-pooling is done in Theano by way of theano.tensor.signal.pool.pool_2d. 


//===
AVERAGING EACH PIXEL WITH ITS NEIGHBORING VALUES BLURS AN IMAGE:
TAKING THE DIFFERENCE BETWEEN A PIXEL AND ITS NEIGHBORS DETECTS EDGES:


In a traditional feedforward neural network we connect each input neuron to each output neuron in the next layer. 
That’s also called a fully connected layer, or affine layer. 

In CNNs we don’t do that. Instead, we use convolutions over the input layer to compute the output. This results in local connections, 
where each region of the input is connected to a neuron in the output


There’s also something something called pooling (subsampling) layers

There are two aspects of this computation worth paying attention to: 
  Location Invariance and Compositionality. 
  
  Let’s say you want to classify whether or not there’s an elephant in an image. 
  Because you are sliding your filters over the whole image 
  you don’t really care where the elephant occurs. 
  In practice,  pooling also gives you invariance to translation, rotation and scaling,


but for NLP
Location Invariance and local Compositionality made intuitive sense for images, 
but not so much for NLP. 
You probably do care a lot where in the sentence a word appears. 

Pixels close to each other are likely to be semantically related (part of the same object), 
but the same isn’t always true for words. 
In many languages, parts of phrases could be separated by several other words. 
The compositional aspect isn’t obvious either



//=== zero padding for edge point
wide / narrow

When I explained convolutions above I neglected a little detail of how we apply the filter. 
Applying a 3×3 filter at the center of the matrix works fine, 
but what about the edges? 

How would you apply the filter to the first element of a matrix that doesn’t have any neighboring elements 
to the top and left? You can use zero-padding. 

All elements that would fall outside of the matrix are taken to be zero. 
By doing this you can apply the filter to every element of your input matrix, 
and get a larger or equally sized output. 

Adding zero-padding is also called wide convolution, and 
not using zero-padding would be a narrow convolution. 

An example in 1D looks like this:
 Filter size 5, input size 7. 
 the narrow convolution yields  an output of size (7-5) + 1=3, and 
 a wide convolution an output of size (7+2*4 - 5) + 1 =11. 
 More generally, the formula for the output size is n_{out}=(n_{in} + 2*n_{padding} - n_{filter}) + 1 .
 
 
 The following from the Stanford cs231 website shows stride sizes of 1 and 2 applied to a one-dimensional input:
 http://cs231n.github.io/convolutional-networks/
 
 a larger stride size may allow you to build a model that behaves somewhat 
 similarly to an RNN, Recursive Neural Network, i.e. looks like a tree.
 
 CNN and RNN
 
 
 //===
 pooling layers, typically applied after the convolutional layers. 
 Pooling layers subsample their input.
  The most common way to do pooling it to apply a max operation to the result of each filter. 
  
  In imagine recognition, pooling also provides 
  basic invariance to translating (shifting) and rotation. 
  When you are pooling over a region, the output will stay approximately the same 
  even if you shift/rotate the image by a few pixels, 
  because the max operations will pick out the same value regardless.
  
//===  Channels are different “views” of your input data. 
For example, in image recognition you typically have RGB (red, green, blue) channels. 
 You can apply convolutions across channels, either with different or equal weights

For NLP  
 You could have a separate channels for different word embeddings (word2vec and GloVe for example), or 
 you could have a channel for the same sentence represented in different languages, or phrased in different ways. 
 
 http://www.wildml.com/2015/11/understanding-convolutional-neural-networks-for-nlp/
  
  
   However the max operation comes with non-convexity therefore it obstructs convex optimization.
   
 
 //=== 
 http://cs231n.github.io/convolutional-networks/
 
 the input layer holds the image, 
 so its width and height would be the dimensions of the image, and 
 the depth(channels) would be 3 (Red, Green, Blue channels).
 
* INPUT [32x32x3] will hold the raw pixel values of the image, in this case an image of width 32, height 32, and with three color channels R,G,B.
* CONV layer will compute the output of neurons that are connected to local regions in the input, each computing a dot product between their weights and a small region they are connected to in the input volume. This may result in volume such as [32x32x12] if we decided to use 12 filters.
* RELU layer will apply an elementwise activation function, such as the max(0,x)max(0,x) thresholding at zero. This leaves the size of the volume unchanged ([32x32x12]).
* POOL layer will perform a downsampling operation along the spatial dimensions (width, height), resulting in volume such as [16x16x12].
* FC (i.e. fully-connected) layer will compute the class scores, resulting in volume of size [1x1x10], where each of the 10 numbers correspond to a class score, such as among the 10 categories of CIFAR-10. As with ordinary Neural Networks and as the name implies, 
each neuron in this layer will be connected to all the numbers in the previous volume.



