# Cat vs Dog: The Permutation Way

<div align= "center" >
<img width="365" height="410" alt="998" src="https://github.com/user-attachments/assets/a5d84df2-42d0-4342-be11-36e6cb4f0a54" />  <img width="265" height="410" alt="9762" src="https://github.com/user-attachments/assets/33080b6b-63f4-4c51-8f75-e0886110ceb9" />
</div>

As a part of our learning during our ML PhD course, we had the task of creating an esoteric algorithm. With the pure intention of practice and having fun while learning the importance of algorithm complexity and the use of data.

For this project, I presented an inefficient esoteric algorithm for solving a classical machine learning problem: image classification between cats and dogs. We know that modern approaches typically rely on Convolutional Neural Networks (CNNs).

Also, on this READ.ME, I'll explain it as simple as possible so you can either check the YouTube video or the .PDF file for the math explanation and final complexity. 

**IMPORTANT!**: The code is adapted to my folders and routes so it's important, if you want to test it, to change those folder routes. :)

## Youtube video ( better explanation! :) )

[very helpful explanation! 👋](https://youtu.be/NYHYGpUi7JM?si=iRDp8Yb0Y9fr1PhS)



## 1. Dataset

The dataset used was selected from Kaggle called [Cats and Dogs Classification Dataset](https://www.kaggle.com/datasets/bhavikjikadara/dog-and-cat-classification-dataset?select=PetImages), contains 24,998 images total divided by 12,499 of cat images and 12,499 of dog images. 

*For this project in particular, having in mind that is extremely inefficient, I only used 10000 in for each folder (cat, dog).*

In this repository you will find only 100 images per folder, just as an example.


## 2. Algorithm description

### 1. Image normalization
All images need to be the size of 64x64 pixels while preserving RGB channels.

### 2. Image flattening
Images are converted into a one-dimensional vector, where each value goes from *[0, 255]*.

### 3. Conversion to a large integer
This step creates a numeric encoding of the entire image.

### 4. Permutation generation
The digits generated of *N* are used to generate a permutation of the pixels indices.

### 5. Pixel rearrangement
This step **INTENTIONALLY** destroys the spatial structure of the image.

### 6. Classification
Then, the scrambled images are classified using a *k-nearest neighbors* classifier. Based on distance between the permuted pixel vectors.

## 3. Best and worst case

Based on the simple explanation of the code, we can already know that our compleity will be extremely big, meaning, inefficiency. Instead of following the classic neural network for image classification, we're using permutation. Inefficient and WRONG. Perfect for practice:

* Worst case: *O(np)*
* Best case: *O(np)*

Being *n* the number of training images and *p* the number of pixel values.

**Have fun!!!**








