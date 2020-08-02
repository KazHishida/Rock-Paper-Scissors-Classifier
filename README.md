# Rock Paper Scissors Classifier
## Introduction
 Uses a CNN to classify pictures of hands as rock, paper, or scissors.
## Dataset
 Dataset was found online on TensorFlow's website at https://www.tensorflow.org/datasets/catalog/rock_paper_scissors. 
## Training
 I used a CNN on 300x300 pictures, although in order to allow my GPU to handle the training process, I was forced to downsample the images to 275x275 in grayscale. I found that while grayscale was a more efficient way of downsampling, I still needed to lower the resolution in order to have a proper number of neurons in the hidden layer.
#### Grayscale:
 ![Example](https://i.gyazo.com/6eed3a2b5d77258cc323d453cfaf05e2.png)
#### Resolution Reduction:
 ![Example](https://i.gyazo.com/f48edba7c1dce3249f7576e126b24307.png)

With the soultion I found, my neural network was able to create models with up to 94% accuracy.
## Testing
Although I could achieve up to 94% accuracy on the renderings from the dataset I found online, using real-life images that I took did not achieve such great results. Although accurate more often than not, it would still make mistakes.
#### Success
![Example](https://i.gyazo.com/1eb65c52a7f817246df79a8c197bd46a.png)
#### Failure
![Example](https://i.gyazo.com/d26cd0ea21110510c930f19b3dd9dcac.png)

## Extra Info
To test your own photos, just add them to the MY_HANDS folder. Photos added to the Data folder will be used in training.
