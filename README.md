# Sleepy Driver Alert - A Deep Learning car safety project

## Objective
A huge number of car accidents take place because the driver slips into a slumber and looses control of the car. One way to prevent such mishaps is using a deep learning program with convolutional neural networks to detect if the driver is feeling sleepy based on his or her eye movement.

## Data
To solve this problem, we need a wide variety of data - people with different heights, skin colors, with and without correctional glasses, with varying amount of light and exposure. Collecting this data would have been a huge problem but fortunately a datset with this data already existed. The dataset has been taken from (Data-Flair)[data-flair.training] and consists of 7000 different images with varying features. They also provide a model architecture file with final weights attached.

## Methodology
(This program uses OpenCV, TensorFlow, Keras and Pygame libraries)
1. We use openCV's input tools to capture an image
2. We then convert the image to greyscale so as to make it usable with openCV library.
3. Create a region of interest i.e the pair of eyes.

