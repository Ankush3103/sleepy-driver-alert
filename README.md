# Sleepy Driver Alert - A Deep Learning car safety project
![Driver sleeping in a Tesla](https://postmediadriving.files.wordpress.com/2019/01/tesla-sleeper.jpg)

## Objective
A huge number of car accidents take place because the driver slips into a slumber and looses control of the car. One way to prevent such mishaps is using a deep learning program with convolutional neural networks to detect if the driver is feeling sleepy based on his or her eye movement.

## Data
To solve this problem, we need a wide variety of data - people with different heights, skin colors, with and without correctional glasses, with varying amount of light and exposure. Collecting this data would have been a huge problem but fortunately a datset with this data already existed. The dataset has been taken from [OpenCV's great haar cascade repository](https://github.com/opencv/opencv/tree/master/data/haarcascades) different images with varying features. 

## Methodology
(This program uses OpenCV, TensorFlow, Keras and Pygame libraries)

- We use openCV's input tools to capture an image

- We then convert the image to greyscale so as to make it usable with openCV library.

- Create a region of interest i.e the pair of eyes. 

- Use CNN classifier to detect wether the eyes are open or not.

- Calculate a relative score for the given image as compared to obtained dataset to detect the amount of drowsiness.

- If sleepy score is beyond a certain score (here, 15), raise an alarm for the driver and/or alert the authorities.

## Running the program

- You must have Python2.7 or higher installed.

- You must have installed, the python libraries used in the program. You may do so by typing the following command in terminal or command prompt;<br><code>pip install tensorflow keras pygame opencv-python-headless</code>

- Then you may clone the repository. 

- run the main file;<br>
    <code>python path/sleepy-driver-detector.py run</code>
    or
    <code>python path/alter_external_method.py</code>
    
- Make sure you have the correct webcam permissions enabled(trust me i spent a lot of time trying to debug something when the actual issue was that my IDE did not have access to my webcam.)

- And finally, don't sleep while driving like come on.

- Here's how the window looks (Don't mind my hair, i just trimmed them on my own like a week back.)

*open*
![open](https://github.com/Ankush3103/sleepy-driver-alert/blob/master/open_not_sleep.png)
*closed*
![closed](https://github.com/Ankush3103/sleepy-driver-alert/blob/master/closed_sleep.png)

## Issues

- Main file (sleepy-driver-alert.py) does not run -- issues with openCV rectangle function and on deleting that all opencv functions are throwing an error.
