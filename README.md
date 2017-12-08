# tiny-YOLO-cnn

This is a reproduction of YOLO algorithm (tiny-YOLO) for computer vision using Keras.

Original [paper](https://arxiv.org/pdf/1612.08242.pdf) and [website](https://pjreddie.com/darknet/yolo/).


The results are kind of meh... Sometimes the boxes get really shaky if objects are moving. Classification is not always very accurate probably because it's the tiny version. But here goes:

![](/gifs/im_a_tie_v2.gif) ![](/gifs/random_video2.gif) 

There are a bunch of way better implementations of tiny-YOLO than this one. But if for some reason you want to try running this code, you have to include [.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/tiny-yolo.cfg) and [weights](https://pjreddie.com/media/files/tiny-yolo.weights) into your wd.
