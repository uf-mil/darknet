#! /usr/bin/env python3
import os
import sys
import cv2
import random
import matplotlib.pyplot as plt

#define utility function
def imShow(path):
  #os.system("matplotlib inline")

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  #plt.rcParams['figure.figsize'] = [10, 5]
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

#define location of datasets as argument
dataset_location = sys.argv[1]

os.system("cp data/obj.names data/coco.names")
#/test has images that we can test our detector on
test_dir = '{dataset_dir}/test'.format(dataset_dir=dataset_location)
test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')]
img_path = test_dir + "/" + random.choice(test_images)

os.system("./darknet detect cfg/custom-yolov4-tiny-detector.cfg backup/custom-yolov4-tiny-detector_best.weights {img_path} -dont-show".format(img_path=img_path))
imShow('/content/darknet/predictions.jpg')