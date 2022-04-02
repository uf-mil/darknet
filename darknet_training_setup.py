#! /usr/bin/env python3
import os
from roboflow import Roboflow

#download the newly released yolov4-tiny weights
if not os.path.exists("yolov4-tiny.conv.29"):
    os.system("wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.conv.29")

if not os.path.exists("yolov4-tiny.weights"):
    os.system("wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights")


##### Replace this block ####
rf = Roboflow(api_key="mzARNr9JA1mnJ93EvCOV")
project = rf.workspace("alex-perez").project("buoy-segmentation")
dataset = project.version("1").download("darknet")
#############################

#Set up training file directories for custom dataset
os.system("cp {dataset_dir}/train/_darknet.labels data/obj.names".format(dataset_dir=dataset.location))

if os.path.exists("data/obj"):
    os.system("rm -rf data/obj")

os.system("mkdir data/obj")

#copy image and labels
os.system("cp {dataset_dir}/train/*.jpg data/obj/".format(dataset_dir=dataset.location))
os.system("cp {dataset_dir}/valid/*.jpg data/obj/".format(dataset_dir=dataset.location))
os.system("cp {dataset_dir}/train/*.txt data/obj/".format(dataset_dir=dataset.location))
os.system("cp {dataset_dir}/valid/*.txt data/obj/".format(dataset_dir=dataset.location))
os.system("rm -rf backup")
os.system("mkdir backup")

with open('data/obj.data', 'w') as out:
  out.write('classes = 3\n')
  out.write('train = data/train.txt\n')
  out.write('valid = data/valid.txt\n')
  out.write('names = data/obj.names\n')
  out.write('backup = backup')

#write train file (just the image list)

with open('data/train.txt', 'w') as out:
  for img in [f for f in os.listdir(dataset.location + '/train') if f.endswith('jpg')]:
    out.write('data/obj/' + img + '\n')

#write the valid file (just the image list)

with open('data/valid.txt', 'w') as out:
  for img in [f for f in os.listdir(dataset.location + '/valid') if f.endswith('jpg')]:
    out.write('data/obj/' + img + '\n')

def file_len(fname):
  with open(fname) as f:
    for i, l in enumerate(f):
      pass
  return i + 1

num_classes = file_len(dataset.location + '/train/_darknet.labels')
max_batches = num_classes*2000
steps1 = .8 * max_batches
steps2 = .9 * max_batches
steps_str = str(steps1)+','+str(steps2)
num_filters = (num_classes + 5) * 3

print("writing config for a custom YOLOv4 detector detecting number of classes: " + str(num_classes))

#Instructions from the darknet repo
#change line max_batches to (classes*2000 but not less than number of training images, and not less than 6000), f.e. max_batches=6000 if you train for 3 classes
#change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
if os.path.exists('./cfg/custom-yolov4-tiny-detector.cfg'): 
    os.remove('./cfg/custom-yolov4-tiny-detector.cfg')

text_file = open("./cfg/custom-yolov4-tiny-detector_template.cfg", "r")
 
#read whole file to a string
data = text_file.read()
text_file.close()
data = data.format(max_batches=max_batches, steps_str=steps_str, num_classes=num_classes, num_filters=num_filters)

f = open("./cfg/custom-yolov4-tiny-detector.cfg", "w")
f.write(data)
f.close()

#begin training
os.system("./darknet detector train data/obj.data cfg/custom-yolov4-tiny-detector.cfg yolov4-tiny.conv.29 -dont_show -map")