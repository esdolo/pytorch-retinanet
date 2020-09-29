import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import torchvision
import time
import os
import copy
import pdb
import time
import argparse

import glob
import sys
import cv2
import csv
import random

def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--model', help='Path to model (.pt) file.')
    parser.add_argument('--data_dir', help='Path to imgs (.jpg) file.')
    parser.add_argument('--num_totest', help='How many imgs to test.')

    parser = parser.parse_args(args)

    filename="./FOD/FOD/train_standard_annotation.csv"

    FODlabels=['nail','clipper_B','clipper_Y','Lstick','butt','foreign']
   
    with open(filename,'r') as f:
        reader=csv.reader(f)
        lines=list(reader)
        random.shuffle(lines)
        idx=0
        for item in lines:
            img = cv2.imread(item[0])

            #import pdb
            #pdb.set_trace()
            if item[1]=='':
                continue

            x1 = int(item[1])
            y1 = int(item[2])
            x2 = int(item[3])
            y2 = int(item[4])
            label_name = item[5]
            draw_caption(img, (x1, y1, x2, y2), label_name)

            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            print(label_name)
            cv2.imwrite('./visualizeResult/visualize'+str(idx)+'.jpg', img)
            if idx>int(parser.num_totest):
                return
            idx=idx+1

if __name__ == '__main__':
 main()