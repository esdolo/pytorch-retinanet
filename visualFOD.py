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
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer
from retinanet import model


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


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

    imglist= sorted(glob.glob(os.path.join(parser.data_dir, 'train_*.jpg')))
    random.shuffle(imglist)
    imglist=imglist[:int(parser.num_totest)]

# 	retinanet = torch.load(parser.model)

# 	use_gpu = True

# 	if use_gpu:
# 		if torch.cuda.is_available():
# 			retinanet = retinanet.cuda()

# 	if torch.cuda.is_available():
# 		retinanet = torch.nn.DataParallel(retinanet).cuda()
# 	else:
# 		retinanet = torch.nn.DataParallel(retinanet)

    retinanet = model.resnet50(num_classes=80, pretrained=True)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet.load_state_dict(torch.load(parser.model))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load(parser.model))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.eval()
    unnormalize = UnNormalizer()

    trans=transforms.ToTensor()

    for idx,path_name in enumerate(imglist):
        with torch.no_grad():
            st = time.time()
            img = cv2.imread(path_name)
            img_tensor=trans(img).unsqueeze(0)
            #pdb.set_trace()

            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(img_tensor.cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(img_tensor.float())
            print('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores.cpu()>0.5)
            
            #img[img<0] = 0
            #img[img>255] = 255
            #img = np.transpose(img, (1, 2, 0))
            #img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = 'Unknown'
                draw_caption(img, (x1, y1, x2, y2), label_name)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                print(label_name)

            cv2.imwrite('./visualizeResult/visualize'+str(idx)+'.jpg', img)
            #cv2.waitKey(0)
            


if __name__ == '__main__':
 main()