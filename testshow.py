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
import random
import skimage

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
	UnNormalizer, Normalizer
from retinanet import model


assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))

def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

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

    #imglist= sorted(glob.glob(os.path.join(parser.data_dir, '*.jpg')))
    #random.shuffle(imglist)
    #imglist=imglist[:int(parser.num_totest)] 
    dataset_train = CSVDataset(train_file='./foreign_object_dataset/train_standard_annotation.csv', class_list='./foreign_object_dataset/class4_classlist.csv',
                                   transform=transforms.Compose([Normalizer(), Resizer()]))

    sampler = AspectRatioBasedSampler(dataset_train, 1, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

# 	retinanet = torch.load(parser.model)

# 	use_gpu = True

# 	if use_gpu:
# 		if torch.cuda.is_available():
# 			retinanet = retinanet.cuda()

# 	if torch.cuda.is_available():
# 		retinanet = torch.nn.DataParallel(retinanet).cuda()
# 	else:
# 		retinanet = torch.nn.DataParallel(retinanet)

    #retinanet = model.resnet50(num_classes=80, pretrained=True)
    #retinanet.load_state_dict(torch.load(parser.model))
    
    retinanet = torch.load(parser.model)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        #retinanet.load_state_dict(torch.load(parser.model))
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        #retinanet.load_state_dict(torch.load(parser.model))
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.eval()
    unnormalize = UnNormalizer()

    trans=transforms.ToTensor()
    FODlabels=['nail','clipper_B','clipper_Y','Lstick','butt','foreign']

    for iter_num, data in enumerate(dataloader_train):
        with torch.no_grad():
            st = time.time()

            if torch.cuda.is_available():
                scores, classification, transformed_anchors =  retinanet(data['img'].cuda().float())

            #scores, classification, transformed_anchors = retinanet(img_tensor.float())
            print('Elapsed time: {}'.format(time.time()-st))
            idxs = np.where(scores.cpu()>0.5)
            
            #img[img<0] = 0
            #img[img>255] = 255
            #img = np.transpose(img, (1, 2, 0))
            #img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            #img=tensor_to_np(data['img']).astype(np.uint8)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            #img=cv2.imread(data['filename'][0])
            #img=cv2.resize(img,(1056,608))

            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()
            img[img<0] = 0
            img[img>255] = 255
            img = np.transpose(img, (1, 2, 0))
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = FODlabels[int(classification[idxs[0][j]])]
                #label_name='FOD'
                draw_caption(img, (x1, y1, x2, y2), label_name)
                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                print(label_name)
            #cv2.imwrite('./visualizeResult/visualize'+str(iter_num)+'.jpg', img)
            #import pdb
            #pdb.set_trace()
            cv2.imwrite('./visualizeResult/visualize'+data['filename'][0].split('/')[-1], img)
            if iter_num>int(parser.num_totest):
                break
            #cv2.waitKey(0)
            


if __name__ == '__main__':
 main()