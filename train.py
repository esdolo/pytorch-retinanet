import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6,7"

import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
from tensorboardX import SummaryWriter

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    
    parser.add_argument('--finetune', help='if load trained retina model', type=bool, default=False)
    parser.add_argument('--gpu', help='', type=bool, default=False)
    parser.add_argument('--batch_size', help='', type=int, default=2)

    parser.add_argument('--c', help='continue with formal model',type=bool,default=False)
    parser.add_argument('--model', help='model path')

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    #sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False)
    sampler = AspectRatioBasedSampler(dataset_train, parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    epochpassed=0
    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if parser.c:
        retinanet=torch.load(parser.model)
        #import pdb
        #pdb.set_trace()
        epochpassed=int(parser.model.split('.')[1].split('_')[-1])
    use_gpu = parser.gpu

    #torch.cuda.set_device(5)
    #import pdb
    #pdb.set_trace()

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if use_gpu and torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()

    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)#original:1e-5
    #optimizer =optim.SGD(retinanet.parameters(), lr=0.01,weight_decay=0.0001, momentum=0.9)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    writer = SummaryWriter()

    for epoch_num in range(parser.epochs):

        retinanet.train()
        retinanet.module.freeze_bn()

        epoch_loss = []
        epoch_classification_loss=[]
        epoch_regression_loss=[]

        for iter_num, data in enumerate(dataloader_train):
            try:
                #import pdb
                #pdb.set_trace()

                optimizer.zero_grad()

                if use_gpu and torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda()])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))
                epoch_classification_loss.append(float(classification_loss))
                epoch_regression_loss.append(float(regression_loss))

                print('Epoch: {} | Iteration: {} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Epoch loss: {:1.5f}\r'.format(
                          epoch_num+epochpassed, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist)),end='')

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
        
        print('Epoch: {}  | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Epoch loss: {:1.5f}'.format(
                        epoch_num+epochpassed,  np.mean(epoch_classification_loss), np.mean(epoch_regression_loss), np.mean(epoch_loss)))
        
        writer.add_scalar('lossrecord/regressionloss', np.mean(epoch_regression_loss),epoch_num+epochpassed)
        writer.add_scalar('lossrecord/classificationloss', np.mean(epoch_regression_loss), epoch_num+epochpassed)
        writer.add_scalar('lossrecord/epochloss', np.mean(epoch_loss), epoch_num+epochpassed)

        if parser.dataset == 'coco':

            print('Evaluating dataset')

            coco_eval.evaluate_coco(dataset_val, retinanet)

        elif parser.dataset == 'csv' and parser.csv_val is not None:

            print('Evaluating dataset')

            mAP = csv_eval.evaluate(dataset_val, retinanet)

        scheduler.step(np.mean(epoch_loss))

        if epoch_num%10==0:
            torch.save(retinanet.module, '{}_retinanet{}_ADAM_{}.pt'.format(parser.dataset, parser.depth,epoch_num+epochpassed))

    retinanet.eval()

    torch.save(retinanet.module, '{}_retinanet{}_ADAM_{}.pt'.format(parser.dataset, parser.depth,parser.epochs+epochpassed))
    writer.close()


if __name__ == '__main__':
    main()
