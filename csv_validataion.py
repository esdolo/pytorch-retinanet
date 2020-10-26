import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import argparse
import torch
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from retinanet.csv_eval import evaluate_csv

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--csv_path', help='Path to csv directory')
    parser.add_argument('--model', help='Path to model', type=str)

    parser = parser.parse_args(args)

    dataset_val = CSVDataset(train_file='./foreign_object_dataset/train_standard_annotation.csv', class_list='./foreign_object_dataset/class4_classlist.csv',
                                   transform=transforms.Compose([Normalizer(), Resizer()]))



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
    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()

    evaluate_csv(dataset_val, retinanet)


if __name__ == '__main__':
    main()
