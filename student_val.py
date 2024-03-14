"""
the general training framework
"""

from __future__ import print_function

import argparse
import socket
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders, get_cifar100_dataloaders_sample, super_get_cifar100_dataloaders

from helper.loops import train_distill as train, validate

def parse_option():
    hostname = socket.gethostname()
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100'], help='dataset')
    # model
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')
    # purification dataset
    parser.add_argument('--threshold', default=0.5, type=float)
    parser.add_argument('--loaddir', default=None, type=str)
    parser.add_argument('--savedir', default=None, type=str)
    parser.add_argument('--mix_up_data_p', default=None, type=str)
    opt = parser.parse_args()
    if hostname.startswith('visiongpu'):
        opt.model_path = './path/to/my/student_model'
        opt.tb_path = './path/to/my/student_tensorboards'
    else:
        opt.model_path = './save/student_model'
        opt.tb_path = './save/student_tensorboards'
    opt.lr_decay_epochs = list([])
    opt.model_t = get_teacher_name(opt.path_t)
    return opt
def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]
def load_student(model_path, n_cls):
    print('==> loading student model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model
def main():
    opt = parse_option()
    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data, train_set = super_get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True,
                                                                    init = False,
                                                                    threshold = opt.threshold,
                                                                    loaddir=opt.loaddir,
                                                                    savedir=opt.savedir,
                                                                    mix_up_data_p=opt.mix_up_data_p)
        n_cls = 100
    else:
        raise NotImplementedError(opt.dataset)

    init_train_loader, _, _, _ = super_get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                        num_workers=opt.num_workers,
                                                                        is_instance=True,
                                                                        init = True,
                                                                        threshold = opt.threshold,
                                                                        loaddir=opt.loaddir,
                                                                        savedir=opt.savedir)
    # model
    model_t = load_student(opt.path_t, n_cls)
    device = torch.device("cuda")
    model_t.to(device)
    model_t.eval()
    criterion_cls = nn.CrossEntropyLoss()
    # append teacher after optimizer to avoid weight_decay
    if torch.cuda.is_available():
        device = torch.device("cuda")
        cudnn.benchmark = True
    # validate teacher accuracy
    acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('student accuracy: ', acc)
if __name__ == '__main__':
    main()
