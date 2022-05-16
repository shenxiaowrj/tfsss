# _*_coding:utf-8_*_
# 主要的训练文件
# 其中借用了一部分misc库 也就是mae写的库

import os
import math
import random
import argparse
from time import time
import glob
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional
from PIL import Image
import torch.utils.data

import timm
from timm.utils import accuracy

from util import misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

# 设备 全局的
# 自动判断 如果cuda可用就用cuda 否则就使用cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.to(device)

    # 创建一个日志
    """继承自mae的方法"""
    metric_logger = misc.MetricLogger(delimiter=" ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    # 读取每个batch之中的数据
    """继承自mae库中的一个方法"""
    for batch in metric_logger.log_every(data_loader, 10, header): ###### 这个10 是什么意思啊？
        # 得到图片 和 标签
        images = batch[0]
        target = batch[-1]
        # 将图片 和 标签传入指定的设备之中 为之后的运算做准备
        images = images.to(device, non_blocking= True)
        target = target.to(device, non_blocking= True) ###### non_blocking是什么意思啊？

        # 输入到模型之中得到输出
        output = model(images)
        # 输入 输出和标签 计算损失
        loss = criterion(output, target)

        # 对输出进行一个softmax     将原本的能量 转换为 概率
        ################ 为什么是在计算完损失函数之后进行的softmax 不是一般都在构建loss之前 在网络之中进行softmax吗？
        output = torch.nn.functional.softmax(output, dim=-1)
        # 计算准确度
        """继承自timm库中的方法"""
        acc1, acc5 = accuracy(output, target, topk=(1,5))  # topk topkey 可以是1 5 也可以1 3 等等 自己设置

        # 将算出来的值 更新在metric的日志里 可以让损失函数计算的值 更加的平滑 比较容易的反应出模型的训练情况
        """继承自mae中的方法"""
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the starts from all processes
    metric_logger.synchronize_between_processes()

    # 打印出得到的相关信息
    print('* Acc@1 {top1.global_avg: .3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1 = metric_logger.acc1, top5 = metric_logger.acc5, losses = metric_logger.loss))

    # 返回经过metric_logger处理过后的结果 存放格式为字典
    """继承自mae库中的方法"""
    return {k: meter.global_avg for k, meter in metric_logger.meters.items() }

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float=0,
                    log_writer = None,args= None):
    model.train(True)

    print_freq = 2

    accum_iter = args.accum_iter   # 隔几个step进行一次梯度更新


    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples1, targets) in enumerate(data_loader):

        samples1 = samples1.to(device, non_blocking = True) # non_blocking 是什么意思啊？
        targets = targets.to(device, non_blocking = True)

        outputs = model(samples1) # output 也就是 logits

        # 这里没有使用变化的学习率
        warmup_lr = args.lr
        optimizer.param_groups[0]["lr"] = warmup_lr

        loss = criterion(outputs, targets)
        loss /= accum_iter


        # loss_scaler = NativeScaler()
        """继承自mae模型中的方法"""
        loss_scaler(loss, optimizer, clip_grad=max_norm, parameters=model.parameters(),
                    create_graph = False,
                    update_grad = (data_iter_step + 1) % accum_iter == 0)

        loss_value = loss.item()

        # 当然，若是设置有accum_iter 则计算多个batch之后 才进行梯度置零
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # 将一些训练的数据传入tensorboard的日志之中
        if log_writer is not None and (data_iter_step + 1) % accum_iter ==0:
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch ) * 10000) # 这里将横轴进行了一个放大 画出来的图更好看
            log_writer.add_scalar('loss', loss_value, epoch_1000x)
            log_writer.add_scalar('lr', warmup_lr, epoch_1000x)
            print(f"Epoch: {epoch}, Step: {data_iter_step}, Loss: {loss}, lr:{warmup_lr}")

def build_transform(is_train, args):
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        print("train transform")
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((args.input_size,args.input_size)),
                torchvision.transforms.RandomHorizontalFlip(),# 水平翻转
                torchvision.transforms.RandomVerticalFlip(),# 垂直翻转
                torchvision.transforms.RandomPerspective(distortion_scale=0.6, p=1.0),# 随机变化一下观看的视角
                torchvision.transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 5)),# 随机加一点高斯噪声
                torchvision.transforms.ToTensor(),# 必备操作 也就是将unit-8格式的数据 转化为 0-1 之间的浮点数

            ]
        )

    # eval transform
    print("eval transform")
    return torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((args.input_size,args.input_size)),
            torchvision.transforms.ToTensor()
        ]
    )

def build_dataset(is_train,args):
    transform = build_transform(is_train,args)

    #################这里写的可能有问题 args里边没有root_path参数 也没有类似“train” “test"的不同的指定
    path = os.path.join(args.root_path, 'train' if is_train else 'test')
    # ImageFolder函数
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    info = dataset._find_classes(path)
    print(f"finding classes from {path}:\t{info[0]}") # 返回一个列表
    print(f"mappping classes from {path} to indexes:\t{info[1]}") # 返回一个字典

    return dataset

def get_args_parser():
    parser = argparse.ArgumentParser('MAE pretraining ', add_help=False)
    parser.add_argument('--batch_size', default=72, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    parser.add_argument('--input_size', default=128, type=int,
                        help='images input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (absolute lr)')

    # parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
    #                     help='dataset path')
    parser.add_argument('--root_path', default='dataset_fruit_veg', type=str,
                        help='dataset path')
    parser.add_argument('--output_dir', default='./output_dir_pretrained1',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir_pretrained1',
                        help='path where to tensorboard log')
    # parser.add_argument('--device', default='cuda',
    #                     help='device to use for training / testing')
    # parser.add_argument('--seed', default=0, type=int)

    # parser.add_argument('--resume', default='',
    #                     help='resume from checkpoint')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser

def main(args, mode='train', test_image_path=''):
    print(f'{mode} mode...')
    if mode == "train":

        # 构建dataset
        dataset_train = build_dataset(is_train=True, args=args)  # build_dataset build_trasform
        dataset_val = build_dataset(is_train=False, args=args)

        sampler_train = torch.utils.data.RandomSampler(dataset_train) #将数据进行打散 随机进行采样
        sampler_val = torch.utils.data.SequentialSampler(dataset_val) #验证的时候就不用了 直接进行顺序的采样就可以了

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            sampler=sampler_train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem, #这是什么意思呀？
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        # 构建模型
        model = timm.create_model(model_name='resnet18', pretrained=True, num_classes=20,drop_rate= 0.1, drop_path_rate=0.1)

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) # 如果p需要梯度 然后才计算进去
        print('number of trainable_params (M): %.2f' % (n_parameters / 1.e6)) # 除以1.e6（一百万） 放入到 以M（million）为单位的

        # 定义一个损失函数 和 优化器
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) # 使用的带有weight decay的 adam优化器 使用的是固定的学习率

        # 创建一个存放日志的文件夹
        os.makedirs(args.log_dir, exist_ok=True)
        # SummaryWriter tensorboard中的方法 可以对生成的日志文件 进行可视化分析
        log_writer = SummaryWriter(log_dir=args.log_dir)
        # NativeScaler 可以对损失函数的值进行平滑 # 并且对其进行反向传播的操作等等 反向传播 更新梯度 更新参数step
        '''继承自mae模型的一个方法'''
        loss_scaler = NativeScaler()

        '''继承自mae模型的一个方法'''
        misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)

        for epoch in range(args.start_epoch, args.epochs):

            # 打印一下现在是第几个epoch
            print(f"Epoch {epoch}")
            # 打印一下训练数据集的大小？？？
            print(f"length of data_loader_train is {len(data_loader_train)}")


            if epoch % 1 == 0:
                print('Evaluating...')
                # 将模型设置为eval模型 让dropout batchnorm等操作失效
                model.eval()
                # 传入验证数据集 模型 指定的设备 得到存有acc1 acc5结果的字典
                test_stats = evaluate(data_loader_val, model, device)
                """evaluate"""
                print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")

                if log_writer is not None:
                    # 在perf中新建三行 分别得到acc1 acc5 loss(经过metric_logger平滑之后) 的变化趋势图 每个epoch产生一个点
                    log_writer.add_scalar('perf/test_acc1', test_stats['acc1'], epoch)
                    log_writer.add_scalar('perf/test_acc5', test_stats['acc5'], epoch)
                    log_writer.add_scalar('perf/test_loss', test_stats['loss'], epoch)
                # 做完推理之后 将模型还原会train的模式 也就是让dropout batchnorm等生效
                model.train()

            print("Training...")
            train_stats = train_one_epoch(model, criterion, data_loader_train, optimizer=optimizer,
                                          device=device, epoch=epoch+1,  loss_scaler=loss_scaler,max_norm=None,
                                          log_writer=log_writer, args=args)

            """train_one_epoch"""

            if args.output_dir:    # output_dir 这个参数是什么意思呀？
                print("saving checkpoints")
                misc.save_model(
                    args=args, model=model, model_without_ddp=model, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch
                )
                """继承自mae模型"""
        else: #info 逻辑
            model = timm.create_model('resnet18', pretrained=True, num_classes=20, drop_rate=0.1, drop_path_rate=0.1)

            class_dict = {'apple': 0, 'banana': 1, 'beetroot': 2, 'bell pepper': 3, 'cabbage': 4, 'capsicum': 5, 'carrot': 6, 'cauliflower': 7, 'chilli pepper': 8, 'corn': 9, 'cucumber': 10, 'eggplant': 11, 'garlic': 12, 'ginger': 13, 'grapes': 14, 'jalepeno': 15, 'kiwi': 16, 'lemon': 17, 'lettuce': 18, 'mango': 19}

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print("number of trainable params (M): %.2f" % (n_parameters / 1.e6))

            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            os.makedirs(args.log_dir,exist_ok=False)
            loss_scaler = NativeScaler()

            misc.load_model(args=args, model_without_ddp=model, optimizer=optimizer, loss_scaler=loss_scaler)
            model.eval()

            image = Image.open(test_image_path).convert('RGB')
            image = image.resize((args.input_size, args.input_size), Image.ANTIALIAS).unsuqeeze(0)
            image = torchvision.transforms.ToTensor()(image)          #扩一维出来 扩出来batch的维度
            #########为什么 ToTensor之后，假的(image)???

            with torch.no_grad():
                output = model(image)

            output = torch.nn.functional.softmax(input=output, dim=-1)
            class_idx = torch.argmax(input=output, dim=1)[0]    #后边这个0是什么意思啊？
            score = torch.max(input=output,dim=1)[0][0]
            print(f"image path is {test_image_path}")
            # 细节的输出操作
            print(f"score is {score.item()}, class id is {class_idx.item()},class name is {list(class_dict.keys())[list(class_dict.values()).index(class_idx)]} ")
            # time.sleep(0.5)

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=False)

    mode = 'train' # infer or train

    if mode == 'train':
        main(args, mode=mode)
    else:
        images = glob.glob('dataset_fruit_veg/test/*/*.jpg')      # 这里需要填入你自己的路径

        for image in images:
            print('\n')
            main(args=args, mode=mode, test_image_path=image)


# 少传一个max_norm 竟然会导致模型不收敛！！！ why?