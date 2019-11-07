#!/usr/bin/env python3

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import math
import time
import shutil

from dataloader import get_dataloaders
from args import arg_parser
from adaptive_inference import dynamic_evaluate
import models
from op_counter import measure_model

args = arg_parser.parse_args()

if args.gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.distributed as dist

device = torch.device('cuda' if args.gpu!=None else 'cpu')

args.grFactor = list(map(int, args.grFactor.split('-')))
args.bnFactor = list(map(int, args.bnFactor.split('-')))
args.nScales = len(args.grFactor)

if args.use_valid:
    args.splits = ['train', 'val', 'test']
else:
    args.splits = ['train', 'val']

if args.data == 'cifar10':
    args.num_classes = 10
elif args.data == 'cifar100':
    args.num_classes = 100
else:
    args.num_classes = 1000

torch.manual_seed(args.seed)

def main():

    global args
    best_prec1, best_epoch = 0.0, 0

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.data.startswith('cifar'):
        IM_SIZE = 32
    else:
        IM_SIZE = 224

    model = getattr(models, args.arch)(args)
    n_flops, n_params = measure_model(model, IM_SIZE, IM_SIZE)    
    torch.save(n_flops, os.path.join(args.save, 'flops.pth'))
    del(model)
        
        
    model = getattr(models, args.arch)(args)

    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        if args.gpu is not None:
            model.features = torch.nn.DataParallel(model.features)
            model.features = model.features
        model.to(device)
    else:
        if args.gpu is not None:
            model = torch.nn.DataParallel(model).to(device)
            #model.to(device)
        else:
            model.to(device)
    #from itertools import chain
    #for t in chain(model.module.parameters(), model.module.buffers()):
    #    model.src_device_obj = t.device
    #    break
    #print("model.src_device_obj:",model.src_device_obj)
    
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataloaders(args)

    if args.evalmode is not None:
        state_dict = torch.load(args.evaluate_from)['state_dict']
        model.load_state_dict(state_dict)
        if args.evalblock is not None:
            assert args.evalblock < args.nBlocks
            dist.init_process_group(backend='gloo', init_method="tcp://"+args.master, rank=args.evalblock, world_size=args.nBlocks)
            sample = torch.zeros((args.batch_size, 3, IM_SIZE, IM_SIZE), dtype=torch.float32)
            dims = []
            if args.gpu is not None:
                sample = sample.to(device)
                for block in model.module.blocks:
                    sample = block(sample)
                    temp = []
                    for i in range(len(sample)):
                        temp.append(sample[i].size())
                    dims.append(temp)
                block = model.module.get_block(args.evalblock)
                classifier = model.module.get_classifier(args.evalblock)
            else:
                for block in model.blocks:
                    sample = block(sample)
                    temp = []
                    for i in range(len(sample)):
                        temp.append(sample[i].size())
                    dims.append(temp)
                block = model.get_block(args.evalblock)
                classifier = model.get_classifier(args.evalblock)
            wholeblock = models.MSDBlock(block, classifier)
            if args.gpu is not None:
                wholeblock = torch.nn.DataParallel(wholeblock).to(device)
            else:
                wholeblock.to(device)
            if args.evalblock == 0:
                validate_block(val_loader, wholeblock, criterion)
            else:
                validate_block2(wholeblock,dims)
            return

        if args.evalmode == 'anytime':
            validate(test_loader, model, criterion)
        else:
            dynamic_evaluate(model, test_loader, val_loader, args)
        return

    scores = ['epoch\tlr\ttrain_loss\tval_loss\ttrain_prec1'
              '\tval_prec1\ttrain_prec5\tval_prec5']

    for epoch in range(args.start_epoch, args.epochs):

        train_loss, train_prec1, train_prec5, lr = train(train_loader, model, criterion, optimizer, epoch)

        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion)

        scores.append(('{}\t{:.3f}' + '\t{:.4f}' * 6)
                      .format(epoch, lr, train_loss, val_loss,
                              train_prec1, val_prec1, train_prec5, val_prec5))

        is_best = val_prec1 > best_prec1
        if is_best:
            best_prec1 = val_prec1
            best_epoch = epoch
            print('Best var_prec1 {}'.format(best_prec1))

        model_filename = 'checkpoint_%03d.pth.tar' % epoch
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict(),
        }, args, is_best, model_filename, scores)

    print('Best val_prec1: {:.4f} at epoch {}'.format(best_prec1, best_epoch))

    ### Test the final model

    print('********** Final prediction results **********')
    validate(test_loader, model, criterion)

    return 

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    # switch to train mode
    model.train()

    end = time.time()

    running_lr = None
    for i, (input, target) in enumerate(train_loader):
        lr = adjust_learning_rate(optimizer, epoch, args, batch=i,
                                  nBatch=len(train_loader), method=args.lr_type)

        if running_lr is None:
            running_lr = lr

        data_time.update(time.time() - end)

        if args.gpu:
            target = target.cuda(async=True)
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        if not isinstance(output, list):
            output = [output]

        loss = 0.0
        for j in range(len(output)):
            loss += criterion(output[j], target_var)

        losses.update(loss.item(), input.size(0))

        for j in range(len(output)):
            prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
            top1[j].update(prec1.item(), input.size(0))
            top5[j].update(prec5.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.avg:.3f}\t'
                  'Data {data_time.avg:.3f}\t'
                  'Loss {loss.val:.4f}\t'
                  'Acc@1 {top1.val:.4f}\t'
                  'Acc@5 {top5.val:.4f}'.format(
                    epoch, i + 1, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, top1=top1[-1], top5=top5[-1]))

    return losses.avg, top1[-1].avg, top5[-1].avg, running_lr

def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()
    top1, top5 = [], []
    for i in range(args.nBlocks):
        top1.append(AverageMeter())
        top5.append(AverageMeter())

    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu:
                target = target.cuda(async=True)
            input = input.to(device)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = model(input_var)
            if not isinstance(output, list):
                output = [output]

            loss = 0.0
            for j in range(len(output)):
                loss += criterion(output[j], target_var)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j].data, target, topk=(1, 5))
                top1[j].update(prec1.item(), input.size(0))
                top5[j].update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                        i + 1, len(val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1[-1], top5=top5[-1]))
    for j in range(args.nBlocks):
        print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[j], top5=top5[j]))
    # print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1[-1], top5=top5[-1]))
    return losses.avg, top1[-1].avg, top5[-1].avg

def validate_block(val_loader, wholeblock, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    wholeblock.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu:
                target = target.cuda(async=True)
            input = input.to(device)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            output = wholeblock(input_var)
            class_result = output[0]
            softmax = nn.Softmax(dim=1).to(device)
            confidence = softmax(class_result).max(dim=1, keepdim=False)
            intermediate_data = output[1]
            if confidence[0] < 2:
                for j in range(len(intermediate_data)):
                    if args.gpu:
                        intermediate_data[j] = intermediate_data[j].cpu()
                    dist.send(intermediate_data[j], dst=1)

            loss = criterion(class_result, target_var)

            losses.update(loss.item(), input.size(0))

            prec1, prec5 = accuracy(class_result.data, target, topk=(1, 5))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}\t'
                      'Loss {loss.val:.4f}\t'
                      'Acc@1 {top1.val:.4f}\t'
                      'Acc@5 {top5.val:.4f}'.format(
                        i + 1, len(val_loader),
                        batch_time=batch_time, data_time=data_time,
                        loss=losses, top1=top1, top5=top5))

    print(' * prec@1 {top1.avg:.3f} prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
    return losses.avg, top1.avg, top5.avg

def validate_block2(wholeblock, dims):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    wholeblock.eval()

    end = time.time()
    with torch.no_grad():
        count = 0
        max_count = 10000;
        while count<max_count:
            """
            dims = [[(1, 40, 32, 32),(1, 80, 16, 16),(1, 160, 8, 8)],
                    [(1, 52, 32, 32),(1, 104, 16, 16),(1, 208, 8, 8)],
                    [(1, 70, 16, 16),(1, 140, 8, 8)],
                    [(1, 94, 16, 16),(1, 188, 8, 8)],
                    [(1, 118, 16, 16),(1, 236, 8, 8)],
                    [(1,152,8,8,)]]
            """
            intermediate_data = []
            for dim in dims[args.evalblock-1]:
                temp = torch.zeros(dim, dtype=torch.float32)
                dist.recv(temp, src=args.evalblock-1)
                intermediate_data.append(temp)
            #intermediate_data = torch.zeros((1, 40, 32, 32), dtype=torch.float32)
            #intermediate_data1 = torch.zeros((1, 80, 16, 16), dtype=torch.float32)
            #intermediate_data2 = torch.zeros((1, 160, 8, 8), dtype=torch.float32)
            #dist.recv(intermediate_data, src=args.evalblock-1)
            #dist.recv(intermediate_data1, src=args.evalblock-1)
            #dist.recv(intermediate_data2, src=args.evalblock-1)
            #intermediate_data = [intermediate_data,intermediate_data1,intermediate_data2]
            if args.gpu:
                for i in range(len(intermediate_data)):
                    intermediate_data[i] = intermediate_data[i].cuda(async=True)
            for i in range(len(intermediate_data)):
                intermediate_data[i] = intermediate_data[i].to(device)

            #intermediate_var = torch.autograd.Variable(intermediate_data)

            data_time.update(time.time() - end)

            #output = wholeblock(intermediate_var)
            output = wholeblock(intermediate_data)
            class_result = output[0]
            softmax = nn.Softmax(dim=1).to(device)
            confidence = softmax(class_result).max(dim=1, keepdim=False)
            #print(confidence[0].shape)
            further_data = output[1]
            #print(further_data)
            if args.evalblock < args.nBlocks-1 and confidence[0] < 2:
                for j in range(len(further_data)):
                    if args.gpu:
                        further_data[j] = further_data[j].cpu()
                    dist.send(further_data[j], dst=args.evalblock+1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if count % args.print_freq == 0:
                print('Epoch: [{0}/{1}]\t'
                      'Time {batch_time.avg:.3f}\t'
                      'Data {data_time.avg:.3f}'.format(
                        count + 1,max_count,
                        batch_time=batch_time, data_time=data_time))
            count += 1


def save_checkpoint(state, args, is_best, filename, result):
    print(args)
    result_filename = os.path.join(args.save, 'scores.tsv')
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = os.path.join(model_dir, filename)
    best_filename = os.path.join(model_dir, 'model_best.pth.tar')
    os.makedirs(args.save, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    print("=> saving checkpoint '{}'".format(model_filename))

    torch.save(state, model_filename)

    with open(result_filename, 'w') as f:
        print('\n'.join(result), file=f)

    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    if is_best:
        shutil.copyfile(model_filename, best_filename)

    print("=> saved checkpoint '{}'".format(model_filename))
    return

def load_checkpoint(args):
    model_dir = os.path.join(args.save, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    if os.path.exists(latest_filename):
        with open(latest_filename, 'r') as fin:
            model_filename = fin.readlines()[0].strip()
    else:
        return None
    print("=> loading checkpoint '{}'".format(model_filename))
    state = torch.load(model_filename)
    print("=> loaded checkpoint '{}'".format(model_filename))
    return state

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precor@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def adjust_learning_rate(optimizer, epoch, args, batch=None,
                         nBatch=None, method='multistep'):
    if method == 'cosine':
        T_total = args.epochs * nBatch
        T_cur = (epoch % args.epochs) * nBatch + batch
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        if args.data.startswith('cifar'):
            lr, decay_rate = args.lr, 0.1
            if epoch >= args.epochs * 0.75:
                lr *= decay_rate ** 2
            elif epoch >= args.epochs * 0.5:
                lr *= decay_rate
        else:
            lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    main()
