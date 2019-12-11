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

if args.blockids:
    args.blockids = list(map(int, args.blockids.split('-')))

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
        else:
            model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.resume and args.evalmode is None:
        checkpoint = load_checkpoint(args)
        if checkpoint is not None:
            args.start_epoch = checkpoint['epoch'] + 1
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])

    cudnn.benchmark = True

    train_loader, val_loader, test_loader = get_dataloaders(args)

    if args.evalmode is not None:
        state_dict = torch.load(args.evaluate_from, map_location=device)['state_dict']
        try:
            model.load_state_dict(state_dict)
        except:
            if args.gpu is None:
                torch.nn.DataParallel(model).load_state_dict(state_dict)  #for cpu running setting loads gpu learned model
            else:
                model.module.load_state_dict(state_dict) #for gpu running setting loads cpu learned model
        if args.blockids is not None:
            assert args.blockids[-1] < args.nBlocks
            dist.init_process_group(backend='gloo', init_method="tcp://"+args.master, rank=args.blockrank, world_size=args.worldsize)
            block_list = []
            if args.gpu is not None:
                if len(args.blockids)>1:
                    for i in range(args.blockids[0], args.blockids[1]+1):
                        block = model.module.get_block(i)
                        block_list.append(torch.nn.DataParallel(block).to(device))
                else:
                    block = model.module.get_block(args.blockids[0])
                    block_list.append(torch.nn.DataParallel(block).to(device))
                classifier = model.module.get_classifier(args.blockids[-1])
                classifier = torch.nn.DataParallel(classifier).to(device)
                dims = model.module.get_dims()
            else:
                if len(args.blockids)>1:
                    for i in range(args.blockids[0], args.blockids[1]+1):
                        block = model.get_block(i)
                        block_list.append(block.to(device))
                else:
                    block = model.get_block(args.blockids[0])
                    block_list.append(block)
                classifier = model.get_classifier(args.blockids[-1])
                dims = model.get_dims()
            if args.blockids[0] == 0:
                validate_block(val_loader, block_list, classifier, criterion)
            else:
                validate_block2(block_list, classifier, dims)
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

        val_loss, val_prec1, val_prec5 = validate(val_loader, model, criterion, epoch)

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

def convert_to_sparse_send(dense_tensor, dst):
    #num_of_dim = torch.tensor(len(dense_tensor.shape), dtype=torch.uint8)
    #dist.send(num_of_dim, dst=dst)
    #dims = torch.tensor(dense_tensor.shape, dtype=torch.int16)
    #dist.send(dims, dst=dst)
    sparse_tensor = dense_tensor.to_sparse()
    nnz = torch.tensor(sparse_tensor._nnz(), dtype=torch.int32)
    dist.send(nnz, dst=dst)
    if nnz!=0:
        indices = sparse_tensor._indices().type(torch.int16)
        dist.send(indices, dst=dst)
        values = sparse_tensor._values()
        dist.send(values, dst=dst)
    
def receive_sparse_convert(dims, src=None):
    #num_of_dim = torch.tensor(0, dtype=torch.uint8)
    #dist.recv(num_of_dim, src=src)
    #num_of_dim = int(num_of_dim)
    #dims = torch.zeros(num_of_dim, dtype=torch.int16)
    #dist.recv(dims, src=src)
    #dims = dims.tolist()
    nnz = torch.tensor(0, dtype=torch.int32)
    dist.recv(nnz, src=src)
    nnz = int(nnz)
    if nnz==0:
        return torch.zeros(dims, dtype=torch.float32)
    else:
        indices = torch.zeros([len(dims), nnz], dtype=torch.int16)
        dist.recv(indices, src=src)
        values = torch.zeros(nnz, dtype=torch.float32)
        dist.recv(values, src=src)
        return torch.sparse.FloatTensor(indices.type(torch.int64), values, dims).to_dense()

def combine_intermediate_data(intermediate_data):
    if len(intermediate_data) == 1:
        return intermediate_data[0]
    elif len(intermediate_data) == 2:
        dim = list(intermediate_data[1].shape)
        dim[1] = int(dim[1]/4)
        dim[2] *= 2
        dim[3] *= 2
        return torch.cat((intermediate_data[0], intermediate_data[1].reshape(dim)), 1)
    elif len(intermediate_data) == 3:
        dim1 = list(intermediate_data[1].shape)
        dim1[1] = int(dim1[1]/4)
        dim1[2] *= 2
        dim1[3] *= 2
        dim2 = list(intermediate_data[2].shape)
        dim2[1] = int(dim2[1]/16)
        dim2[2] *= 4
        dim2[3] *= 4
        return torch.cat((intermediate_data[0], intermediate_data[1].reshape(dim1), intermediate_data[2].reshape(dim2)), 1)
    else:
        print("combine_intermediate_data input length error")
        exit(-1)

def get_combined_dim(batch_size, dim):
    if len(dim) == 1:
        new_dim = [batch_size, dim[0][1], dim[0][2], dim[0][3]]
        return new_dim
    elif len(dim) == 2:
        new_dim = [batch_size, dim[0][1]+int(dim[1][1]/4), dim[0][2], dim[0][3]]
        return new_dim
    elif len(dim) == 3:
        new_dim = [batch_size, dim[0][1]+int(dim[1][1]/4)+int(dim[2][1]/16), dim[0][2], dim[0][3]]
        return new_dim
    else:
        print("get_combined_dim input length error")
        exit(-1)

def split_intermediate_data(recv_data, dim):
    intermediate_data = []
    if len(dim) == 1:
        intermediate_data.append(recv_data)
        return intermediate_data
    elif len(dim) == 2:
        intermediate_data = list(torch.split(recv_data, [dim[0][1], int(dim[1][1]/4)], dim=1))
        shape = list(intermediate_data[1].shape)
        shape[1] *= 4
        shape[2] = int(shape[2]/2)
        shape[3] = int(shape[3]/2)
        intermediate_data[1] = intermediate_data[1].reshape(shape)
        return intermediate_data
    elif len(dim) == 3:
        intermediate_data = list(torch.split(recv_data, [dim[0][1], int(dim[1][1]/4), int(dim[2][1]/16)], dim=1))
        shape1 = list(intermediate_data[1].shape)
        shape1[1] *= 4
        shape1[2] = int(shape1[2]/2)
        shape1[3] = int(shape1[3]/2)
        intermediate_data[1] = intermediate_data[1].reshape(shape1)
        shape2 = list(intermediate_data[2].shape)
        shape2[1] *= 16
        shape2[2] = int(shape2[2]/4)
        shape2[3] = int(shape2[3]/4)
        intermediate_data[2] = intermediate_data[2].reshape(shape2)
        return intermediate_data
    else:
        print("split_intermediate_data input length error")
        exit(-1)

def combine_conf_class(confidence, class_result):
    conf = torch.clamp(confidence*32767., 0., 32767.).type(torch.int16)
    class_result = class_result.type(torch.int16)
    return torch.cat((torch.unsqueeze(conf,0), torch.unsqueeze(class_result,0)))

def split_conf_class(recv_data):
    conf = recv_data[0].type(torch.float32)/32767.
    class_result = recv_data[1]
    return conf, class_result

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
        #output = []
        loss = 0.0
        
        weight = [10.]*args.nBlocks
        weight[-1] = 0.
        if epoch>=args.epochs*3/4:
            for j in range(len(output)):
                loss += weight[j] * torch.norm(output[j][1], p=1)/output[j][1].numel()
        for j in range(len(output)):
            loss += criterion(output[j][0], target_var)

        losses.update(loss.item(), input.size(0))

        for j in range(len(output)):
            prec1, prec5 = accuracy(output[j][0].data, target, topk=(1, 5))
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

def validate(val_loader, model, criterion, epoch=None):
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
            #output = []
            loss = 0.0

            weight = [10.]*args.nBlocks
            weight[-1] = 0.
            if epoch is not None and epoch>=args.epochs*3/4:
                for j in range(len(output)):
                    loss += weight[j] * torch.norm(output[j][1], p=1)/output[j][1].numel()
            for j in range(len(output)):
                loss += criterion(output[j][0], target_var)

            losses.update(loss.item(), input.size(0))

            for j in range(len(output)):
                prec1, prec5 = accuracy(output[j][0].data, target, topk=(1, 5))
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
    return losses.avg, top1[-1].avg, top5[-1].avg

def validate_block(val_loader, block_list, classifier, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    for block in block_list:
        block.eval()
    classifier.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            if args.gpu:
                target = target.cuda(async=True)
            input = input.to(device)

            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            data_time.update(time.time() - end)

            if len(block_list)==1:
                intermediate_data = block_list[0](input_var)
            else:
                intermediate_data = input_var
                for block in block_list:
                    intermediate_data = block(intermediate_data)
            class_result = classifier(intermediate_data)
            softmax = nn.Softmax(dim=1).to(device)
            confidence = softmax(class_result).max(dim=1, keepdim=False)

            ids = torch.zeros(args.batch_size, dtype=torch.int32)
            for j in range(args.batch_size):
                ids[j] = i * args.batch_size + j
            idx = confidence.values < args.confidence
            if len(confidence.values[idx]) > 0:
                batch_size = torch.tensor(len(confidence.values[idx]), dtype=torch.int8)
                dist.send(batch_size, dst=1)
                dist.send(ids[idx], dst=1)
                for j in range(len(intermediate_data)):
                    intermediate_data[j] = intermediate_data[j][idx]
                    #if args.gpu:
                        #intermediate_data[j] = intermediate_data[j].cpu()
                    #dist.send(intermediate_data[j][idx], dst=1)
                    #convert_to_sparse_send(intermediate_data[j][idx], dst=1)
                send_data = combine_intermediate_data(intermediate_data)
                if args.gpu:
                    send_data = send_data.cpu()
                convert_to_sparse_send(send_data, dst=1)

                count = len(confidence.values[idx])
                while count > 0:
                    dist.recv(batch_size)
                    ids = torch.zeros(batch_size, dtype=torch.int32)
                    dist.recv(ids)
                    #conf = torch.zeros(batch_size, dtype=torch.float32)
                    #dist.recv(conf)
                    #final_classification = torch.zeros(batch_size, dtype=torch.int64)
                    #dist.recv(final_classification)
                    recv_data = torch.zeros((2,batch_size), dtype=torch.int16)
                    dist.recv(recv_data)
                    conf, final_classification = split_conf_class(recv_data)
                    count -= int(batch_size)

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

def validate_block2(block_list, classifier, dims):
    batch_time = AverageMeter()
    losses = AverageMeter()
    data_time = AverageMeter()

    top1 = AverageMeter()
    top5 = AverageMeter()

    for block in block_list:
        block.eval()
    classifier.eval()

    end = time.time()
    with torch.no_grad():
        count = 0
        max_count = 10000;
        while count<max_count:
            intermediate_data = []
            batch_size = torch.tensor(0, dtype=torch.int8)
            dist.recv(batch_size, src=args.blockrank-1)
            ids = torch.zeros(batch_size, dtype=torch.int32)
            dist.recv(ids, src=args.blockrank-1)
            dim = get_combined_dim(int(batch_size), dims[args.blockids[0]])
            recv_data = receive_sparse_convert(dim, src=args.blockrank-1)
            intermediate_data = split_intermediate_data(recv_data, dims[args.blockids[0]])
            if args.gpu:
                for i in range(len(intermediate_data)):
                    intermediate_data[i] = intermediate_data[i].cuda(async=True)
            for i in range(len(intermediate_data)):
                intermediate_data[i] = intermediate_data[i].to(device)

            data_time.update(time.time() - end)

            if len(block_list)==1:
                further_data = block_list[0](intermediate_data)
            else:
                further_data = intermediate_data
                for block in block_list:
                    further_data = block(further_data)
            class_result = classifier(further_data)
            softmax = nn.Softmax(dim=1).to(device)
            confidence = softmax(class_result).max(dim=1, keepdim=False)

            idx = confidence.values < args.confidence
            if args.blockids[-1] < args.nBlocks-1 and len(confidence.values[idx]) > 0:
                batch_size = torch.tensor(len(confidence.values[idx]), dtype=torch.int8)
                dist.send(batch_size, dst=args.blockrank+1)
                dist.send(ids[idx], dst=args.blockrank+1)
                for j in range(len(further_data)):
                    further_data[j] = further_data[j][idx]
                send_data = combine_intermediate_data(further_data)
                if args.gpu:
                    send_data = send_data.cpu()
                convert_to_sparse_send(send_data, dst=args.blockrank+1)

            idx = confidence.values >= args.confidence
            if args.blockids[-1] == args.nBlocks-1 or len(confidence.values[idx]) > 0:
                if args.blockids[-1] == args.nBlocks-1:
                    batch_size = torch.tensor(len(confidence.values), dtype=torch.int8)
                else:
                    batch_size = torch.tensor(len(confidence.values[idx]), dtype=torch.int8)
                dist.send(batch_size, dst=0)
                if args.gpu:
                    class_conf = confidence.values.cpu()
                    class_result = confidence.indices.cpu()
                else:
                    class_conf = confidence.values
                    class_result = confidence.indices
                if args.blockids[-1] == args.nBlocks-1:
                    dist.send(ids, dst=0)
                    send_data = combine_conf_class(class_conf, class_result)
                else:
                    dist.send(ids[idx], dst=0)
                    send_data = combine_conf_class(class_conf[idx], class_result[idx])
                dist.send(send_data, dst=0)

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
    state = torch.load(model_filename, map_location=device)
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

