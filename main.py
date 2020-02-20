import argparse
import os
import sys
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import compression

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--data', default='./data', type=str,
                    help='path to dataset')
parser.add_argument('--dataset', default='cifar10', type=str,
                    help='(cifar10, cifar100, imagenet)')
parser.add_argument('-a', '--arch', default='resnet56',
                    help='model architecture (default: resnet56)')
parser.add_argument('-j', '--workers', default=4, type=int,
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', default=0.1, type=float,
                    help='initial learning rate')
parser.add_argument('-ds', '--decay-schedule', type=int, nargs='+', default=[100,150],
                    help='learning rate decaying epochs')
parser.add_argument('-opt', '--optimizer', type=str, default='sgd',
                    help='choose from (sgd, adagrad)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='momentum')
parser.add_argument('-wd', '--weight-decay', default=5e-4, type=float,
                    help='weight decay (default: 5e-4)')
parser.add_argument('-p', '--print-freq', default=391, type=int,
                    help='print frequency (default: 391)')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--root', default=0, type=int,
                    help='root node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')



parser.add_argument('--method', default='va', type=str,
                    help='choose from (va, va-Sign, va-TopK, va-GSpar, ' +
                                          'ef-Sign, ef-TopK, ef-GSpar, ' +
                                           'r-Sign,  r-TopK,  r-GSpar)')
parser.add_argument('--spar', default=0.1, type=float,
                    help='sparsity.')
parser.add_argument('-rp', '--reset-period', default=0, type=int,
                    help='reset residuals period')
parser.add_argument('-ap', '--average-period', default=0, type=int,
                    help='arverage residuals period')
parser.add_argument('-cb', '--compress-back', action='store_true',
                    help='compress the message the server sends back')
parser.add_argument('-sec', '--server-error-compensate', action='store_true',
                    help='server error compensation')
parser.add_argument('--eval-ref', action='store_true',
                    help='evaluate validation of reference point')
parser.add_argument('--eval-r', action='store_true',
                    help='evalueate training of dist-r point')

best_acc1 = 0


def main():
    args = parser.parse_args()
    print(args, flush=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    ngpus_per_node = torch.cuda.device_count()
    # Since we have ngpus_per_node processes per node, the total world_size
    # needs to be adjusted accordingly
    args.world_size = ngpus_per_node * args.world_size
    # Use torch.multiprocessing.spawn to launch distributed processes: the
    # main_worker process function
    mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # For multiprocessing distributed training, rank needs to be the
    # global rank among all the processes
    args.rank = args.rank * ngpus_per_node + gpu
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size, rank=args.rank)
    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.dataset == 'cifar10':
        import models
        model = models.__dict__[args.arch](num_classes=10)
    elif args.dataset == 'cifar100':
        import models
        model = models.__dict__[args.arch](num_classes=100)
    elif args.dataset == 'imagenet':
        import torchvision.models as models
        model = models.__dict__[args.arch]()
    else:
        assert False

    torch.cuda.set_device(args.gpu)
    model.cuda(args.gpu)

    # sync the model among workers
    for k, v in model.state_dict().items():
        dist.broadcast(v.data, src=args.root)

    args.batch_size = int(args.batch_size / args.world_size)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), args.lr,
                                        weight_decay=args.weight_decay)
    else:
        assert False

    cudnn.benchmark = True

    # Data loading code
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                         std=[0.247, 0.243, 0.262])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose([transforms.ToTensor(), normalize])

        _dataset = datasets.CIFAR10 if args.dataset == 'cifar10' else datasets.CIFAR100
        train_dataset = _dataset(root=args.data, train=True, download=True,
                                 transform=train_transform)
        val_dataset = _dataset(root=args.data, train=False, download=True,
                               transform=val_transform)
    elif args.dataset == 'imagenet':
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # residuals for dist-rsgd
    old_p = [p.data.clone().detach().zero_() for p in model.parameters()]
    residuals = [p.data.clone().detach().zero_() for p in model.parameters()]
    s_residuals = [p.data.clone().detach().zero_() for p in model.parameters()]

    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        lr = adjust_learning_rate(optimizer, epoch, args)
        print('[worker '+str(args.rank)+']',
              'Current lr: {:.6f}'.format(lr), flush=True)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args,
              old_p, residuals, s_residuals, val_loader)

        # evaluate on validation set
        if args.method.startswith('va'):
            acc1 = validate(val_loader, model, criterion, args,
                            prefix='[worker '+str(args.rank)+']')
        elif args.method.startswith('ef-'):
            acc1 = validate(val_loader, model, criterion, args,
                            prefix='[worker '+str(args.rank)+']')
        elif args.method.startswith('r-'):
            acc1 = validate(val_loader, model, criterion, args,
                            prefix='[worker '+str(args.rank)+'][dist-r]')

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        if is_best:
            print('[worker '+str(args.rank)+']',
                  'Current best acc@1: {:6.2f}'.format(acc1), flush=True)


def train(train_loader, model, criterion, optimizer, epoch, args,
          old_p, residuals, s_residuals, val_loader):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    if args.eval_r:
        losses_r = AverageMeter('Loss', ':.4e')
        top1_r = AverageMeter('Acc@1', ':6.2f')
        top5_r = AverageMeter('Acc@5', ':6.2f')
        progress_r = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses_r, top1_r, top5_r],
            prefix="Epoch: [{}]".format(epoch))

    p_names = [name for name, p in model.named_parameters()]
    delta_p = [p.data.clone().detach().zero_() for p in model.parameters()]

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        cur_iter = i + epoch * len(train_loader)
        data_time.update(time.time() - end)

        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        optimizer.zero_grad()

        # step ahead in dist-rsgd
        if args.method.startswith('r-'):
            for idx, p in enumerate(model.parameters()):
                if args.reset_period and (cur_iter+1) % args.reset_period == 0:
                    residuals[idx].zero_()

                if args.average_period and (cur_iter+1) % args.average_period == 0:
                    for res_ in residuals:
                        dist.all_reduce(res_, op=dist.ReduceOp.SUM)
                        res_ /= args.world_size

                p.data.sub_(residuals[idx])

            if args.eval_ref and i == len(train_loader)-1: # validate ref (auxiliary)
                validate(val_loader, model, criterion, args,
                         prefix='[worker '+str(args.rank)+'][ref]')
                model.train()

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        loss.backward()

        with torch.no_grad():
            for k, v in model.state_dict().items():
                if not k in p_names: # broadcast non-param buffers from rank 0
                    dist.broadcast(v.data, src=args.root)

            if args.method.startswith('va'):
                # compress
                if args.method.startswith('va-'):
                    func = compression.__dict__[args.method[3:]]
                    func([p.grad for p in model.parameters()], spar=args.spar)
                # reduce
                for p in model.parameters():
                    dist.reduce(p.grad, dst=args.root, op=dist.ReduceOp.SUM)
                    if args.rank == args.root:
                        p.grad /= args.world_size
                # compress back
                if args.compress_back and args.method.startswith('va-') and args.rank == args.root:
                    func([p.grad for p in model.parameters()], spar=args.spar)
                # broadcast
                for p in model.parameters():
                    dist.broadcast(p.grad, src=args.root)
                optimizer.step()


            elif args.method.startswith('ef-'):
                optimizer.step()
                for idx, p in enumerate(model.parameters()):
                    delta_p[idx] = old_p[idx] - p.data + residuals[idx]

                # compress delta_p, send to server
                func = compression.__dict__[args.method[3:]]
                func(delta_p, spar=args.spar)
                for idx, p in enumerate(model.parameters()):
                    residuals[idx] = (old_p[idx] - p.data + residuals[idx]) - delta_p[idx]
                    dist.reduce(delta_p[idx], dst=args.root, op=dist.ReduceOp.SUM)
                    if args.rank == args.root:
                        delta_p[idx] /= args.world_size

                # compress at server side, broadcast back
                if args.rank == args.root and args.compress_back:
                    if args.server_error_compensate:
                        for idx, p in enumerate(model.parameters()):
                            s_residuals[idx] += delta_p[idx]
                            delta_p[idx].copy_(s_residuals[idx])
                    func(delta_p, spar=args.spar)
                    if args.server_error_compensate:
                        for idx, p in enumerate(model.parameters()):
                            s_residuals[idx] -= delta_p[idx]
                for idx, p in enumerate(model.parameters()):
                    dist.broadcast(delta_p[idx], src=args.root)

                # update at worker side
                for idx, p in enumerate(model.parameters()):
                    p.data.copy_(old_p[idx] - delta_p[idx])
                    old_p[idx].copy_(p.data)


            elif args.method.startswith('r-'):
                optimizer.step()
                for idx, p in enumerate(model.parameters()):
                    delta_p[idx] = old_p[idx] - p.data

                # compress delta_p, send to server
                func = compression.__dict__[args.method[2:]]
                func(delta_p, spar=args.spar)
                for idx, p in enumerate(model.parameters()):
                    residuals[idx] = (old_p[idx] - p.data) - delta_p[idx]
                    dist.reduce(delta_p[idx], dst=args.root, op=dist.ReduceOp.SUM)
                    if args.rank == args.root:
                        delta_p[idx] /= args.world_size

                # compress at server side, broadcast back
                if args.rank == args.root and args.compress_back:
                    if args.server_error_compensate:
                        for idx, p in enumerate(model.parameters()):
                            s_residuals[idx] += delta_p[idx]
                            delta_p[idx].copy_(s_residuals[idx])
                    func(delta_p, spar=args.spar)
                    if args.server_error_compensate:
                        for idx, p in enumerate(model.parameters()):
                            s_residuals[idx] -= delta_p[idx]
                for idx, p in enumerate(model.parameters()):
                    dist.broadcast(delta_p[idx], src=args.root)

                # update at worker side
                for idx, p in enumerate(model.parameters()):
                    p.data.copy_(old_p[idx] - delta_p[idx])
                    old_p[idx].copy_(p.data)

                # evaluate training of ref
                if args.eval_r:
                    model.eval()
                    output_r = model(images)
                    loss_r = criterion(output_r, target)
                    acc1_r, acc5_r = accuracy(output_r, target, topk=(1, 5))

                    losses_r.update(loss_r.item(), images.size(0))
                    top1_r.update(acc1_r[0], images.size(0))
                    top5_r.update(acc5_r[0], images.size(0))
                    model.train()


            else:
                assert False

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i+1) % args.print_freq == 0:
            if args.method.startswith('va'):
                progress.display(i, prefix='[worker '+str(args.rank)+']')
            elif args.method.startswith('ef-'):
                progress.display(i, prefix='[worker '+str(args.rank)+']')
            elif args.method.startswith('r-'):
                if args.eval_r:
                    progress_r.display(i, prefix='[worker '+str(args.rank)+'][dist-r]')
                progress.display(i, prefix='[worker '+str(args.rank)+'][ref]')


def validate(val_loader, model, criterion, args, prefix=''):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        progress.display(i, prefix=prefix)

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=''):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, prefix=''):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print(prefix, '\t'.join(entries), flush=True)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    assert len(args.decay_schedule) >= 2

    if epoch < args.decay_schedule[0]:
        lr = args.lr
    elif epoch < args.decay_schedule[1]:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
