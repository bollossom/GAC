import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast
import torch.cuda.amp
from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader
from functions import TET_loss
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
def train(epoch, args):
    running_loss = 0
    start = time.time()
    net.train()
    correct = 0.0
    num_sample = 0
    for batch_index, (images, labels) in enumerate(ImageNet_training_loader):
        if args.gpu:
            labels = labels.cuda(non_blocking=True)
            images = images.cuda(non_blocking=True)
        num_sample += images.size()[0]
        optimizer.zero_grad()
        r = np.random.rand(1)
        with autocast():
            if args.beta > 0 and r < args.cutmix_prob:
                lam = np.random.beta(args.beta, args.beta)
                rand_index = torch.randperm(images.size()[0]).cuda()
                target_a = labels
                target_b = labels[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                    # adjust lambda to exactly match pixel ratio
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
                    # compute output
                outputs = net(images)
                mean_out = outputs.mean(1)
                loss = criterion(mean_out, target_a) * lam + criterion(mean_out,target_b) * (1. - lam)
            else:
                # compute output
                outputs = net(images)
                mean_out = outputs.mean(1)
                loss = criterion(mean_out, labels)
            _, predicted = mean_out.cpu().max(1)
            correct += float(predicted.eq(labels.cpu()).sum().item())
            running_loss += loss.item()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        n_iter = (epoch - 1) * len(ImageNet_training_loader) + batch_index + 1

    finish = time.time()
    if args.local_rank == 0:
        writer.add_scalar('Train/acc', correct / num_sample * 100, epoch)
    print("Training accuracy: {:.2f} of epoch {}".format(
        correct / num_sample * 100, epoch))
    print('epoch {} training time consumed: {:.2f}s'.format(
        epoch, finish - start))


@torch.no_grad()
def eval_training(epoch, args):

    start = time.time()
    net.eval()

    test_loss = 0.0
    correct = 0.0
    real_batch = 0
    for (images, labels) in ImageNet_test_loader:
        real_batch += images.size()[0]
        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        mean_out = outputs.mean(1)
        

        loss = TET_loss(outputs,labels,criterion,1.0,1e-3)
        # criterion(mean_out,labels)
        test_loss += loss.item()

        _, predicted = mean_out.cpu().max(1)
        correct += float(predicted.eq(labels.cpu()).sum().item())

    finish = time.time()
    print('Evaluating Network.....')
    print(
        'Test set: Average loss: {:.4f}, Accuracy: {:.4f}%, Time consumed:{:.2f}s'
        .format(test_loss * args.b / len(ImageNet_test_loader.dataset),
                correct / real_batch * 100, finish - start))
    
    if args.local_rank == 0:
        # add information to tensorboard
        writer.add_scalar(
            'Test/Average loss',
            test_loss * args.b / len(ImageNet_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy',
                          correct / real_batch * 100, epoch)

    return correct / len(ImageNet_test_loader.dataset)


# for resnet-104
class CrossEntropyLabelSmooth(nn.Module):

    def __init__(self, num_classes=1000, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1),
                                                       1)
        targets = (1 -
                   self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss



    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu',
                        action='store_true',
                        default=True,
                        help='use gpu or not')
    parser.add_argument('-b',
                        type=int,
                        default=256,
                        help='batch size for dataloader')
    parser.add_argument('-lr',
                        type=float,
                        default=0.1,
                        help='initial learning rate')
    parser.add_argument('--local_rank',
                        default=-1,
                        type=int,
                        help='node rank for distributed training')
    parser.add_argument('--beta', default=1.0, type=float,
                    help='hyperparameter beta')
    parser.add_argument('--cutmix_prob', default=1.0, type=float,
                    help='cutmix probability')
    args = parser.parse_args()
    print(args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)

    SEED = 445 #445
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    net = get_network(args)
    net.cuda()
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    net = torch.nn.parallel.DistributedDataParallel(
        net, device_ids=[args.local_rank])

    # to load a pretrained model
    # map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}
    # net.load_state_dict(
    #     torch.load("path", map_location=map_location))
    
    num_gpus = torch.cuda.device_count()
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    print(num_gpus)
    # data preprocessing:
    ImageNet_training_loader = get_training_dataloader(
        traindir="/dataset/ImageNet2012/train",
        num_workers=16,
        batch_size=args.b // num_gpus,
        shuffle=False,
        sampler=1  # to enable sampler for DDP
    )

    ImageNet_test_loader = get_test_dataloader(valdir="/dataset/ImageNet2012/val",
                                               num_workers=16,
                                               batch_size=args.b // num_gpus,
                                               shuffle=False,
                                               sampler=1)
    # learning rate should go with batch size.
    b_lr = args.lr

    criterion = TET_loss().cuda()
    # criterion =CrossEntropyLabelSmooth().cuda()
    optimizer = optim.SGD([{
        'params': net.parameters(),
        'initial_lr': b_lr
    }],
                          momentum=0.9,
                          lr=b_lr,
                          weight_decay=1e-5)
    train_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.EPOCH, eta_min=0, last_epoch=0)
    iter_per_epoch = len(ImageNet_training_loader)
    LOG_INFO = "ImageNet_ACC"
    checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net,
                                   str(args.b), str(args.lr), LOG_INFO,
                                   settings.TIME_NOW)

    # use tensorboard
    if args.local_rank == 0:
        if not os.path.exists(settings.LOG_DIR):
            os.mkdir(settings.LOG_DIR)
        writer = SummaryWriter(
            log_dir=os.path.join(settings.LOG_DIR, args.net, str(args.b),
                                 str(args.lr), LOG_INFO, settings.TIME_NOW))

    # create checkpoint folder to save model
    if args.local_rank == 0:
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        checkpoint_path = os.path.join(checkpoint_path,
                                       '{net}-{epoch}-{type}.pth')
    scaler = torch.cuda.amp.GradScaler()
    best_acc = 0.0
    
    epoch = 250
    print(epoch)
    for epoch in range(1, epoch + 1):
        train(epoch, args)

        train_scheduler.step()
        acc = eval_training(epoch, args)
        if  best_acc < acc:
            
            best_acc = acc
            print(epoch,best_acc)
            torch.save(net.state_dict(), '.pth')
