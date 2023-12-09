import argparse
import os
import torch
from models.MS_ResNet import *
import data_loaders
from functions import seed_all

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='PyTorch Gated Attention Coding')
parser.add_argument('--start_epoch',
                    default=0,
                    type=int,
                    metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b',
                    '--batch_size',
                    default=32,
                    type=int,
                    metavar='N',
                    help='mini-batch size (default: 64), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')


parser.add_argument('--seed',
                    default=1000,
                    type=int,
                    help='seed for initializing training. ')
parser.add_argument('-T',
                    '--time',
                    default=6,
                    type=int,
                    metavar='N',
                    help='snn simulation time steps (default: 2)')
parser.add_argument('-j',
                    '--workers',
                    default=16,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 10)')

args = parser.parse_args()

@torch.no_grad()
def test(model, test_loader, device):
    correct = 0
    total = 0
    model.eval()
    for batch_idx, (inputs, targets) in enumerate(test_loader):
        inputs = inputs.to(device)
        outputs = model(inputs)
        mean_out = outputs.mean(1)
        _, predicted = mean_out.cpu().max(1)
        total += float(targets.size(0))
        correct += float(predicted.eq(targets).sum().item())
        if batch_idx % 100 == 0:
            acc = 100. * float(correct) / float(total)
            print(batch_idx, len(test_loader), ' Acc: %.5f' % acc)
    final_acc = 100 * correct / total
    return final_acc


if __name__ == '__main__':

    seed_all(args.seed)

    train_dataset, val_dataset = data_loaders.build_cifar(use_cifar10=False)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)
    parallel_model = msresnet18(num_classes=100)
    parallel_model.T = 6
    parallel_model = torch.nn.DataParallel(parallel_model)
    parallel_model.to(device)
    # load pretain_model
    state_dict = torch.load('CIFAR100_T=6.pth')
    parallel_model.module.load_state_dict(state_dict, strict=False)

    print("Parameter numbers: {}".format(
        sum(p.numel() for p in parallel_model.parameters())))
    facc = test(parallel_model, test_loader, device)
    print('Test acc={:.3f}'.format(facc))


