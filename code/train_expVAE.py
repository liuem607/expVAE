import argparse
import torch
import torch.optim as optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

import os
import shutil
import numpy as np

from model import ConvVAE
import OneClassMnist
cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = torch.device("cuda" if cuda else "cpu")


def loss_function(recon_x, x, mu, logvar):
    # reconstruction loss
    BCE = F.binary_cross_entropy(recon_x.view(-1, 784), x.view(-1, 784), reduction='sum')

    # KL divergence loss
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

### Training #####
def train(epoch, model, train_loader, optimizer, args):
    model.train()
    train_loss = 0

    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)

        loss = loss_function(recon_batch, data, mu, logvar)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)

    return train_loss

### Validating ####
def test(epoch, model, test_loader, args):
    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)

            recon_batch, mu, logvar = model(data)

            test_loss += loss_function(recon_batch, data, mu, logvar).item()

    test_loss /= len(test_loader.dataset)

    return test_loss


def save_checkpoint(state, is_best, outdir):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    checkpoint_file = os.path.join(outdir, 'checkpoint.pth')
    best_file = os.path.join(outdir, 'model_best.pth')
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, best_file)


def main():
    parser = argparse.ArgumentParser(description='Explainable VAE MNIST Example')
    parser.add_argument('--result_dir', type=str, default='train_results', metavar='DIR',
                        help='output directory')
    parser.add_argument('--ckpt_dir', type=str, default='ckpt', metavar='DIR',
                        help='ckpt directory')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--resume', default=False, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: None')

    # model options
    parser.add_argument('--latent_size', type=int, default=32, metavar='N',
                        help='latent vector size of encoder')
    parser.add_argument('--one_class', type=int, default=3, metavar='N',
                        help='inlier digit for one-class VAE training')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    one_class = args.one_class # Choose the inlier digit to be 3
    one_mnist_train_dataset = OneClassMnist.OneMNIST('./data', one_class, train=True, download=True, transform=transforms.ToTensor())
    one_mnist_test_dataset = OneClassMnist.OneMNIST('./data', one_class, train=False, transform=transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(
        one_mnist_train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader = torch.utils.data.DataLoader(
        one_mnist_test_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = ConvVAE(args.latent_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    start_epoch = 0
    best_test_loss = np.finfo('f').max

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print('=> loading checkpoint %s' % args.resume)
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch'] + 1
            best_test_loss = checkpoint['best_test_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint %s' % args.resume)
        else:
            print('=> no checkpoint found at %s' % args.resume)

    for epoch in range(start_epoch, args.epochs):
        train_loss = train(epoch, model, train_loader, optimizer, args)
        test_loss = test(epoch, model, test_loader,args) 

        print('Epoch [%d/%d] loss: %.3f val_loss: %.3f' % (epoch + 1, args.epochs, train_loss, test_loss))

        is_best = test_loss < best_test_loss
        best_test_loss = min(test_loss, best_test_loss)
        save_checkpoint({
            'epoch': epoch,
            'best_test_loss': best_test_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best, os.path.join('./',args.ckpt_dir))

        # Visualize sample validation result
        with torch.no_grad():
            sample = torch.randn(64, 32).to(device)
            sample = model.decode(sample).cpu()
            img = make_grid(sample)
            save_dir = os.path.join('./',args.result_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            save_image(sample.view(64, 1, 28, 28), os.path.join(save_dir,'sample_' + str(epoch) + '.png'))


if __name__ == '__main__':
    main()