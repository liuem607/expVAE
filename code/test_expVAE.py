import argparse
import torch
from torchvision import datasets, transforms

import os
import numpy as np

from model import ConvVAE
import OneClassMnist
from gradcam import GradCAM
import cv2
from PIL import Image
from torchvision.utils import save_image, make_grid


cuda = torch.cuda.is_available()
if cuda:
    print('cuda available')

device = torch.device("cuda" if cuda else "cpu")

### Save attention maps  ###
def save_cam(image, filename, gcam):
    gcam = gcam - np.min(gcam)
    gcam = gcam / np.max(gcam)
    h, w, d = image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(255 * gcam), cv2.COLORMAP_JET)
    gcam = np.asarray(gcam, dtype=np.float) + \
        np.asarray(image, dtype=np.float)
    gcam = 255 * gcam / np.max(gcam)
    gcam = np.uint8(gcam)
    cv2.imwrite(filename, gcam)

def main():
    parser = argparse.ArgumentParser(description='Explainable VAE MNIST Example')
    parser.add_argument('--result_dir', type=str, default='test_results', metavar='DIR',
                        help='output directory')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')

    # model options
    parser.add_argument('--latent_size', type=int, default=32, metavar='N',
                        help='latent vector size of encoder')
    parser.add_argument('--model_path', type=str, default='./ckpt/model_best.pth', metavar='DIR',
                        help='pretrained model directory')
    parser.add_argument('--one_class', type=int, default=8, metavar='N',
                        help='outlier digit for one-class VAE training')

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    one_class = args.one_class # Choose the current outlier digit to be 8
    one_mnist_test_dataset = OneClassMnist.OneMNIST('./data', one_class, train=False, transform=transforms.ToTensor())

    test_loader = torch.utils.data.DataLoader(
        one_mnist_test_dataset,
        batch_size=args.batch_size, shuffle=False, **kwargs)

    model = ConvVAE(args.latent_size).to(device)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['state_dict'])
    mu_avg, logvar_avg = 0, 1
    gcam = GradCAM(model, target_layer='encoder.2', cuda=True) 
    test_index=0
    for batch_idx, (x, _) in enumerate(test_loader):
        model.eval()
        x = x.to(device)
        x_rec, mu, logvar = gcam.forward(x)

        model.zero_grad()
        gcam.backward(mu, logvar, mu_avg, logvar_avg)
        gcam_map = gcam.generate() 

        ## Visualize and save attention maps  ##
        x = x.repeat(1, 3, 1, 1)
        for i in range(x.size(0)):
            raw_image = x[i] * 255.0
            ndarr = raw_image.permute(1, 2, 0).cpu().byte().numpy()
            im = Image.fromarray(ndarr.astype(np.uint8))
            im_path = args.result_dir
            if not os.path.exists(im_path):
                os.mkdir(im_path)
            im.save(os.path.join(im_path,
                             "{}-{}-origin.png".format(test_index, str(one_class))))

            file_path = os.path.join(im_path,
                                 "{}-{}-attmap.png".format(test_index, str(one_class)))
            r_im = np.asarray(im)
            save_cam(r_im, file_path, gcam_map[i].squeeze().cpu().data.numpy())
            test_index += 1

if __name__ == '__main__':
    main()