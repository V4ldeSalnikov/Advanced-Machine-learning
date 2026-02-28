#
from train import *
from priors import *
from vae import *
#
import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm
#
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
import glob

# Parse arguments
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')

args = parser.parse_args()
print('# Options')
for key, value in sorted(vars(args).items()):
    print(key, '=', value)

device = args.device

# Load MNIST as binarized at 'thresshold' and create data loaders
thresshold = 0.5
mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                batch_size=args.batch_size, shuffle=True)
mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                            transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                batch_size=args.batch_size, shuffle=True)

# Define prior distribution
M = args.latent_dim
prior = MoGPrior(M)

# Define encoder and decoder networks
encoder_net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, M*2),
)

decoder_net = nn.Sequential(
    nn.Linear(M, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, 784),
    nn.Unflatten(-1, (28, 28))
)

# Define VAE model
decoder = BernoulliDecoder(decoder_net)
encoder = GaussianEncoder(encoder_net)
model = VAE_Monte(prior, decoder, encoder).to(device)

# Choose mode to run
if args.mode == 'train':
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Train model
    train(model, optimizer, mnist_train_loader, args.epochs, args.device)

    # Save model
    torch.save(model.state_dict(), args.model)

elif args.mode == 'sample':
    model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

    # Generate samples
    model.eval()
    with torch.no_grad():
        samples = (model.sample(64)).cpu() 
        save_image(samples.view(64, 1, 28, 28), args.samples)