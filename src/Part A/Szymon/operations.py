#importing project functions from other .py files
from train import *
from priors import *
from vae import *
from support import *
#importing torch modules
import torch
import torch.nn as nn
import torch.utils.data
#importing torchvision modules
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

#parse arguments (basically receiving options specyfied in the terminal)
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('mode', type=str, default='train', choices=['train', 'evaluate', 'sample'], help='what to do when running the script (default: %(default)s)')
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
#Defining priors
M = args.latent_dim
prior_Gaus = GaussianPrior(M)
prior_MoG = MoGPrior(M)
#Defining encoder and decoder networks
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
#declaring decoder and encoder
decoder = BernoulliDecoder(decoder_net)
encoder = GaussianEncoder(encoder_net)
#declaring VAE models
model_Gaus = VAE_KL(prior_Gaus, decoder, encoder).to(device)
model_MoG = VAE_Monte(prior_MoG, decoder, encoder).to(device)
#Chocie of the mode
if args.mode == 'train':
    #Defining optimizers
    optimizer_Gaus = torch.optim.Adam(model_Gaus.parameters(), lr=1e-3)
    optimizer_MoG= torch.optim.Adam(model_MoG.parameters(), lr=1e-3)
    #Training models
    train(model_Gaus, optimizer_Gaus, mnist_train_loader, args.epochs, args.device)
    train(model_MoG, optimizer_MoG, mnist_train_loader, args.epochs, args.device)
    #Saving models
    torch.save(model_Gaus.state_dict(), args.model + '_Gaus.pt')
    torch.save(model_MoG.state_dict(), args.model + '_MoG.pt')
elif args.mode == 'evaluate':
    #loading models
    model_Gaus.load_state_dict(torch.load(args.model + '_Gaus.pt', map_location=torch.device(args.device)))
    model_MoG.load_state_dict(torch.load(args.model + '_MoG.pt', map_location=torch.device(args.device)))
    #evaluating models on test set
    ll_Gaus = evaluate_test_elbo(model_Gaus, mnist_test_loader, device)
    ll_MoG = evaluate_test_elbo(model_MoG, mnist_test_loader, device)
    #Evaluate models
    print(f"log-likelihood ELBO Gaussian Prior: {ll_Gaus:.4f}")
    print(f"log-likelihood ELBO Mixture of Gaussians Prior: {ll_MoG:.4f}")
elif args.mode == 'sample':
    #loading models
    model_Gaus.load_state_dict(torch.load(args.model + '_Gaus.pt', map_location=torch.device(args.device)))
    model_MoG.load_state_dict(torch.load(args.model + '_MoG.pt', map_location=torch.device(args.device)))
    #Generating samples
    model_Gaus.eval()
    model_MoG.eval()
    with torch.no_grad():
        #generating
        samples_Gaus = (model_Gaus.sample(64)).cpu() 
        samples_MoG = (model_MoG.sample(64)).cpu() 
        #saving
        save_image(samples_Gaus.view(64, 1, 28, 28), args.samples + '_Gaus.png')
        save_image(samples_MoG.view(64, 1, 28, 28), args.samples + '_MoG.png')