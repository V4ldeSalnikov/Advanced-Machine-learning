#importing project functions from other .py files
from train import *
from priors import *
from vae import *
from support import *
from flow import *
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
parser.add_argument('mode', type=str, default='train', choices=['train', 'evaluate', 'sample', 'plot'], help='what to do when running the script (default: %(default)s)')
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
#flow prior
base = GaussianBase(M)
#transformations
transformations =[]
mask = torch.Tensor([1 if (i+j) % 2 == 0 else 0 for i in range(28) for j in range(28)])
    
num_transformations = 5
num_hidden = 8

# Make a mask that is 1 for the first half of the features and 0 for the second half
mask = torch.zeros((M,))
mask[M//2:] = 1
    
for i in range(num_transformations):
    mask = (1-mask) # Flip the mask
    scale_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M))
    translation_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M))
    transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

# Define flow model
prior_flow = Flow(base, transformations).to(args.device)

#helper to create independent encoder/decoder instances per model
def make_encoder_decoder(M):
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 784),
        nn.ReLU(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, M*2),
    )
    decoder_net = nn.Sequential(
        nn.Linear(M, 128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.ReLU(),
        nn.Linear(784, 784),
        nn.Unflatten(-1, (28, 28))
    )
    return GaussianEncoder(encoder_net), BernoulliDecoder(decoder_net)

#declaring VAE models, each with its own encoder and decoder
encoder_Gaus, decoder_Gaus = make_encoder_decoder(M)
encoder_MoG,  decoder_MoG  = make_encoder_decoder(M)
encoder_Flow, decoder_Flow  = make_encoder_decoder(M)
model_Gaus = VAE_KL(prior_Gaus, decoder_Gaus, encoder_Gaus).to(device)
model_MoG = VAE_Monte(prior_MoG, decoder_MoG, encoder_MoG).to(device)
model_Flow = VAE_Monte(prior_flow, decoder_Flow, encoder_Flow).to(device)
#Chocie of the mode
if args.mode == 'train':
    #Defining optimizers
    optimizer_Gaus = torch.optim.Adam(model_Gaus.parameters(), lr=1e-3)
    optimizer_MoG= torch.optim.Adam(model_MoG.parameters(), lr=1e-3)
    optimizer_Flow = torch.optim.Adam(model_Flow.parameters(), lr=1e-3)
    #Training models
    train(model_Gaus, optimizer_Gaus, mnist_train_loader, args.epochs, args.device)
    train(model_MoG, optimizer_MoG, mnist_train_loader, args.epochs, args.device)
    train(model_Flow, optimizer_Flow, mnist_train_loader, args.epochs, args.device)
    #Saving models
    torch.save(model_Gaus.state_dict(), args.model + '_Gaus.pt')
    torch.save(model_MoG.state_dict(), args.model + '_MoG.pt')
    torch.save(model_Flow.state_dict(), args.model + '_Flow.pt')
elif args.mode == 'evaluate':
    #loading models
    model_Gaus.load_state_dict(torch.load(args.model + '_Gaus.pt', map_location=torch.device(args.device)))
    model_MoG.load_state_dict(torch.load(args.model + '_MoG.pt', map_location=torch.device(args.device)))
    model_Flow.load_state_dict(torch.load(args.model + '_Flow.pt', map_location=torch.device(args.device))) 
    #evaluating models on test set
    ll_Gaus = evaluate_test_elbo(model_Gaus, mnist_test_loader, device)
    ll_MoG = evaluate_test_elbo(model_MoG, mnist_test_loader, device)
    ll_Flow = evaluate_test_elbo(model_Flow, mnist_test_loader, device)
    #Evaluate models
    print(f"log-likelihood ELBO Gaussian Prior: {ll_Gaus:.4f}")
    print(f"log-likelihood ELBO Mixture of Gaussians Prior: {ll_MoG:.4f}")
    print(f"log-likelihood ELBO Flow Prior: {ll_Flow:.4f}")
elif args.mode == 'sample':
    #loading models
    model_Gaus.load_state_dict(torch.load(args.model + '_Gaus.pt', map_location=torch.device(args.device)))
    model_MoG.load_state_dict(torch.load(args.model + '_MoG.pt', map_location=torch.device(args.device)))
    model_Flow.load_state_dict(torch.load(args.model + '_Flow.pt', map_location=torch.device(args.device)))
    #Generating samples
    model_Gaus.eval()
    model_MoG.eval()
    model_Flow.eval()
    with torch.no_grad():
        #generating
        samples_Gaus = (model_Gaus.sample(64)).cpu() 
        samples_MoG = (model_MoG.sample(64)).cpu() 
        samples_Flow = (model_Flow.sample(64)).cpu()
        #saving
        save_image(samples_Gaus.view(64, 1, 28, 28), args.samples + '_Gaus.png')
        save_image(samples_MoG.view(64, 1, 28, 28), args.samples + '_MoG.png')
        save_image(samples_Flow.view(64, 1, 28, 28), args.samples + '_Flow.png')
elif args.mode == 'plot':
        model_Gaus.load_state_dict(torch.load(args.model + '_Gaus.pt', map_location=torch.device(args.device)))
        model_MoG.load_state_dict(torch.load(args.model + '_MoG.pt', map_location=torch.device(args.device)))
        model_Flow.load_state_dict(torch.load(args.model + '_Flow.pt', map_location=torch.device(args.device)))

        # 10% of test set 
        # plot_dataset = mnist_test_loader.dataset
        # n_plot = max(1, len(plot_dataset) // 10)
        # plot_subset = torch.utils.data.Subset(plot_dataset, range(n_plot))
        # plot_loader = torch.utils.data.DataLoader(plot_subset, batch_size=args.batch_size, shuffle=False)

        # Plot posterior samples
        print("\nPlotting approximate posterior samples...")
        plot_posterior_samples(model_Gaus, mnist_test_loader, args.device, save_path='plot_Gaus.png')
        plot_posterior_samples(model_MoG,  mnist_test_loader, args.device, save_path='plot_MoG.png')
        plot_posterior_samples(model_Flow, mnist_test_loader, args.device, save_path='plot_Flow.png')