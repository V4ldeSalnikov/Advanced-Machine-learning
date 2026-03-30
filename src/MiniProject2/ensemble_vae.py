# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
import torch.optim as optim
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)


class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()

    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                loss = model(x)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break
def pullback_metric(z, decoder):#it is economic version, i.e. not true pullback metric
    #detach to break gradient from previous computations, clone to create a new tensor, and requires_grad_ to enable gradient computation for z (for later use in autograd)
    z = z.detach().clone().requires_grad_(True) #points in the latent space, shape (N, 2)
    #geting the mean of the decoder at the points z, which will be used to compute the Jacobian
    mean_points = decoder(z).mean
    #mu.shape = (N, D), z.shape = (N, 2)
    num_points= mean_points.shape[0]
    #preallocating the pullback metric tensor, which will have shape (N, 2, 2) since we are in a 2D latent space and the metric is a 2x2 matrix for each point
    metric = torch.zeros(num_points, 2, 2, device=z.device)
    #looping through each point in the latent space
    for i in range(num_points):
        #selecting the i-th point in the latent space, and enabling gradient computation for it
        z_i = z[i:i+1]
        z_i = z_i.clone().requires_grad_(True)#cloning to have new copy and enabling grad
        #evaluating the decoder at the i-th point to get the mean in the data space, which will be used to compute the Jacobian
        mu_i = decoder(z_i).mean
        #computing the Jacobian of the decoder at the i-th point
        jacobian = torch.autograd.grad(mu_i.sum(), z_i)[0][0]  # .sum() to get scalar output for grad, [0][0] to get the Jacobian vector of shape (D,) where D is the dimension of the data space
        #multiplying the Jacobian by its transpose to get the pullback metric at the i-th point
        metric[i] = jacobian[None, :] @ jacobian[:, None]
    #returning the output
    return metric
#some version with computing full Jacobian (was too expensvie for my laptop, but it is more correct(at least I think so). So above version is being run )
#looping through each point in the latent space  
#    for i in range(num_points):
#        #selecting the i-th point in the latent space, and enabling gradient computation for it
#        z_i = z[i:i+1]
#        z_i = z_i.clone().requires_grad_(True)#cloning to have new copy and enabling grad
#        #evaluating the decoder at the i-th point to get the mean in the data space, which will be used to compute the Jacobian
#        mu_i = decoder(z_i).mean
#        #computing the Jacobian of the decoder at the i-th point
#        mu_flat = mu_i.reshape(-1)
#        jacobian = []
#        for d in range(mu_flat.shape[-1]):
#            grad_d = torch.autograd.grad(mu_flat[d], z_i, retain_graph=True)[0][0]
#            jacobian.append(grad_d)
#        jacobian = torch.stack(jacobian, dim=0) 
#        #multiplying the Jacobian by its transpose to get the pullback metric at the i-th point
#        metric[i] = jacobian.T @ jacobian

def plot_metric (metric,rangex,rangey):
    X,Y = torch.meshgrid ((rangex, rangey), indexing='ij')
    XY = torch.concatenate((X.reshape(-1,1),Y.reshape(-1,1)), dim =1)
    G = metric(XY) # NxDxD
    trG = G[:,0,0] + G[:,1,1] # N   
    im = plt.imshow(trG.reshape(X.shape).detach().numpy().T,
               extent=(rangex[0], rangex[-1], rangey[0], rangey[-1]),
               origin="lower")
class PLcurve :
    def __init__ (self,x0,x1,N) :
        """
        Represent the piecewise linear curve connecting x0 to x1 using a
        total of N nodes ( including end - points )
        """
        super(PLcurve,self).__init__ ()
        self.x0 = x0.reshape(1,-1) # 1 xD
        self.x1 = x1.reshape(1,-1) # 1 xD
        self.N = N

        t = torch.linspace(0, 1, N).reshape(N, 1) # Nx1
        c = (1 - t) @ self.x0 + t @ self.x1 # NxD
        self.params = c[1:-1] # (N -2) xD
        self.params.requires_grad = True
    def points(self):
        c = torch.concatenate((self.x0, self.params, self.x1), axis=0) #NxD
        return c
    def plot (self,state,col_cur, label=None):
        #convert to numpy for plotting
        c = self.points().detach().numpy()
        #changing information depending when curve is plotted
        if(state == 'curves before optimization'):
            style = '-'
        elif(state == 'curves after optimization'):
            style = '--o'
        #plotting the curve
        plt.plot(c[:,0],c[:,1],style, color=col_cur, markersize=3, label=label) 


def curve_energy (metric, curve):#energy =  Σᵢ Δzᵢᵀ G(zᵢ) Δzᵢ, where zᵢ are the points along the curve, Δzᵢ are the differences between consecutive points, and G(zᵢ) is the pullback metric at point zᵢ.
    #getting the pullback metrics for the points of the curve
    metric_points = metric(curve[:-1]) # (N -1) 
    #getting the differences between consecutive points of the curve (velocity vectors).
    c1 = curve[1:] # ((N -1),D), 1,2,3,...,N
    c2 = curve[:-1] # ((N -1),D),  0,1,2,...,N-1
    delta = c1 - c2 #  Δzᵢ, shape ((N -1),D)
    #calculating tmp[i] = G(zᵢ) Δzᵢ efficiently.
    tmp = torch.bmm(metric_points, delta.unsqueeze(-1)).squeeze(-1) # (N -1) xD
    #calculating this step: Σᵢ Δzᵢ · tmp[i] = Σᵢ Δzᵢᵀ G(zᵢ) Δzᵢ
    energy = torch.sum(delta * tmp)
    #returning output
    return energy

def connecting_geodesic_LBFGS(metric, curve, lr=1e-3, max_iter=1000):
    #setting the optimizer to optimize the parameters of the curve
    opt = optim.LBFGS([curve.params], lr=lr)
    #setting the closure function
    def closure():
        opt.zero_grad()#clearing the gradients of the curve parameters before computing the energy and its gradients
        energy = curve_energy(metric, curve.points())#getting the energy
        energy.backward()#computing the gradients of the energy with respect to the curve parameters using backpropagation
        return energy
    #opti loop
    for iter in range(max_iter):
        opt.step(closure)#opti step

def connecting_geodesic_adam(metric, curve, lr=1e-3, max_iter=1000):
    #setting the optimizer to optimize the parameters of the curve
    opt = optim.Adam([curve.params], lr=lr)
    #opti loop
    for iter in range(max_iter):
        opt.zero_grad()#clearing the gradients of the curve parameters before computing the energy and its gradients
        energy = curve_energy(metric, curve.points())#getting the energy
        energy.backward()#computing the gradients of the energy with respect to the curve parameters using backpropagation
        opt.step()#opti step

def connecting_geodesic(opt_cho,metric, curve, lr=1e-3, max_iter=1000):       
    if(opt_cho == 'LBFGS'):
        connecting_geodesic_LBFGS(metric, curve, lr, max_iter)
    elif(opt_cho == 'Adam'):
        connecting_geodesic_adam(metric, curve, lr, max_iter)

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="train",
        choices=["train", "sample", "eval", "geodesics"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=50,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, stride=2, padding=1),
            nn.Flatten(),
            nn.Linear(512, 2 * M),
        )
        return encoder_net

    def new_decoder():
        decoder_net = nn.Sequential(
            nn.Linear(M, 512),
            nn.Unflatten(-1, (32, 4, 4)),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            args.epochs_per_decoder,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":
        #getting information from user input (or default values)
        num_curves =args.num_curves
        num_points_curve = args.num_t
        #loading and preparing model
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()
        #loading the data
        all_z = []
        all_labels = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                #data
                x = x.to(device)
                #latent representation
                z = model.encoder(x).mean.detach().to(device)  # encoding the test data
                #adding results of current iteration
                all_z.append(z)
                all_labels.append(y)
        #converting from torch tensors to numpy arrays for easier handling in plotting
        all_z = torch.cat(all_z, dim=0).numpy()      # shape: (num_samples, 2)
        all_labels = torch.cat(all_labels, dim=0).numpy()  # shape: (num_samples,)
        #getting the range of latent variables for plotting the metric
        z1_min = all_z[:, 0].min().item()
        z1_max = all_z[:, 0].max().item()
        z2_min = all_z[:, 1].min().item()
        z2_max = all_z[:, 1].max().item()
        #start of the plot
        plt.figure(figsize=(8, 8))
        #plotting the latent representations of the test data, colored by their labels
        scatter = plt.scatter(all_z[:, 0], all_z[:, 1], c=all_labels, cmap="Grays", s=5)
        plt.xlabel('z1')
        plt.ylabel('z2')
        #plotting the pullback metric as a heatmap in the latent space, where the color intensity represents the trace of the metric tensor, which gives an indication of how much the decoder stretches or compresses the latent space at different points
        metric = lambda z: pullback_metric(z, model.decoder)
        plot_metric(metric, torch.linspace(z1_min, z1_max, 100),torch.linspace(z2_min, z2_max, 100))
        #geodesic curves
        colors = plt.cm.tab10(np.linspace(0, 1, num_curves))
        for i in range (num_curves):
            #current curve color
            col_cur = colors[i]
            #getting random indices
            idx1 = torch.randint(0, all_z.shape[0], (1,))
            idx2 = torch.randint(0, all_z.shape[0], (1,))
            #getting data points based on the random indices
            z1 = torch.from_numpy(all_z[idx1]).float().squeeze(0)
            z2 = torch.from_numpy(all_z[idx2]).float().squeeze(0)
            # plot the selected endpoints
            plt.scatter([z1[0], z2[0]], [z1[1], z2[1]], color=col_cur, s=30)
            #creating a curve object to represent the geodesic
            c = PLcurve (z1 , z2 , num_points_curve)
            #geodesics before optimization
            label = 'curves before optimization' if i == 0 else None
            c.plot("curves before optimization",col_cur, label=label)
            print('Energy before optimization is {} '.format(curve_energy(metric,c.points()).item()))
            #optimizing the geodesic
            connecting_geodesic ('Adam',metric,c)
            #geodesics after optimization
            label = 'curves after optimization' if i == 0 else None
            print ('Energy after optimization is {} '.format(curve_energy(metric,c.points()).item()))
            c.plot("curves after optimization",col_cur,label=label)
            #adding some desriptions (finishing touches for the plot)
            plt.legend()
            plt.title('Geodesics in the latent space')
        #saving the plot
        experiments_folder = args.experiment_folder
        plt.savefig(f"{experiments_folder}/pullback_metric_trace.png", dpi=100, bbox_inches='tight')