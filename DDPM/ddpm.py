# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-02-11)

import torch
import torch.nn as nn
from tqdm import tqdm


class DDPM(nn.Module):
    def __init__(self, network, beta_1=1e-4, beta_T=2e-2, T=100):
        """
        Initialize a DDPM model.

        Parameters:
        network: [nn.Module]
            The network to use for the diffusion process.
        beta_1: [float]
            The noise at the first step of the diffusion process.
        beta_T: [float]
            The noise at the last step of the diffusion process.
        T: [int]
            The number of steps in the diffusion process.
        """
        super(DDPM, self).__init__()
        self.network = network
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.T = T

        self.beta = nn.Parameter(torch.linspace(beta_1, beta_T, T), requires_grad=False)
        self.alpha = nn.Parameter(1 - self.beta, requires_grad=False)
        self.alpha_cumprod = nn.Parameter(self.alpha.cumprod(dim=0), requires_grad=False)
    
    def negative_elbo(self, x):
        """
        Evaluate the DDPM negative ELBO on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The negative ELBO of the batch of dimension `(batch_size,)`.
        """

        batch_size = x.shape[0]
        device = x.device

        # Sample a time step uniformly for each data point.
        t = torch.randint(1, self.T + 1, (batch_size,), device=device)

        # Sample Gaussian noise
        epsilon = torch.randn_like(x)
        alpha_bar_t = self.alpha_cumprod[t - 1]
        alpha_bar_t = alpha_bar_t.reshape(batch_size, *([1] * (x.dim() - 1)))
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1.0 - alpha_bar_t) * epsilon

        # Predict noise using normalized time input.
        t_input = (t.float() / self.T).unsqueeze(-1)
        epsilon_theta = self.network(x_t, t_input)

        # Per-sample objective from Algorithm 1.
        neg_elbo = (epsilon_theta - epsilon).square()
        neg_elbo = neg_elbo.flatten(start_dim=1).sum(dim=1)

        return neg_elbo

    def sample(self, shape):
        """
        Sample from the model.

        Parameters:
        shape: [tuple]
            The shape of the samples to generate.
        Returns:
        [torch.Tensor]
            The generated samples.
        """
        # Sample x_t for t=T (i.e., Gaussian noise)
        x_t = torch.randn(shape).to(self.alpha.device)
        batch_size = shape[0]

        # Sample x_{t-1} given x_t until x_0 is sampled.
        for t in range(self.T-1, -1, -1):
            alpha_t = self.alpha[t]
            beta_t = self.beta[t]
            alpha_bar_t = self.alpha_cumprod[t]

            t_input = torch.full(
                (batch_size, 1),
                (t + 1) / self.T,
                device=x_t.device,
                dtype=x_t.dtype,
            )
            epsilon_theta = self.network(x_t, t_input)

            z = torch.randn_like(x_t) if t > 0 else torch.zeros_like(x_t)

            x_t = (
                (x_t - ((1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t)) * epsilon_theta)
                / torch.sqrt(alpha_t)
                + torch.sqrt(beta_t) * z
            )

        return x_t

    def loss(self, x):
        """
        Evaluate the DDPM loss on a batch of data.

        Parameters:
        x: [torch.Tensor]
            A batch of data (x) of dimension `(batch_size, *)`.
        Returns:
        [torch.Tensor]
            The loss for the batch.
        """
        return self.negative_elbo(x).mean()


def train(model, optimizer, data_loader, epochs, device):
    """
    Train a Flow model.

    Parameters:
    model: [Flow]
       The model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            if isinstance(x, (list, tuple)):
                x = x[0]
            x = x.to(device)
            optimizer.zero_grad()
            loss = model.loss(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


if __name__ == "__main__":
    import math
    import argparse
    import torch.utils.data
    from torchvision import datasets, transforms
    from torchvision.utils import save_image
    from unet import Unet

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'sample'], help='what to do when running the script')
    parser.add_argument('--model', type=str, default='mnist_ddpm.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='mnist_samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='V', help='learning rate for training (default: %(default)s)')
    parser.add_argument('--T', type=int, default=1000, help='number of diffusion steps (default: %(default)s)')
    parser.add_argument('--num-samples', type=int, default=64, help='number of MNIST samples to generate (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x + torch.rand(x.shape) / 255.0),
        transforms.Lambda(lambda x: (x - 0.5) * 2.0),
        transforms.Lambda(lambda x: x.flatten()),
    ])

    train_data = datasets.MNIST(
        'data/',
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
    )

    D = 28 * 28
    network = Unet()
    model = DDPM(network, T=args.T).to(args.device)

    if args.mode == 'train':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        train(model, optimizer, train_loader, args.epochs, args.device)
        torch.save(model.state_dict(), args.model)
    else:
        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))
        model.eval()
        with torch.no_grad():
            samples = model.sample((args.num_samples, D)).cpu()

        samples = samples / 2 + 0.5
        samples = samples.clamp(0.0, 1.0)
        nrow = max(1, int(math.sqrt(args.num_samples)))
        save_image(samples.view(-1, 1, 28, 28), args.samples, nrow=nrow)
