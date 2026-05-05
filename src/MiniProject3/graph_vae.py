import torch
import torch.nn as nn
import torch.distributions as td
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch
import networkx as nx
import matplotlib.pyplot as plt


def resolve_device(requested_device):
    xpu_backend = getattr(torch, 'xpu', None)
    xpu_available = xpu_backend is not None and torch.xpu.is_available()
    mps_backend = getattr(torch.backends, 'mps', None)
    mps_available = mps_backend is not None and mps_backend.is_available()

    if requested_device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if xpu_available:
            return torch.device('xpu')
        if mps_available:
            return torch.device('mps')
        return torch.device('cpu')

    if requested_device == 'cuda':
        if not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is not available.')
        return torch.device('cuda')

    if requested_device == 'xpu':
        if not xpu_available:
            raise ValueError('XPU was requested but is not available.')
        return torch.device('xpu')

    if requested_device == 'mps':
        if not mps_available:
            raise ValueError('MPS was requested but is not available.')
        return torch.device('mps')

    if requested_device == 'cpu':
        return torch.device('cpu')

    raise ValueError(f'Unsupported device: {requested_device}')


dataset = TUDataset(root='./data/', name='MUTAG')
node_feature_dim = 7

# decoder out size will depend on the largest graph in the dataset
N_max = max(data.num_nodes for data in dataset) 

rng = torch.Generator().manual_seed(0)
train_dataset, val_dataset, test_dataset = random_split(
    dataset, (100, 44, 44), generator=rng
)

# node count distribution from training set
train_node_counts = [dataset[int(i)].num_nodes for i in train_dataset.indices]

# sample number of nodes for generated graphs according to the training distribution
def sample_num_nodes(n_samples=1):
    counts = torch.tensor(train_node_counts, dtype=torch.float)
    idx = torch.multinomial(torch.ones(len(counts)), n_samples, replacement=True)
    return counts[idx].long().tolist()

# batch to dense adjacency and mask
def to_dense(data, N_max, device):

    A = to_dense_adj(data.edge_index, data.batch, max_num_nodes=N_max)
    _, node_mask = to_dense_batch(data.x, data.batch, max_num_nodes=N_max)

    # pair-level mask
    mask = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)

    diag = torch.eye(N_max, dtype=torch.bool, device=device).unsqueeze(0)
    mask = mask & ~diag

    return A.to(device), mask.to(device)


class GaussianPrior(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.mean = nn.Parameter(torch.zeros(latent_dim), requires_grad=False)
        self.std = nn.Parameter(torch.ones(latent_dim), requires_grad=False)

    def forward(self):
        return td.Independent(td.Normal(self.mean, self.std), 1)


class EncoderNet(nn.Module):
    def __init__(self, node_feature_dim, state_dim, num_message_passing_rounds, latent_dim):
        super().__init__()
        self.state_dim = state_dim
        self.num_message_passing_rounds = num_message_passing_rounds

        # node features into state space
        self.input_net = nn.Sequential(
            nn.Linear(node_feature_dim, state_dim),
            nn.ReLU()
        )

        # message net and update net per round
        self.message_net = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU())
            for _ in range(num_message_passing_rounds)
        ])
        self.update_net = nn.ModuleList([
            nn.Sequential(nn.Linear(state_dim, state_dim), nn.ReLU())
            for _ in range(num_message_passing_rounds)
        ])

        self.output_net = nn.Linear(state_dim, 2 * latent_dim)

    def forward(self, x, edge_index, batch):
        num_nodes = x.shape[0]
        num_graphs = batch.max() + 1

        # node states from features
        state = self.input_net(x)

        # Message passing rounds
        for r in range(self.num_message_passing_rounds):
            message = self.message_net[r](state)
            aggregated = x.new_zeros((num_nodes, self.state_dim))
            aggregated = aggregated.index_add(0, edge_index[1], message[edge_index[0]])
            state = state + self.update_net[r](aggregated)

        # Sum-pool all node states into one graph-level vector
        graph_state = x.new_zeros((num_graphs, self.state_dim))
        graph_state = graph_state.index_add(0, batch, state)

        return self.output_net(graph_state)


class GraphEncoder(nn.Module):
    def __init__(self, encoder_net):
        super().__init__()
        self.encoder_net = encoder_net

    def forward(self, x, edge_index, batch):
        mean, logstd = torch.chunk(self.encoder_net(x, edge_index, batch), 2, dim=-1)
        return td.Independent(td.Normal(mean, torch.exp(logstd)), 1)


class DecoderNet(nn.Module):
    def __init__(self, latent_dim, hidden_dim, N_max):
        super().__init__()
        self.N_max = N_max
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, N_max * N_max),
        )

    def forward(self, z):
        return self.net(z).view(-1, self.N_max, self.N_max)


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        super().__init__()
        self.decoder_net = decoder_net

    def forward(self, z):
        logits = self.decoder_net(z)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class GraphVAE(nn.Module):
    def __init__(self, prior, encoder, decoder, N_max):
        super().__init__()
        self.prior = prior
        self.encoder = encoder
        self.decoder = decoder
        self.N_max = N_max

    def elbo(self, data, A_dense, mask):

        q = self.encoder(data.x, data.edge_index, data.batch)
        z = q.rsample()

        logits = self.decoder.decoder_net(z)
        log_probs = td.Bernoulli(logits=logits).log_prob(A_dense)
        recon = (log_probs * mask.float()).sum() / mask.float().sum()
        kl = td.kl_divergence(q, self.prior()).mean()

        return recon - kl

    def forward(self, data, A_dense, mask):
        return -self.elbo(data, A_dense, mask)

    @torch.no_grad()
    def sample(self, N_sizes): # generate one graph per entry in N_sizes; return: list of (N, N) adjacency tensors

        n = len(N_sizes)
        z = self.prior().sample(torch.Size([n])).to(self.prior.mean.device)
        logits = self.decoder.decoder_net(z)          
        probs = torch.sigmoid(logits)

        graphs = []
        for i, N in enumerate(N_sizes):
            p = probs[i, :N, :N]                     

            upper = torch.bernoulli(torch.triu(p, diagonal=1))
            A = upper + upper.T 
            graphs.append(A)

        return graphs


def train(model, optimizer, train_loader, val_loader, epochs, N_max, device):
    train_elbos, val_elbos = [], []

    for epoch in range(epochs):
        
        model.train()
        epoch_elbo = 0.
        for data in train_loader:
            data = data.to(device)
            A_dense, mask = to_dense(data, N_max, device)

            optimizer.zero_grad()
            loss = model(data, A_dense, mask)
            loss.backward()
            optimizer.step()

            epoch_elbo += -loss.item() * data.num_graphs

        train_elbos.append(epoch_elbo / len(train_loader.dataset))

        
        model.eval()
        with torch.no_grad():
            val_elbo = 0.
            for data in val_loader:
                data = data.to(device)
                A_dense, mask = to_dense(data, N_max, device)
                val_elbo += model.elbo(data, A_dense, mask).item() * data.num_graphs
            val_elbos.append(val_elbo / len(val_loader.dataset))

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}: train ELBO = {train_elbos[-1]:.3f}, val ELBO = {val_elbos[-1]:.3f}')

    return train_elbos, val_elbos


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'xpu', 'mps'], help='compute device (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='latent space dimension (default: %(default)s)')
    parser.add_argument('--rounds', type=int, default=3, metavar='N', help='message passing rounds (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N', help='training epochs (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='training batch size (default: %(default)s)')
    parser.add_argument('--model', type=str, default='src/MiniProject3/smodelGNN.pt', help='file to save/load model (default: %(default)s)')
    args = parser.parse_args()
    device = resolve_device(args.device)

    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(f'  {key} = {value}')
    print(f'  resolved_device = {device}')

    torch.manual_seed(42)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=44, shuffle=False)

    state_dim = 64
    hidden_dim = 256

    prior = GaussianPrior(args.latent_dim)
    enc_net = EncoderNet(node_feature_dim, state_dim, args.rounds, args.latent_dim)
    dec_net = DecoderNet(args.latent_dim, hidden_dim, N_max)
    encoder = GraphEncoder(enc_net)
    decoder = BernoulliDecoder(dec_net)
    model = GraphVAE(prior, encoder, decoder, N_max).to(device)

    if args.mode == 'train':
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_elbos, val_elbos = train(
            model, optimizer, train_loader, val_loader, args.epochs, N_max, device
        )

        torch.save(model.state_dict(), args.model)
        
        warmup = min(20, len(train_elbos) // 5)
        plt.figure()
        plt.plot(train_elbos[warmup:], label='Train')
        plt.plot(val_elbos[warmup:],   label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('ELBO')
        plt.legend()
        plt.tight_layout()
        plt.savefig('src/MiniProject3/elbo_curve.png', dpi=150)

    # just for testing few samples after training
    elif args.mode == 'sample':
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()

        N_sizes = sample_num_nodes(n_samples=10)
        graphs = model.sample(N_sizes)
        print(f'\nGenerated {len(graphs)} graphs')
        for i, (G, N) in enumerate(zip(graphs, N_sizes)):
            edges = int(G.sum().item() // 2)
            print(f'  Graph {i}: {N} nodes, {edges} edges')

        n_cols = 5
        n_rows = (len(graphs) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
        axes = axes.flatten()
        for i, (G_adj, N) in enumerate(zip(graphs, N_sizes)):
            G_nx = nx.from_numpy_array(G_adj.cpu().numpy())
            nx.draw(G_nx, ax=axes[i], with_labels=False, node_color='lightblue', edge_color='gray', node_size=100)
            axes[i].set_title(f'{N} nodes')
        for ax in axes[len(graphs):]:
            ax.axis('off')
        plt.savefig('src/MiniProject3/sampled_graphs.png', dpi=150, bbox_inches='tight')
