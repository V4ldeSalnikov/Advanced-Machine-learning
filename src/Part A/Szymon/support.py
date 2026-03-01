#importing libraries
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde
from matplotlib.patches import Patch


#functions
def evaluate_test_elbo(model, test_loader, device):
    #setting model to eval mode
    model.eval()
    #setting variables to update
    total_elbo = 0.0 #variable that will hold total elbo over the test set (partial elbo added each iteration(batch))
    total_n = 0 #variable holding total number of samples
    #going through test set
    with torch.no_grad():
        for x, _ in test_loader:
            #preparing batch
            x = x.to(device)
            batch_size = x.size(0)
            #getting mean ELBO for the batch
            batch_elbo = model.elbo(x)
            #updating variables (the calculations are done in a way to have each batch equal weight as last one might have different size, i.e. final value needs to be calculated as mean over all samples, not mean over batches to let eahc batch have equal weight)
            total_elbo += batch_elbo.item() * batch_size
            total_n += batch_size
    #calulating mean elbo over the whole test set
    mean_elbo = total_elbo / total_n                  
    #returning results
    return mean_elbo

def plot_posterior_samples(model, data_loader, device, n_samples_per_image=1, save_path='posterior_samples.png'):
    """
    Plot aggregate posterior density (KDE over encoder samples) vs prior density.
    For latent dimensions > 2, PCA projects everything onto the first two components.
    """

    model.eval()

    all_samples = []

    with torch.no_grad():
        for batch in data_loader:
            x, _ = batch
            x = x.to(device)
            q = model.encoder(x)
            for _ in range(n_samples_per_image):
                z = q.rsample()
                all_samples.append(z.cpu().numpy())

    all_samples = np.vstack(all_samples)
    M = all_samples.shape[1]

    # PCA if M>2
    if M > 2:
        pca = PCA(n_components=2)
        samples_2d = pca.fit_transform(all_samples)
        explained_var = pca.explained_variance_ratio_
        xlabel = f'PC1 ({explained_var[0]:.1%})'
        ylabel = f'PC2 ({explained_var[1]:.1%})'
        title = f'Aggregate Posterior & Prior Density (PCA projected from {M}D)'
    else:
        pca = None
        samples_2d = all_samples
        xlabel = 'Latent Dimension 0'
        ylabel = 'Latent Dimension 1'
        title = 'Aggregate Posterior & Prior Density'

    # Build 2D grid covering the posterior scatter with a small margin
    margin = 0.5
    x_min, x_max = samples_2d[:, 0].min() - margin, samples_2d[:, 0].max() + margin
    y_min, y_max = samples_2d[:, 1].min() - margin, samples_2d[:, 1].max() + margin
    grid_res = 150
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                          np.linspace(y_min, y_max, grid_res))
    grid_2d = np.column_stack([gx.ravel(), gy.ravel()])

    # Prior density: inverse-project grid to M-dim, evaluate prior
    grid_md = pca.inverse_transform(grid_2d) if pca is not None else grid_2d
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_md, dtype=torch.float32).to(device)
        log_prob = model.prior().log_prob(grid_tensor).cpu().numpy()
    prior_density = np.exp(log_prob).reshape(grid_res, grid_res)

    # Aggregate posterior density via KDE over the 2D projected samples
    kde = gaussian_kde(samples_2d.T)
    agg_posterior = kde(grid_2d.T).reshape(grid_res, grid_res)

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.contourf(gx, gy, agg_posterior, levels=15, cmap='Blues', alpha=0.6)
    ax.contour(gx, gy, agg_posterior, levels=15, colors='blue', linewidths=0.8, alpha=0.7)
    ax.contourf(gx, gy, prior_density, levels=15, cmap='Reds', alpha=0.3)
    ax.contour(gx, gy, prior_density, levels=15, colors='red', linewidths=0.8, alpha=0.7)

    legend_elements = [Patch(facecolor='blue', alpha=0.6, label='Aggregate Posterior q(z)'),
                       Patch(facecolor='red', alpha=0.3, label='Prior p(z)')]
    ax.legend(handles=legend_elements)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()