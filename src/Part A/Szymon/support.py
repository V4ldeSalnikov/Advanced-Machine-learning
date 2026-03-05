#importing libraries
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import gaussian_kde, multivariate_normal
from matplotlib.patches import Patch
from priors import GaussianPrior, MoGPrior
from flow import Flow
import torch.nn.functional as F

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


def plot_posterior_samples(model, data_loader, device, save_path='posterior_prior.png', n_sub=None):

    model.eval()
    mus_list, post_z_list = [], []
    n_collected = 0
    with torch.no_grad():
        for x, _ in data_loader:
            if n_sub is not None and n_collected >= n_sub:
                break
            q = model.encoder(x.to(device))
            mus_list.append(q.base_dist.loc.cpu().numpy())
            post_z_list.append(q.rsample().cpu().numpy())
            n_collected += x.size(0)
    mus    = np.vstack(mus_list)     # (N, M)  encoder means
    post_z = np.vstack(post_z_list)  # (N, M)  posterior samples
    if n_sub is not None:
        mus, post_z = mus[:n_sub], post_z[:n_sub]
    N, M = mus.shape

    if M > 2:
        pca = PCA(n_components=2)
        pca.fit(mus)
        P  = pca.components_  # (2, M) — orthonormal rows
        ev = pca.explained_variance_ratio_
        xlabel       = f'PC1  ({ev[0]:.1%} of encoder-mean variance)'
        ylabel       = f'PC2  ({ev[1]:.1%} of encoder-mean variance)'
        title_suffix = f'  [PCA projected from {M}D, fitted on encoder means]'
    else:
        P = np.eye(2)
        xlabel, ylabel = 'z₁', 'z₂'
        title_suffix   = ''

    mus_2d    = mus    @ P.T  # (N, 2)  projected means
    post_z_2d = post_z @ P.T  # (N, 2)  projected posterior samples

    prior_module = model.prior  
    prior_ref_2d = None         

    if isinstance(prior_module, Flow):
        n_flow = min(N * 3, 10000)  
        with torch.no_grad():
            flow_samples = prior_module.sample(torch.Size([n_flow])).cpu().numpy()
        prior_ref_2d = flow_samples @ P.T  # (n_flow, 2)

    elif isinstance(prior_module, MoGPrior):
        comp_means_np = prior_module.means.detach().cpu().numpy() 
        prior_ref_2d  = comp_means_np @ P.T                        

    all_ref = np.vstack([post_z_2d, prior_ref_2d]) if prior_ref_2d is not None else post_z_2d
    margin  = 1.5
    x_min, x_max = all_ref[:, 0].min() - margin, all_ref[:, 0].max() + margin
    y_min, y_max = all_ref[:, 1].min() - margin, all_ref[:, 1].max() + margin
    grid_res = 100  
    gx, gy   = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                            np.linspace(y_min, y_max, grid_res))
    grid_2d  = np.column_stack([gx.ravel(), gy.ravel()])  # (G, 2)
    G = len(grid_2d)

    kde_post      = gaussian_kde(post_z_2d.T)
    agg_posterior = kde_post(grid_2d.T)

    if isinstance(prior_module, GaussianPrior):
        prior_density = multivariate_normal.pdf(grid_2d,
                                                 mean=np.zeros(2),
                                                 cov=np.eye(2))
        prior_label = 'Prior p(z) = N(0,I)  [exact]'

    elif isinstance(prior_module, MoGPrior):
        with torch.no_grad():
            w          = torch.softmax(prior_module.logits, dim=0).cpu().numpy()     # (K,)
            comp_means = prior_module.means.cpu().numpy()                             # (K, M)
            comp_stds  = (F.softplus(prior_module.log_stds) + 1e-5).cpu().numpy()   # (K, M)
        K = len(w)
        comp_means_2d = comp_means @ P.T                           # (K, 2)
        comp_stds2    = comp_stds ** 2                             # (K, M)
        P_comp        = P[None, :, :] * comp_stds2[:, None, :]    # (K, 2, M)
        comp_covs_2d  = P_comp @ P.T + 1e-6 * np.eye(2)           # (K, 2, 2)

        prior_density = np.zeros(G)
        for k in range(K):
            prior_density += w[k] * multivariate_normal.pdf(grid_2d,
                                                              mean=comp_means_2d[k],
                                                              cov=comp_covs_2d[k])
        prior_label = f'Prior p(z) = MoG ({K} components)  [exact projected mixture]'

    else:  # Flow
        kde = gaussian_kde(prior_ref_2d.T)
        prior_density = kde(grid_2d.T)
        prior_label = 'Prior p(z) = Flow  [KDE on samples — no exact marginal]'

    agg_posterior = agg_posterior.reshape(grid_res, grid_res)
    prior_density  = prior_density.reshape(grid_res, grid_res)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.contourf(gx, gy, agg_posterior, levels=15, cmap='Blues', alpha=0.6)
    ax.contour( gx, gy, agg_posterior, levels=15, colors='blue', linewidths=0.8, alpha=0.7)
    ax.contourf(gx, gy, prior_density,  levels=15, cmap='Reds',  alpha=0.3)
    ax.contour( gx, gy, prior_density,  levels=15, colors='red',  linewidths=0.8, alpha=0.7)

    legend_elements = [
        Patch(facecolor='blue', alpha=0.6, label='Aggregate Posterior q(z)'),
        Patch(facecolor='red',  alpha=0.3, label=prior_label),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Aggregate Posterior & Prior{title_suffix}', fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.close()