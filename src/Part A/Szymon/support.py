#importing libraries
import torch

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
    Sample from the approximate posterior and plot colored by class label,
    overlaid with prior density contours. For latent dimensions > 2, PCA
    projects everything onto the first two components; the prior is evaluated
    by inverse-transforming grid points back to M-dim space.
    """

    model.eval()

    all_samples = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            x, y = batch
            x = x.to(device)
            q = model.encoder(x)

            # Sample from posterior for each image in batch
            for _ in range(n_samples_per_image):
                z = q.rsample()
                all_samples.append(z.cpu().numpy())
                all_labels.extend(y.numpy())

    all_samples = np.vstack(all_samples)
    all_labels = np.array(all_labels)

    M = all_samples.shape[1]

    # PCA if M>2
    if M > 2:
        pca = PCA(n_components=2)
        samples_2d = pca.fit_transform(all_samples)
        explained_var = pca.explained_variance_ratio_
        xlabel = f'PC1 ({explained_var[0]:.1%})'
        ylabel = f'PC2 ({explained_var[1]:.1%})'
        title = f'Posterior Samples & Prior Density (PCA projected from {M}D)'
    else:
        pca = None
        samples_2d = all_samples
        xlabel = 'Latent Dimension 0'
        ylabel = 'Latent Dimension 1'
        title = 'Posterior Samples & Prior Density'

    # Build 2D grid covering the posterior scatter with a small margin
    margin = 0.5
    x_min, x_max = samples_2d[:, 0].min() - margin, samples_2d[:, 0].max() + margin
    y_min, y_max = samples_2d[:, 1].min() - margin, samples_2d[:, 1].max() + margin
    grid_res = 150
    gx, gy = np.meshgrid(np.linspace(x_min, x_max, grid_res),
                          np.linspace(y_min, y_max, grid_res))
    grid_2d = np.column_stack([gx.ravel(), gy.ravel()])  # (grid_res^2, 2)

    # Map 2D grid points to M-dim space for prior evaluation
    # For M>2: inverse PCA reconstructs the closest point on the PCA subspace
    grid_md = pca.inverse_transform(grid_2d) if pca is not None else grid_2d

    # Evaluate prior log-density on the grid (works for both GaussianPrior and MoGPrior)
    with torch.no_grad():
        grid_tensor = torch.tensor(grid_md, dtype=torch.float32).to(device)
        log_prob = model.prior().log_prob(grid_tensor).cpu().numpy()
    density = np.exp(log_prob).reshape(grid_res, grid_res)

    plt.figure(figsize=(12, 10))
    colors = plt.cm.tab10(np.arange(10))

    plt.contourf(gx, gy, density, levels=15, cmap='Greys', alpha=0.25)
    plt.contour(gx, gy, density, levels=15, colors='grey', linewidths=0.8, alpha=0.7)

    # Posterior samples coloured by class
    for label in range(10):
        mask = all_labels == label
        plt.scatter(samples_2d[mask, 0], samples_2d[mask, 1],
                   c=[colors[label]], label=f'Class {label}', alpha=0.6, s=30)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Posterior samples plot saved to {save_path}")
    plt.close()