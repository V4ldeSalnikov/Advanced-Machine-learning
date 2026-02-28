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