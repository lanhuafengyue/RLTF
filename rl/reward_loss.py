# noisy MLE
import torch

def gaussian_likelihood(y_pred, y_true, sigma):
    dist = torch.distributions.Normal(y_pred, sigma)
    log_likelihood = dist.log_prob(y_true)
    return log_likelihood.mean()

def MLE(y_pred, y_true, sigma):
    return -gaussian_likelihood(y_pred, y_true, sigma)

