import numpy as np

def negative_log_likelihood(y_pred, y, T=1):
    """
    Compute negative log likelihood, i.e. - sum of log prob of correct labels, from logits and labels
    y_pred: (n_samples, n_labels)
    y: (n_samples,)
    T: positive float, temperature (higher T makes higher likelihood)
    """
    y_pred = y_pred - np.max(y_pred, axis=1, keepdims=True) # normalize for numerical stability
    exp = np.exp(y_pred)
    probs = exp / np.sum(exp, axis=1, keepdims=True)
    likelihood = probs[np.arange(len(y)), y]
    return -np.sum(np.log(likelihood)/T)


