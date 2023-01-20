import torch


def wasserstein1d(x, y):
    x1, _ = torch.sort(x, dim=0)
    y1, _ = torch.sort(y, dim=0)
    z = (x1-y1).view(-1)
    n, l = x.size()
    return torch.dot(z, z) / (n*l)


def clt_sw(x, y):
    n, dim = x.shape
    
    meanx = torch.mean(x, dim=0)
    xc = x - meanx
    gamma_xc = torch.mean(torch.linalg.norm(xc, dim=1) ** 2) / dim

    meany = torch.mean(y, dim=0)
    yc = y - meany
    gamma_yc = torch.mean(torch.linalg.norm(yc, dim=1) ** 2) / dim

    mean_term = 1 / dim * torch.linalg.norm(meanx - meany) ** 2
    sw2 = mean_term + (gamma_xc ** (1/2) - gamma_yc ** (1/2)) ** 2
    return sw2


def cov(x, rowvar=False, bias=False, ddof=None, aweights=None):
    """Estimates covariance matrix like numpy.cov"""
    # ensure at least 2D
    if x.dim() == 1:
        x = x.view(-1, 1)

    # treat each column as a data point, each row as a variable
    if rowvar and x.shape[0] != 1:
        x = x.t()

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    w = aweights
    if w is not None:
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=torch.float)
        w_sum = torch.sum(w)
        avg = torch.sum(x * (w/w_sum)[:,None], 0)
    else:
        avg = torch.mean(x, 0)

    # Determine the normalization
    if w is None:
        fact = x.shape[0] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * torch.sum(w * w) / w_sum

    xm = x.sub(avg.expand_as(x))

    if w is None:
        X_T = xm.t()
    else:
        X_T = torch.mm(torch.diag(w), xm).t()

    c = torch.mm(X_T, xm)
    c = c / fact

    return c.squeeze()
