__all__ = ['Sign', 'TopK', 'USpar']

def Sign(params, **kwargs):
    r"""Scaled sign compression.

    Args:
        params (iterable): iterable of torch.tensor
    """

    l1 = 0
    n_params = 0
    for p in params:
        l1 += p.abs().sum()
        n_params += p.numel()

    for p in params:
        p.sign_().mul_(l1 / n_params)


def TopK(params, **kwargs):
    r"""Top-k sparsification.

    Args:
        params (iterable): iterable of torch.tensor
        spar (float): the top ?% (sparsity)
    """

    spar = kwargs['spar']
    for p in params:
        k = max(1, int(p.numel() * spar))
        vals, indices = p.view(-1).abs().topk(k)
        p_sign = p.sign()
        p.zero_().view(-1).scatter_(0, indices, vals).mul_(p_sign.view(-1))


def USpar(params, **kwargs):
    r"""Unbiased gradient sparsification.

    Args:
        params (iterable): iterable of torch.tensor
        spar (float): the probability to send a gradient element (sparsity)
    """

    spar = kwargs['spar']
    for p in params:
        mask = p.clone().uniform_(0,1).le_(spar)
        p.mul_(mask).mul_(1.0/spar)
