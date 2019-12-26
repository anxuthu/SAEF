__all__ = ['Sign', 'TopK']

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
        topk (int): the K in top-k
    """

    topk = kwargs['topk']
    for p in params:
        k = max(1, int(p.numel() * topk))
        vals, indices = p.view(-1).abs().topk(k)
        p_sign = p.sign()
        p.zero_().view(-1).scatter_(0, indices, vals).mul_(p_sign.view(-1))
