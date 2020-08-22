# Based on https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7

import torch


def jacobian(y, x, create_graph=False):
    '''
    Computes the Jacobian of tensor y w.r.t tensor x.

    :param y: Target tensor. Must be the result of differentiable operations from x.
    :param x: Input tensor. Must require gradients.
    :param create_graph: Useful for the computation of Hessian.
    :return: the Jacobian of y w.r.t x.
    '''
    jac = []
    flat_y = y.reshape(-1)
    grad_y = torch.zeros_like(flat_y)
    for i in range(len(flat_y)):
        grad_y[i] = 1.
        grad_x, = torch.autograd.grad(flat_y, x, grad_y,
                                      retain_graph=True,
                                      create_graph=create_graph)
        jac.append(grad_x.reshape(x.shape))
        grad_y[i] = 0.
    return torch.stack(jac).reshape(y.shape + x.shape)


def hessian(y, x):
    '''
    Computes the Hessian of tensor y w.r.t tensor x.

    :param y: Target tensor. Must be the result of differentiable operations from x.
    :param x: Input tensor. Must require gradients.
    :return: the Hessian of y w.r.t x.
    '''
    return jacobian(jacobian(y, x, create_graph=True), x)


def f(x):
    return torch.norm(x)


if __name__ == '__main__':
    a = torch.ones(4, requires_grad=True)
    print(a)
    print(f(a))
    print(jacobian(f(a), a))
    print(hessian(f(a), a))

