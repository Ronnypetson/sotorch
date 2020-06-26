import numpy as np
from scipy.optimize import minimize
import torch
from grad import jacobian, hessian


class Minimizer:
    def __init__(self, objective,
                 device='cpu',
                 dtype=torch.double):
        '''
        :param objective: a callable that receives a tensor of parameters and returns a scalar tensor.
                        It should be end-to-end differentiable (e.g. composed of differentiable
                        PyTorch functions).
        '''
        self._obj_tc = objective
        self.device = device
        self.dtype = dtype

    def _obj_npy(self, x, *args):
        '''
        Auxiliary objective function compatible with NumPy.
        :param x: a tensor.
        :return: the objective value at x to be minimized
        '''
        x = torch.from_numpy(x)
        x = x.requires_grad_(True)
        y = self._obj_tc(x, *args)
        y = y.detach().numpy()
        return y
    
    def _jac_npy(self, x, *args):
        x = torch.from_numpy(x)
        x = x.requires_grad_(True)
        jac = jacobian(self._obj_tc(x, *args), x)
        jac = jac.detach().numpy()
        return jac
    
    def _hess_npy(self, x, *args):
        x = torch.from_numpy(x)
        x = x.requires_grad_(True)
        hess = hessian(self._obj_tc(x, *args), x)
        hess = hess.detach().numpy()
        return hess

    def minimize(self, x0,
                 args=(),
                 bounds=None,
                 options=None):
        x0 = x0.detach().numpy()

        #res = minimize(self.objective,\
        #               Tfoe0,method='BFGS',\
        #               jac=True,\
        #               options={'disp': True,\
        #                        'maxiter':100,\
        #                        'gtol':1e-8})

        res = minimize(self._obj_npy,
                       x0, args=args,
                       method='Newton-CG',
                       jac=self._jac_npy,
                       hess=self._hess_npy,
                       bounds=bounds,
                       options=options)
        
        #res = minimize(self.objective,\
        #               Tfoe0,method='Newton-CG',\
        #               jac=True,\
        #               options={'disp': False,\
        #                        'maxiter':100,\
        #                        'gtol':1e-8})
        
        #res = minimize(self.obj_npy,\
        #               Tfoe0,method='Newton-CG',\
        #               jac=self.jac_npy,\
        #               hess=self.hess_npy,\
        #               options={'disp': False,\
        #                        'maxiter':100,\
        #                        'gtol':1e-8})
        
        return res.x


if __name__ == '__main__':
    def f(x, *y):
        return torch.norm(x) + sum(y)

    #dtype = torch.float

    opt = Minimizer(f)
    x0 = torch.ones(10)
    args = (1, 2, 3)
    options = {'disp': True}

    with torch.autograd.set_detect_anomaly(False):
        x = opt.minimize(x0, args, options=options)
        print(x)
