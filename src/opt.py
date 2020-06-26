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
                 options=None,
                 batchwise=False):
        x0 = x0.detach().numpy()

        if batchwise:
            all_res = []
            b = x0.shape[0]
            if bounds is None:
                bounds = [None] * b
            if args == () or args == [] or args is None:
                args = [None] * b

            for i, x0_ in enumerate(x0):
                res = minimize(self._obj_npy,
                               x0_, args=args[i],
                               method='Newton-CG',
                               jac=self._jac_npy,
                               hess=self._hess_npy,
                               bounds=bounds[i],
                               options=options)
                all_res.append(res.x)
            res = np.array(all_res)
        else:
            res = minimize(self._obj_npy,
                           x0, args=args,
                           method='Newton-CG',
                           jac=self._jac_npy,
                           hess=self._hess_npy,
                           bounds=bounds,
                           options=options)
            res = res.x

        ans = res.reshape(x0.shape)
        ans = torch.from_numpy(ans)
        return ans


if __name__ == '__main__':
    def f(x, *y):
        return torch.norm(x) + sum(y)

    #dtype = torch.float

    opt = Minimizer(f)
    x0 = torch.randn(10, 10)
    options = {'disp': True}
    bwise = True
    args = (1, 2, 3)
    bounds = None

    if bwise:
        args = [args] * x0.size(0)
        bounds = [bounds] * x0.size(0)

    with torch.autograd.set_detect_anomaly(False):
        x = opt.minimize(x0, args,
                         bounds=bounds,
                         options=options,
                         batchwise=bwise)
        print(x)
