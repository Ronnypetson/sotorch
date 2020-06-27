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

    def _minimize(self, x0, **kwargs):
        args = kwargs['args']
        if 'method' in kwargs:
            method = kwargs['method']
        else:
            method = None
        if 'jac' in kwargs and kwargs['jac'] == None:
            jac = None
        elif method in ['CG', 'BFGS', 'Newton-CG', 'L-BFGS-B',
                        'TNC', 'SLSQP', 'dogleg', 'trust-ncg',
                        'trust-krylov', 'trust-exact', 'trust-constr']:
            jac = self._jac_npy
        else:
            jac = None
        if 'hess' in kwargs and kwargs['hess'] == None:
            hess = None
        elif method in ['Newton-CG', 'dogleg', 'trust-ncg',
                        'trust-krylov', 'trust-exact', 'trust-constr']:
            hess = self._hess_npy
        else:
            hess = None
        if 'hessp' in kwargs:
            raise NotImplementedError('There is no support for \'hessp\' currently.')
        if 'bounds' in kwargs:
            bounds = kwargs['bounds']
        else:
            bounds = None
        if 'options' in kwargs:
            options = kwargs['options']
        else:
            options = None
        if 'constraints' in kwargs:
            constraints = kwargs['constraints']
        else:
            constraints = ()
        if 'tol' in kwargs:
            tol = kwargs['tol']
        else:
            tol = None
        if 'callback' in kwargs:
            callback = kwargs['callback']
        else:
            callback = None

        batchwise = kwargs['batchwise']

        x0 = x0.detach().numpy()
        x0_shape = x0.shape

        suc = []
        msg = []
        if batchwise:
            all_res = []
            b = x0.shape[0]

            if method == 'trust-constr':
                x0 = x0.reshape(b,-1)

            if bounds is None:
                bounds = [None] * b
            if args == () or args == [] or args is None:
                args = [None] * b
            if constraints == ():
                constraints = [()] * b
            if tol is None:
                tol = [None] * b

            for i, x0_ in enumerate(x0):
                res = minimize(self._obj_npy,
                               x0_, args=args[i],
                               method=method,
                               jac=jac,
                               hess=hess,
                               bounds=bounds[i],
                               options=options,
                               constraints=constraints[i],
                               tol=tol[i],
                               callback=callback)
                all_res.append(res.x)
                suc.append(res.success)
                msg.append(res.message)
            res = np.array(all_res)
        else:
            if method == 'trust-constr':
                x0 = x0.reshape(-1)

            res = minimize(self._obj_npy,
                           x0, args=args,
                           method=method,
                           jac=jac,
                           hess=hess,
                           bounds=bounds,
                           options=options,
                           constraints=constraints,
                           tol=tol,
                           callback=callback)
            suc.append(res.success)
            msg.append(res.message)
            res = res.x

        ans = res.reshape(x0_shape)
        ans = torch.from_numpy(ans)
        return ans, suc, msg

if __name__ == '__main__':
    def f(x, *y):
        return torch.norm(x) + sum(y)

    #dtype = torch.float

    opt = Minimizer(f)
    x0 = torch.randn(2, 3)
    options = {'disp': False}
    bwise = True
    args = (1, 2, 3)
    bounds = None
    constraints = ()
    all_methods = ['Newton-CG', 'dogleg', 'trust-ncg',
                    'trust-krylov', 'trust-exact', 'trust-constr']
    all_methods += ['CG', 'BFGS', 'L-BFGS-B',
                    'TNC', 'SLSQP']
    all_methods += ['Nelder-Mead', 'Powell', 'COBYLA']

    if bwise:
        args = [args] * x0.size(0)
        bounds = [bounds] * x0.size(0)

    with torch.autograd.set_detect_anomaly(False):
        for method in all_methods:
            x, _, _ = opt._minimize(x0, args=args,
                             method=method,
                             bounds=bounds,
                             options=options,
                             constraints=constraints,
                             batchwise=bwise)
            print(f'{method}: OK.')
