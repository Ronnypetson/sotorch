import torch
from src.opt import Minimizer

if __name__ == '__main__':
    def f(x, *y):
        return torch.norm(x) + sum(y)

    opt = Minimizer(f)
    x0 = torch.randn(2, 3, 4)
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
            x, _, _ = opt.minimize(x0, args=args,
                             method=method,
                             bounds=bounds,
                             options=options,
                             constraints=constraints,
                             batchwise=bwise)
            print(f'{method}: {x} {opt.min_obj}')
            print('OK.')
            print()
