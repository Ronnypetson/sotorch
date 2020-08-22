# sotorch
Second-order optimization methods compatible with PyTorch. There is no need to implement the Jacobians or Hessians if the objective function is end-to-end differentiable.

It is a wrapper of part of scipy.optimize.minimize and based on PyTorch's autograd.

# Install
```python

pip install sotorch-1.0-py3-none-any.whl

```
## Dependencies
PyTorch >= 1.3.1

SciPy >= 1.3.1

NumPy >= 1.17.2

(probably works with older versions too, but not guaranteed).

## Usage
```python
import torch
from sotorch.opt import Minimizer

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
```

## References
https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html (contains references to the original papers where each method was publicized).

https://pytorch.org/docs/stable/autograd.html#functional-higher-level-api

https://gist.github.com/apaszke/226abdf867c4e9d6698bd198f3b45fb7
