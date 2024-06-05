import torch
import torch.optim as optim

from utils import ternarize


class ADMMOptimizer(optim.Optimizer):
    def __init__(self, params, fixed_alpha=None, base_optimizer_cls=optim.SGD, **base_optimizer_args):
        defaults = dict(base_optimizer_args=base_optimizer_args)
        super(ADMMOptimizer, self).__init__(params, defaults)
        self.fixed_alpha = fixed_alpha
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_optimizer_args)
        self.z = [torch.zeros_like(p.data) for p in self.param_groups[0]['params']]
        self.u = [torch.zeros_like(p.data) for p in self.param_groups[0]['params']]

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        self.base_optimizer.step(closure)

        with torch.no_grad():
            for i, param in enumerate(self.param_groups[0]['params']):
                self.z[i] = self.z[i].to(param.device)
                self.u[i] = self.u[i].to(param.device)
                tensor = param.data + self.u[i]
                if self.fixed_alpha is None:
                    alpha = 0.7 * tensor.abs().mean().item()
                else:
                    alpha = self.fixed_alpha
                self.z[i].copy_(ternarize(tensor, alpha=alpha))
                self.u[i].add_(param.data - self.z[i])
                param.data.copy_(self.z[i])

        return loss

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
