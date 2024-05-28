import torch
import torch.optim as optim


class ADMMOptimizer(optim.Optimizer):
    def __init__(self, params, rho=1e-4, base_optimizer_cls=optim.SGD, **base_optimizer_args):
        defaults = dict(rho=rho, base_optimizer_args=base_optimizer_args)
        super(ADMMOptimizer, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer_cls(self.param_groups, **base_optimizer_args)
        self.z = [torch.zeros_like(p.data) for p in self.param_groups[0]['params']]
        self.u = [torch.zeros_like(p.data) for p in self.param_groups[0]['params']]

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        rho = self.defaults['rho']

        # Base optimizer step
        self.base_optimizer.step(closure)

        # ADMM update
        with torch.no_grad():
            for i, param in enumerate(self.param_groups[0]['params']):
                self.z[i] = self.z[i].to(param.device)
                self.u[i] = self.u[i].to(param.device)
                self.z[i].copy_(self.proximal_mapping(param.data + self.u[i], rho))
                self.u[i].add_(param.data - self.z[i])
                param.data.copy_(self.z[i])

        return loss

    @staticmethod
    def proximal_mapping(tensor, rho):
        delta = 0.7 * tensor.norm(1).item() / tensor.numel()
        alpha = 0.7 * tensor.abs().mean().item()
        return alpha * ((tensor.abs() > delta).float() * tensor.sign())

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)
