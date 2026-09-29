import torch
import torch.nn as nn
from torch import Tensor
from einops import einsum, rearrange
from jaxtyping import Float, Bool, Int
from math import sqrt,cos,pi

def cross_entropy(
        logits: Float[Tensor,"... vocab_size"],
        target: Int[Tensor, "..."]
    ) -> Float[Tensor,"..."]:
    max_logit = torch.max(logits,dim = -1,keepdim = True).values
    denominator = torch.exp(logits - max_logit).sum(dim = -1)
    result = logits.gather(dim = -1,index = target.unsqueeze(-1)).squeeze(-1)   #key
    loss = - result + torch.max(logits,dim = -1).values + torch.log(denominator)
    return loss.mean()

class AdamW(torch.optim.Optimizer):
    def __init__(
            self,
            params,
            lr: float = 1e-3,
            betas: tuple[float,float] = (0.9,0.99),
            eps: float = 1e-8,
            weight_decay: float = 0.01
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)
    @torch.no_grad() #为了告诉 PyTorch：下面的参数更新只是优化算法操作，不要把它记录到自动求导计算图中。
    def step(self, closure = None):
        loss = None 
        if closure is not None:
            with torch.enable_grad():
                loss = closure() #closure暂时没什么用

        for group in self.param_groups:
            lr = group['lr']
            beta_1, beta_2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get('t', 1)
                if 'm' not in state:
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                m = state['m']
                v = state['v']
                grad = p.grad
                alpha = lr * (sqrt(1-beta_2 ** t) / (1-beta_1 ** t))
                #原地操作，节省显存
                m.mul_(beta_1).add_(grad, alpha = 1-beta_1)
                v.mul_(beta_2).addcmul_(grad,grad,value=1-beta_2) #原地加上两个 Tensor 的逐元素乘积
                p -= lr * weight_decay * p
                p -= alpha * (m / (torch.sqrt(v) + eps))
                state["t"] = t + 1 # Increment iteration number.
        return loss

def learning_rate_schedule(t,alpha_max,alpha_min,T_w,T_c) -> float:
    if t< T_w:
        return (t/T_w) * alpha_max
    elif t <= T_c:
        return alpha_min + (1/2) * (1 + cos(((t-T_w)/(T_c-T_w))*pi)) * (alpha_max-alpha_min)
    else:
        return alpha_min
    
@torch.no_grad()
def gradient_clipping(params, M, eps = 1e-6):
    grads = [p.grad for p in params if p.grad is not None]
    if len(grads) == 0:
        return 0
    norm = torch.sqrt(sum(torch.sum(grad ** 2) for grad in grads))
    if norm > M:
        scale = M/(norm+eps)
        for grad in grads:
            grad.mul_(scale)
        return torch.sqrt(sum(torch.sum(grad ** 2) for grad in grads)).item()
    else:
        return norm.item()