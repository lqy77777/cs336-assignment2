import torch
import torch.nn as nn
from math import sqrt
from torch import Tensor
from einops import einsum, rearrange
from jaxtyping import Float, Bool, Int

class Linear(nn.Module):
    def __init__(
            self,
            in_features: int,
            out_features: int,
            device = None,
            dtype = None
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(out_features, in_features,device = device,dtype = dtype))
        self.std = sqrt(2/(out_features+in_features))
        nn.init.trunc_normal_(self.weights,mean =0.0,
                              std = self.std,
                              a = -3 * self.std,b = 3 * self.std)
    def forward(self, x: Tensor)-> Tensor:
        return einsum(self.weights,x,"d_out d_in, ... d_in -> ... d_out")
class embedding(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            d_model: int,
            device = None,
            dtype = None
    ):
        super().__init__()
        self.weights = nn.Parameter(torch.empty(vocab_size,d_model,device = device,dtype = dtype))
        nn.init.trunc_normal_(self.weights,mean = 0.0,
                              std = 1.0, a = -3.0, b = 3.0)
    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weights[token_ids]
class rmsnorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            eps: float = 1e-5,
            device = None,
            dtype = None
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(d_model,device = device,dtype = dtype))
    def forward(self,x : Tensor) -> Tensor:
        """
        本作业要求计算 RMSNorm 时先把输入转成 float32，
        计算完成后，再把输出转回输入原来的类型。 这是为了提高数值稳定性
        """
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(x.square().mean(dim = -1, keepdim = True) + self.eps)
        result = (x * self.weights) / rms
        return result.to(in_dtype)

class FFN(nn.Module):
    def __init__(
            self,
            d_ff: int,
            d_model: int,
            device = None,
            dtype = None
    ):
        super().__init__()
        self.W1 = Linear(d_model,d_ff,device=device,dtype =dtype)
        self.W2 = Linear(d_ff,d_model,device=device,dtype =dtype)
        self.W3 = Linear(d_model,d_ff,device=device,dtype =dtype)
    def forward(self, x:Tensor)->Tensor:
        W_1 = self.W1(x)
        SiLU = (W_1/ (1 + torch.exp(-W_1)))
        return self.W2(SiLU * self.W3(x))
def softmax(x : Tensor,dim : int) -> Tensor:
    max_value = x.max(dim = dim,keepdim = True).values
    shifted_x = x - max_value
    denominator = torch.exp(shifted_x).sum(dim = dim,keepdim = True)
    return torch.exp(shifted_x) / denominator
class rope(nn.Module):
    def __init__(
            self,
            theta: float,
            d_k: int,
            max_seq_len: int,
            device = None,
            dtype = None
    ):
        super().__init__()
        numerator = torch.arange(0,max_seq_len,device= device,dtype= dtype)
        denominator = 1 / torch.pow(theta,(2 * torch.arange(1,d_k // 2 + 1,device= device,dtype= dtype) - 2)/ d_k)
        positions = einsum(numerator,denominator,"i,k -> i k")
        #将rope的cos和sin注册为buffer，这样不会被优化器训练，也能跟随模块调用.to(device)
        #persistent = False时不会写入state_dict
        self.register_buffer("cos",torch.cos(positions),persistent = False)
        self.register_buffer("sin",torch.sin(positions),persistent = False)
    def forward(
            self,
            x: Float[Tensor,"... seq_len d_k"], 
            token_positions: Int[Tensor, "... seq_len"]
        ) -> Tensor:
        odd = x[..., ::2]
        even = x[..., 1::2]
        cos, sin = self.cos[token_positions], self.sin[token_positions]
        a = cos * odd - sin * even
        b = sin * odd + cos * even
        return torch.stack((a,b), dim = -1).flatten(-2)
def scaled_dot_product_attention(
        Q: Float[Tensor, " ... query d_k"],
        K: Float[Tensor, " ... key d_k"],
        V: Float[Tensor, " ... key d_v"],
        mask: Bool[Tensor, "... query key"] = None
    ) -> Float[Tensor, " ... query d_v"]:
    d_k = Q.size(-1)
    weight = einsum(Q,K,"... query d_k, ... key d_k -> ... query key")
    if mask is not None:
        weight = torch.where(mask,weight, float('-inf'))
    return einsum(softmax(weight / sqrt(d_k), dim = -1),V, "... query key, ... key d_v -> ... query d_v")
    

class multihead_self_attention(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            theta: float = None,
            max_seq_len: int = None,
            device = None,
            dtype = None
        ):
        super().__init__()
        self.W_Q = Linear(d_model,d_model,device = device,dtype = dtype)
        self.W_K = Linear(d_model,d_model,device = device,dtype = dtype)
        self.W_V = Linear(d_model,d_model,device = device,dtype = dtype)
        self.W_O = Linear(d_model,d_model,device = device,dtype = dtype)
        self.d_model = d_model
        self.num_heads = num_heads
        self.theta = theta
        if theta is not None:
            self.rope = rope(self.theta, d_model // num_heads, max_seq_len,device = device, dtype=dtype)
    def forward(self,x: Tensor, token_positions = None) -> Tensor:
        d_k = self.d_model // self.num_heads
        seq_len = x.size(-2)
        Q, K, V = self.W_Q(x), self.W_K(x), self.W_V(x)
        Q = rearrange(Q,"... seq_len (h d) -> ... h seq_len d", d = d_k)
        K = rearrange(K,"... seq_len (h d) -> ... h seq_len d", d = d_k)
        V = rearrange(V,"... seq_len (h d) -> ... h seq_len d", d = d_k)
        if self.theta is not None:
            if token_positions is None:
                token_positions = torch.arange(0,seq_len,device = x.device)
            token_positions = token_positions.unsqueeze(-2)   #why 
            Q,K = self.rope(Q,token_positions),self.rope(K,token_positions)
        mask = torch.tril(torch.ones(seq_len,seq_len,device = x.device,dtype = torch.bool))
        A = scaled_dot_product_attention(Q,K,V,mask)
        A = rearrange(A, "... h seq_len d -> ... seq_len (h d)")
        return self.W_O(A)
class transformer_block(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float,
            max_seq_len: int,
            device = None,
            dtype = None
    ):
        super().__init__()
        self.rms1 = rmsnorm(d_model,device = device,dtype = dtype)
        self.Attention = multihead_self_attention(d_model,num_heads,theta,max_seq_len,device = device, dtype = dtype)
        self.rms2 = rmsnorm(d_model,device = device,dtype = dtype)
        self.FFN = FFN(d_ff, d_model,device,dtype)
    def forward(self,x:Tensor) -> Tensor:
        y = x + self.Attention(self.rms1(x))
        return y + self.FFN(self.rms2(y))

class transformer_lm(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            seq_len: int,
            num_layers: int,
            d_model: int,
            num_heads: int,
            d_ff: int,
            theta: float,
            device = None,
            dtype = None
    ):
        super().__init__()
        self.embedding = embedding(vocab_size,d_model,device,dtype)
        self.layers = nn.ModuleList([transformer_block(d_model,num_heads,d_ff,theta,seq_len,device,dtype) 
                                     for _ in range(num_layers) ])
        self.rms = rmsnorm(d_model,device=device,dtype =dtype)
        self.lm_head = Linear(d_model,vocab_size,device,dtype)
    def forward(self,x:Tensor) -> Tensor:
        x = self.embedding(x)
        for block in self.layers:
            x = block(x)
        return self.lm_head(self.rms(x))