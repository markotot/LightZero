from typing import Tuple

import numpy as np
import torch
from sympy.stats.sampling.sample_numpy import numpy


class Cache:
    def __init__(self, num_samples: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        assert embed_dim % num_heads == 0
        self._n, self._cache, self._size = num_samples, None, None
        self._reset = lambda n: torch.empty(n, num_heads, max_tokens, embed_dim // num_heads, device=device)  # (B, nh, T, hs)
        self.reset()

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        n, num_heads, _, head_dim = self._cache.shape
        return n, num_heads, self._size, head_dim

    def reset(self) -> None:
        self._cache = self._reset(self._n)
        self._size = 0

    def prune(self, mask: np.ndarray) -> None:
        assert mask.ndim == 1 and mask.shape[0] == self.shape[0]
        self._cache = self._cache[mask]
        self._n = self._cache.shape[0]

    def get(self) -> torch.Tensor:
        return self._cache[:, :, :self._size, :]

    def update(self, x: torch.Tensor) -> None:
        assert (x.ndim == self._cache.ndim) and all([x.size(i) == self._cache.size(i) for i in (0, 1, 3)])
        assert self._size + x.size(2) <= self._cache.shape[2]
        self._cache = AssignWithoutInplaceCheck.apply(self._cache, x, 2, self._size, self._size + x.size(2))
        self._size += x.size(2)


class KVCache:
    def __init__(self, n: int, num_heads: int, max_tokens: int, embed_dim: int, device: torch.device) -> None:
        self._k_cache = Cache(n, num_heads, max_tokens, embed_dim, device)
        self._v_cache = Cache(n, num_heads, max_tokens, embed_dim, device)

    @property
    def shape(self) -> Tuple[int, int, int, int]:
        return self._k_cache.shape

    def reset(self) -> None:
        self._k_cache.reset()
        self._v_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        self._k_cache.prune(mask)
        self._v_cache.prune(mask)

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._k_cache.get(), self._v_cache.get()

    def update(self, k: torch.Tensor, v: torch.Tensor):
        self._k_cache.update(k)
        self._v_cache.update(v)


class KeysValues:
    def __init__(self, n: int, num_heads: int, max_tokens: int, embed_dim: int, num_layers: int, device: torch.device) -> None:

        self.n = n
        self.num_heads = num_heads
        self.max_tokens = max_tokens
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.device = device

        self._keys_values = tuple([KVCache(n, num_heads, max_tokens, embed_dim, device) for _ in range(num_layers)])

    def __getitem__(self, key: int) -> KVCache:
        return self._keys_values[key]

    def __len__(self):
        return len(self._keys_values)

    @property
    def size(self):
        return self._keys_values[0].shape[2]

    def reset(self) -> None:
        for kv_cache in self._keys_values:
            kv_cache.reset()

    def prune(self, mask: np.ndarray) -> None:
        for kv_cache in self._keys_values:
            kv_cache.prune(mask)

    def to_device(self, device: str):
        """
        Transfer all KVCache objects within the KeysValues object to a certain device.
        Not used in the current implementation.

        Arguments:
            - self._keys_values (KeysValues): The KeysValues object to be transferred.
            - device (str): The device to transfer to.
        Returns:
            - keys_values (KeysValues): The KeysValues object with its caches transferred to the specified device.
        """
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        for kv_cache in self._keys_values:
            kv_cache._k_cache._cache = kv_cache._k_cache._cache.to(device)
            kv_cache._v_cache._cache = kv_cache._v_cache._cache.to(device)
        return self._keys_values

    def to_numpy(self):

        x = np.random.normal(0, 1, (self.num_layers, 2, 1, self.num_heads, self.max_tokens, self.embed_dim // self.num_heads)).astype(np.float32)
        # x = np.zeros((self.num_layers, 2, 1, self.num_heads, self.max_tokens, self.embed_dim // self.num_heads))
        # for index, kv_cache in enumerate(self._keys_values):
        #     x[index][0] = kv_cache._k_cache._cache.detach().cpu().numpy()
        #     x[index][1] = kv_cache._v_cache._cache.detach().cpu().numpy()
        return x

    def to_tensor(self, x, device):
        pass
        # for index, element in enumerate(x):
        #     k_cache= torch.from_numpy(element[0]).to(device)
        #     v_cache = torch.from_numpy(element[1]).to(device)
        #     self._keys_values[index]._k_cache._cache = k_cache
        #     self._keys_values[index]._v_cache._cache = v_cache

class AssignWithoutInplaceCheck(torch.autograd.Function):
    """
    Inspired from : https://discuss.pytorch.org/t/disable-in-place-correctness-version-check-any-other-workaround/90738/4
    Warning : do not use it to overwrite a slice twice.
    """

    @staticmethod
    def get_slice(dim: int, start: int, stop: int) -> Tuple[slice]:
        return tuple([slice(None), ] * dim + [slice(start, stop)])

    @staticmethod
    def forward(ctx, input: torch.Tensor, value: torch.Tensor, dim: int, start: int, stop: int) -> torch.Tensor:
        ctx.dim = dim
        ctx.start = start
        ctx.stop = stop
        input.data[AssignWithoutInplaceCheck.get_slice(dim, start, stop)] = value
        return input

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor) -> Tuple[torch.Tensor]:
        return grad_out, grad_out[AssignWithoutInplaceCheck.get_slice(ctx.dim, ctx.start, ctx.stop)], None, None, None
