"""
Core ESMSEU Regularizer: Heat kernel diffusion and stochastic trace estimation.
Implements Φ(R) and Tr(H) from the ESMSEU action [Juhariah et al., 2025].
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class ESMSEURegularizer:
    """
    ESMSEU Regularizer for curvature diffusion in parameter space.

    Computes total regularization: λ1 * Φ(R) + λ2 * R², where:
    - Φ(R): Heat kernel-smoothed curvature, K_τ(x,x') = (4πτ)^{-n/2} exp(-d²/(4τ)).
    - R: Curvature proxy via trace(Hessian) ≈ Hutchinson estimator.

    Args:
        lambda1 (float): Weight for heat kernel term (default: 0.1).
        lambda2 (float): Weight for curvature squared term (default: 0.01).
        tau (float): Diffusion time for heat kernel (default: 0.1).
        m (int): Hutchinson samples (default: 2; low-overhead).
        param_dim (int): Approx. manifold dimension (default: input_dim * layers).

    Example:
        reg = ESMSEURegularizer(lambda1=0.1)
        total_reg = reg.compute(model, loss_fn, dataloader)
    """
    def __init__(
        self,
        lambda1: float = 0.1,
        lambda2: float = 0.01,
        tau: float = 0.1,
        m: int = 2,
        param_dim: Optional[int] = None,
    ):
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.tau = tau
        self.m = m
        self.param_dim = param_dim
        self._device = None

    def _to_device(self, obj: torch.Tensor) -> torch.Tensor:
        """Move tensor to model device."""
        if self._device is None:
            self._device = next(iter(obj.parameters())) if hasattr(obj, 'parameters') else 'cpu'
        return obj.to(self._device)

    def hutchinson_trace(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        batch_size: int = 32,
    ) -> float:
        """
        Stochastic trace estimator: Tr(H) ≈ (1/m) ∑ v_i^T (H v_i), v_i ~ N(0,I).
        [Hutchinson, 1989; integrated in ESMSEU action].

        Args:
            model: PyTorch model.
            loss_fn: Loss function.
            dataloader: Data for gradient computation.
            batch_size: Sub-batch size.

        Returns:
            float: Estimated trace.
        """
        params = list(model.parameters())
        trace_est = 0.0
        n_batches = 0

        model.eval()
        with torch.no_grad():
            for batch in dataloader:
                if n_batches >= 1:  # One batch for efficiency
                    break
                x, y = self._to_device(batch[0][:batch_size]), self._to_device(batch[1][:batch_size])
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)

                for _ in range(self.m):
                    v = [torch.randn_like(p) for p in params]
                    Hv = torch.autograd.grad(
                        grads, params, grad_outputs=v, retain_graph=True,
                        allow_unused=True, create_graph=False
                    )
                    trace_contrib = sum(
                        (h * vi).sum().item() for h, vi in zip(Hv, v)
                        if h is not None and vi is not None
                    )
                    trace_est += trace_contrib
                n_batches += 1

        return trace_est / (self.m * n_batches) if n_batches > 0 else 0.0

    def heat_kernel_reg(self, model: nn.Module, n: int) -> float:
        """
        Heat kernel Φ(R): Approximate diffusion over parameter manifold.
        K_τ ≈ (4πτ)^{-n/2} exp(-||params||² / (4τ)), proxy for geometric smoothing.

        Args:
            model: PyTorch model.
            n: Manifold dimension (param_dim).

        Returns:
            float: Smoothed curvature term.
        """
        params = torch.cat([p.flatten() for p in model.parameters()])
        d_squared = torch.norm(params) ** 2 / len(params)
        kernel = (4 * np.pi * self.tau) ** (-n / 2) * torch.exp(-d_squared / (4 * self.tau))
        return kernel.item()

    def compute(
        self,
        model: nn.Module,
        loss_fn: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        input_dim: int = 784,  # e.g., MNIST flattened
    ) -> float:
        """
        Total ESMSEU regularization: λ1 Φ(R) + λ2 [Tr(H)]².

        Args:
            model: Trained model.
            loss_fn: Task loss.
            dataloader: Current task data.
            input_dim: Approx. for n (param space dim).

        Returns:
            float: Regularization value.
        """
        n = self.param_dim or (input_dim * 10)  # Rough estimate for MLP layers
        trace = self.hutchinson_trace(model, loss_fn, dataloader)
        phi_r = self.heat_kernel_reg(model, n)
        reg_total = self.lambda1 * phi_r + self.lambda2 * (trace ** 2)
        logger.debug(f"Trace: {trace:.4f}, Φ(R): {phi_r:.4f}, Total Reg: {reg_total:.4f}")
        return reg_total
