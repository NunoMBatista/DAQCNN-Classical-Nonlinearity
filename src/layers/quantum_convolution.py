import torch
import torch.nn as nn

from src.layers.daqk import DAQKLayer
from src.physics.kernel_topologies import get_3x3_kernel_set


class QuantumConv2d(nn.Module):
    """Sliding-window quantum convolution over grayscale images.

    Each image is min-max normalised to [0, pi] (per image, not per patch), cut
    into kernel_size x kernel_size patches with nn.Unfold, and every patch goes
    through DAQKLayer. Output: (B, num_kernels * measurements_per_kernel, H_out, W_out).
    """

    def __init__(
        self,
        kernel_size=3,
        stride=1,
        kernel_topology_names=None,
        scaling_factor=1.0,
        evolution_time=0.2,
        mode="trotter",
        quantum_device="default.qubit",
        quantum_device_kwargs=None,
        interface="torch",
        include_correlators=False,
        encoding_mode="digital",
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_qubits = kernel_size * kernel_size

        if kernel_topology_names is None:
            kernel_topology_names = list(get_3x3_kernel_set().keys())

        self.quantum_kernel = DAQKLayer(
            n_qubits=self.n_qubits,
            grid_size=kernel_size,
            scaling_factor=scaling_factor,
            evolution_time=evolution_time,
            mode=mode,
            kernel_topology_names=kernel_topology_names,
            quantum_device=quantum_device,
            quantum_device_kwargs=quantum_device_kwargs,
            interface=interface,
            include_correlators=include_correlators,
            encoding_mode=encoding_mode,
        )
        self.num_kernels = self.quantum_kernel.num_kernels
        self.measurements_per_kernel = self.quantum_kernel.measurements_per_kernel
        self.out_channels = self.num_kernels * self.measurements_per_kernel

        self.unfold = nn.Unfold(kernel_size=kernel_size, stride=stride)

    def _normalize_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Per-image min-max normalisation to [0, pi]."""
        x_min = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        x_max = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        x_norm = (x - x_min) / (x_max - x_min + 1e-8)
        return x_norm * torch.pi

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if C != 1:
            raise ValueError(f"QuantumConv2d expects 1-channel images, got {C}")
        h_out = (H - self.kernel_size) // self.stride + 1
        w_out = (W - self.kernel_size) // self.stride + 1

        x = self._normalize_inputs(x)
        patches = self.unfold(x)                                   # (B, k*k, n_patches)
        n_patches = patches.shape[-1]
        patches = patches.transpose(1, 2).reshape(-1, self.n_qubits)
        q_out = self.quantum_kernel(patches)                       # (B*n_patches, out_channels)
        q_out = q_out.reshape(B, n_patches, self.out_channels).transpose(1, 2)
        return q_out.reshape(B, self.out_channels, h_out, w_out)
