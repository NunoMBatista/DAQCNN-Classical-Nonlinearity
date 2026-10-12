import torch
import torch.nn as nn

from src.layers.quantum_convolution import QuantumConv2d
from src.utils.training_utils import build_classification_head


class DAQCNN(nn.Module):
    """Quantum convolution front-end (frozen) + trainable CNN head.

    When a cached feature set is used, the training loop sets bypass_quantum=True
    and the model receives the cached quantum features instead of images.
    override_quantum_out_channels sets the head's input width from the cache.
    The quantum layer is always built with digital encoding: cached runs never
    call it, so encoding_mode only matters for cache lookup.
    """

    def __init__(
        self,
        num_classes: int,
        kernel_size: int = 3,
        stride: int = 1,
        kernel_topology_names=None,
        scaling_factor: float = 1.0,
        evolution_time: float = 0.2,
        mode: str = "trotter",
        dropout: float = 0.1,
        activation: str = "relu",
        quantum_device: str = "default.qubit",
        quantum_device_kwargs=None,
        classical_device=None,
        interface: str = "torch",
        override_quantum_out_channels: int = None,
        include_correlators: bool = False,
        head_hidden_channels: int = 64,
    ):
        super().__init__()

        self.bypass_quantum = False

        self.quantum_convolutional_layer = QuantumConv2d(
            kernel_size=kernel_size,
            stride=stride,
            kernel_topology_names=kernel_topology_names,
            scaling_factor=scaling_factor,
            evolution_time=evolution_time,
            mode=mode,
            quantum_device=quantum_device,
            quantum_device_kwargs=quantum_device_kwargs,
            interface=interface,
            include_correlators=include_correlators,
        )

        if override_quantum_out_channels is not None:
            out_ch = override_quantum_out_channels
        else:
            out_ch = self.quantum_convolutional_layer.out_channels

        self.head = build_classification_head(
            out_ch, num_classes, dropout, activation, hidden_channels=head_hidden_channels
        )

        if classical_device is not None:
            self.to(classical_device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.bypass_quantum:
            x = self.quantum_convolutional_layer(x).float()
        target_device = next(self.head.parameters()).device
        if x.device != target_device:
            x = x.to(target_device)
        return self.head(x)
