from itertools import combinations

import pennylane as qml
import torch
import torch.nn as nn
from pennylane import numpy as np

import src.physics.evolution as evo
import src.physics.hamiltonian as phys
import src.physics.kernel_topologies as topologies

try:
    import jax

    jax.config.update("jax_enable_x64", True)  # the exact (qml.evolve) solver needs float64
except ImportError:  # jax is only needed for mode="exact"
    pass


class DAQKLayer(nn.Module):
    """Fixed (untrained) Rydberg quantum kernels applied to flattened patches.

    Input (batch, n_qubits) with values in [0, pi]. Output (batch, num_kernels *
    measurements_per_kernel): per topology, the n_qubits <Z_i> followed (if
    include_correlators) by every <Z_i Z_j>, i < j, in itertools.combinations order.

    encoding_mode="digital": RY(x_i) then H on each qubit, then evolution under H(t).
    encoding_mode="analog": start in |0...0>, x_i enters H(t) as a local term x_i n_i.
    """

    def __init__(
        self,
        n_qubits=9,
        grid_size=3,
        scaling_factor=1.0,
        evolution_time=0.2,
        mode="trotter",
        kernel_topology_names=("kings",),
        quantum_device="default.qubit",
        quantum_device_kwargs=None,
        interface="torch",
        include_correlators=False,
        encoding_mode="digital",
    ):
        super().__init__()

        self.n_qubits = n_qubits
        self.wires = list(range(n_qubits))
        self.mode = mode
        self.evolution_time = evolution_time
        self.interface = interface
        self.include_correlators = include_correlators
        self.encoding_mode = encoding_mode
        self.kernel_topology_names = kernel_topology_names

        n_zz = len(list(combinations(self.wires, 2))) if include_correlators else 0
        self.measurements_per_kernel = n_qubits + n_zz

        coord_sets = topologies.build_kernel_coordinate_sets(grid_size, kernel_topology_names)
        self.num_kernels = len(coord_sets)

        def make_circuit(H):
            def circuit(inputs):
                if encoding_mode == "digital":
                    for i in range(n_qubits):
                        qml.RY(inputs[..., i], wires=i)
                        qml.Hadamard(wires=i)
                    evo.evolve_analog_block(H, time_interval=[0, evolution_time], mode=mode, dt=0.05)
                else:
                    # params: [omega scale, delta scale, interaction (unused), x_0, ..., x_{n-1}]
                    if torch.is_tensor(inputs):
                        p_list = [1.0, 1.0, 1.0] + inputs.tolist()
                    else:
                        p_list = [1.0, 1.0, 1.0] + list(inputs)
                    evo.evolve_analog_block(H, time_interval=[0, evolution_time], mode=mode, dt=0.05,
                                            params=p_list)

                measurements = [qml.expval(qml.PauliZ(i)) for i in self.wires]
                if include_correlators:
                    for i, j in combinations(self.wires, 2):
                        measurements.append(qml.expval(qml.PauliZ(i) @ qml.PauliZ(j)))
                return measurements

            return circuit

        # one Hamiltonian, device and QNode per topology
        self.kernel_circuits = []
        for coords in coord_sets:
            hamiltonian = phys.get_rydberg_hamiltonian(
                self.wires, coords, scaling_factor=scaling_factor,
                use_local_detuning=(encoding_mode == "analog"),
            )
            dev = qml.device(quantum_device, wires=self.n_qubits, **(quantum_device_kwargs or {}))
            self.kernel_circuits.append(
                qml.QNode(make_circuit(hamiltonian), dev, interface=interface, diff_method=None)
            )

    def forward(self, x):
        per_kernel = []
        for kernel_circuit in self.kernel_circuits:
            if self.interface == "autograd" or self.mode == "exact":
                circuit_output = np.array(kernel_circuit(x.detach().cpu().numpy()))
                out = torch.as_tensor(circuit_output, device=x.device, dtype=x.dtype)
            elif self.encoding_mode == "analog" and x.ndim > 1:
                # the data sit in the Hamiltonian coefficients, which
                # ApproxTimeEvolution cannot batch, so run patch by patch
                batch_out = []
                for i in range(x.shape[0]):
                    batch_out.append(torch.stack(kernel_circuit(x[i]), dim=0))
                out = torch.stack(batch_out, dim=0)
            else:
                # batched digital run returns one (batch,) tensor per measurement
                out = torch.stack(kernel_circuit(x), dim=1)
            per_kernel.append(out)
        return torch.cat(per_kernel, dim=1)
