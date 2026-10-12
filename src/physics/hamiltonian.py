import numpy as np
import numpy.linalg as npl
import pennylane as qml


def _scalar_param(p, default=1.0):
    """A ramp parameter as a float; accepts None, scalars and (jax/numpy) arrays."""
    if p is None:
        return float(default)
    if isinstance(p, (int, float)):
        return float(p)
    arr = np.asarray(p)
    if arr.shape == ():
        return float(arr)
    return float(arr.flat[0]) if arr.size > 0 else float(default)


def projector_eta(wire):
    """Rydberg occupation n = (1 - Z) / 2."""
    return 0.5 * (qml.Identity(wire) - qml.PauliZ(wire))


def get_rydberg_hamiltonian(wires, coordinates, scaling_factor=1.0, use_local_detuning=False):
    """H(t) = (Omega(t)/2) sum X_i - Delta(t) sum n_i + sum_{i<j} C6/r_ij^6 n_i n_j  [+ sum x_i n_i].

    Omega(t) = Omega0 * t and Delta(t) = Delta0 * t, with Omega0 and Delta0 the first
    two parameters (1.0 by default). scaling_factor is C6. With use_local_detuning
    (analog encoding) one extra term x_i n_i per atom is added, its coefficient x_i
    being the pixel value passed as a parameter.
    """
    n_qubits = len(wires)

    H_drive_base = qml.Hamiltonian([0.5] * n_qubits, [qml.PauliX(w) for w in wires])
    H_detuning_base = qml.Hamiltonian([-1.0] * n_qubits, [projector_eta(w) for w in wires])

    coeffs_int = []
    ops_int = []
    for i in range(n_qubits):
        for j in range(i + 1, n_qubits):
            delta = np.asarray(coordinates[i], dtype=float) - np.asarray(coordinates[j], dtype=float)
            r_ij = float(npl.norm(delta))
            if r_ij > 1e-6:
                coeffs_int.append(scaling_factor / (r_ij**6))
                ops_int.append(projector_eta(wires[i]) @ projector_eta(wires[j]))
    H_interaction = qml.Hamiltonian(coeffs_int, ops_int)

    def drive_ramp(params, t):
        return _scalar_param(params, default=1.0) * t

    def detuning_ramp(params, t):
        return _scalar_param(params, default=1.0) * t

    def constant_one(params, t):
        return 1.0

    coeffs = [drive_ramp, detuning_ramp]
    ops = [H_drive_base, H_detuning_base]

    if len(coeffs_int) > 0:
        ops.append(H_interaction)
        coeffs.append(constant_one)

    if use_local_detuning:
        def local_coeff(p, t):
            return p

        for i in range(n_qubits):
            ops.append(projector_eta(wires[i]))
            coeffs.append(local_coeff)

    # each coefficient function receives its own entry of the params list
    return qml.pulse.ParametrizedHamiltonian(coeffs, ops)
