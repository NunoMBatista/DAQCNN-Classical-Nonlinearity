import numpy as np
import pennylane as qml


def evolve_analog_block(hamiltonian, time_interval, mode="trotter", dt=0.05, params=None):
    """Apply exp(-i int H(t) dt) over time_interval = [t_start, t_end] to the circuit.

    mode="trotter": H is frozen at the start of each step of length dt and each
    step is one first-order ApproxTimeEvolution (the production feature map).
    mode="exact": PennyLane's ODE solver (qml.evolve).
    params defaults to 1.0 for every coefficient of H.
    """
    if params is None:
        params = [1.0 for _ in hamiltonian.coeffs]
    elif isinstance(params, (int, float)):
        params = [float(params) for _ in hamiltonian.coeffs]

    t_start, t_end = time_interval

    if mode == "exact":
        qml.evolve(hamiltonian)(params=np.asarray(params, dtype=float),
                                t=np.asarray(time_interval, dtype=float))
    elif mode == "trotter":
        steps = int(round((t_end - t_start) / dt))
        for step in range(steps):
            H_static = hamiltonian(params=params, t=t_start + step * dt)
            qml.ApproxTimeEvolution(H_static, dt, n=1)
    else:
        raise ValueError(f"Unknown evolution mode: {mode}")
