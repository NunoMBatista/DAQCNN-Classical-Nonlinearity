"""Atom coordinates for the eight 3x3 kernel topologies.

Atom i sits under patch pixel i (row-major). The coupling between two atoms is
C6 / r^6, so moving an atom to FAR effectively decouples it.
"""

import numpy as np

FAR = 100.0


def get_3x3_kernel_set():
    """name -> (9, 2) array of atom coordinates."""
    kernels = {}

    base = []
    for r in range(3):
        for c in range(3):
            base.append([float(r), float(c)])
    base = np.array(base, dtype=float)

    # full unit-spacing grid: nearest and diagonal neighbours coupled
    kernels["kings"] = base.copy()

    # rows decoupled from each other
    coords_horz = base.copy()
    coords_horz[:, 0] *= FAR
    kernels["horizontal"] = coords_horz

    # columns decoupled from each other
    coords_vert = base.copy()
    coords_vert[:, 1] *= FAR
    kernels["vertical"] = coords_vert

    # plus sign: the four corners are isolated
    coords_cross = base.copy()
    for idx in [0, 2, 6, 8]:
        coords_cross[idx] = [FAR, FAR]
    kernels["cross"] = coords_cross

    # hollow centre
    coords_ring = base.copy()
    coords_ring[4] = [FAR, FAR]
    kernels["ring"] = coords_ring

    # all nine atoms on one line, unit spacing
    coords_chain = np.zeros((9, 2), dtype=float)
    for i in range(9):
        coords_chain[i] = [0.0, float(i)]
    kernels["chain"] = coords_chain

    # centre atom 4 at the origin, the other eight on the unit circle
    coords_star = np.zeros((9, 2), dtype=float)
    coords_star[4] = [0.0, 0.0]
    angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    spoke_idx = [0, 1, 2, 3, 5, 6, 7, 8]
    for i, idx in enumerate(spoke_idx):
        coords_star[idx] = [np.cos(angles[i]), np.sin(angles[i])]
    kernels["star"] = coords_star

    # kings scaled by 1.5: every coupling ~11x weaker, diagonals 1/8 of the nearest ones
    coords_grid = base.copy()
    coords_grid *= 1.5
    kernels["grid"] = coords_grid

    # Names of the poly-2 / RFF classical caches. Never simulated (the cache is
    # always found and the quantum layer is bypassed); listed so that DAQCNN can
    # be constructed from those configs.
    for synthetic in ("poly2", "rff45", "rff180"):
        kernels[synthetic] = base.copy()

    return kernels


def build_kernel_coordinate_sets(grid_size, names):
    """Coordinate arrays for the named topologies, in the order given."""
    if grid_size != 3:
        raise ValueError(f"Only 3x3 kernels are supported, got grid_size={grid_size}")
    kernel_map = get_3x3_kernel_set()
    kernel_set = []
    for n in names:
        if n not in kernel_map:
            raise ValueError(f"Kernel '{n}' not found. Available kernels: {list(kernel_map)}")
        kernel_set.append(kernel_map[n])
    return kernel_set
