import logging
import numpy as np
import openpnm as op
from porespy.filters import trim_nonpercolating_paths
from porespy.tools import Results
from porespy.generators import faces


logger = logging.getLogger(__name__)
ws = op.Workspace()


__all__ = ["tortuosity_fd",
           "check_percolating",
           "fickian_diffusion"]

def check_percolating(im, axis):
    r"""
    Trims floating pores in the specified direction. Throws an error if no pores
    remain after trimming.

    Parameters
    ----------
    im : ndarray
        The binary image to analyze with ``True`` indicating phase of interest
    axis : int
        The axis along which to trim pores

    Returns
    -------
    im : ndarray
        The binary image with the floating pores trimmed
    """
    # Obtain original porosity
    eps0 = im.sum(dtype=np.int64) / im.size

    # Remove floating pores
    inlets = faces(im.shape, inlet=axis)
    outlets = faces(im.shape, outlet=axis)
    im = trim_nonpercolating_paths(im, inlets=inlets, outlets=outlets)

    # Check if porosity is changed after trimmimg floating pores
    eps = im.sum(dtype=np.int64) / im.size
    if not eps:
        raise Exception("No pores remain after trimming floating pores")
    if eps < eps0:  # pragma: no cover
        logger.warning("Found non-percolating regions, were filled to percolate")

    return im

def fickian_diffusion(im, axis, cL=1.0, cR=0.0, solver=None):
    r"""
    Performs Fickian Diffusion on the image in the specified direction.

    Parameters
    ----------
    im : ndarray
        The binary image to analyze with ``True`` indicating phase of interest
    axis : int
        The axis along which to apply boundary conditions
    cL : float
        The Dirichlet boundary condition to be applied at the inlet
    cR : float
        The Dirichlet boundary condition to be applied at the outlet
    
    Returns
    -------

    results : Results object
        The following values are computed and returned as attributes:
        ======================= ===================================================
        Attribute               Description
        ======================= ===================================================
        r_in                    Molar flowrate into the image from the specified axis
        c_map                   The concentration map of the image calculated from
                                Fickian diffusion
        ======================= ===================================================
    
    """

    openpnm_v3 = op.__version__.startswith('3')

    # Generate a Cubic network to be used as an orthogonal grid
    net = op.network.CubicTemplate(template=im, spacing=1.0)
    if openpnm_v3:
        phase = op.phase.Phase(network=net)
    else:
        phase = op.phases.GenericPhase(network=net)
    phase['throat.diffusive_conductance'] = 1.0
    # Run Fickian Diffusion on the image
    fd = op.algorithms.FickianDiffusion(network=net, phase=phase)
    # Choose axis of concentration gradient
    inlets = net.coords[:, axis] <= 1
    outlets = net.coords[:, axis] >= im.shape[axis] - 1
    # Boundary conditions on concentration
    fd.set_value_BC(pores=inlets, values=cL)
    fd.set_value_BC(pores=outlets, values=cR)
    if openpnm_v3:
        if solver is None:
            solver = op.solvers.PyamgRugeStubenSolver(tol=1e-8)
        fd._update_A_and_b()
        fd.x, info = solver.solve(fd.A.tocsr(), fd.b)
        if info:
            raise Exception(f'Solver failed to converge, exit code: {info}')
    else:
        fd.settings.update({'solver_family': 'scipy', 'solver_type': 'cg'})
        fd.run()
    
        # Calculate molar flow rate
    r_in = fd.rate(pores=inlets)[0]
    r_out = fd.rate(pores=outlets)[0]
    if not np.allclose(-r_out, r_in, rtol=1e-4):  # pragma: no cover
        logger.error(f"Inlet/outlet rates don't match: {r_in:.4e} vs. {r_out:.4e}")

    # Free memory
    ws.close_project(net.project)

    conc = np.zeros(im.size, dtype=float)
    conc[net['pore.template_indices']] = fd['pore.concentration']
    conc = conc.reshape(im.shape)

    return (conc)
    

def tortuosity_fd(im, axis, c_map=None, cL=1.0, cR=0.0):
    r"""
    Calculates the tortuosity of image in the specified direction.

    Parameters
    ----------
    im : ndarray
        The binary image to analyze with ``True`` indicating phase of interest
    axis : int
        The axis along which to apply boundary conditions
    c_map : float
        The concentration map of the image. Can be obtained from `fickian_diffusion`.
    Returns
    -------
    results : Results object
        The following values are computed and returned as attributes:

        =================== ===================================================
        Attribute           Description
        =================== ===================================================
        im                  The image as provided
        tortuosity          Calculated using the ``effective_porosity`` as
                            :math:`\tau = \frac{D_{AB}}{D_{eff}} \cdot
                            \varepsilon`.
        porosity            Porosity of the as-received the image
        =================== ===================================================

    Examples
    --------
    `Click here
    <https://porespy.org/examples/simulations/reference/tortuosity_fd.html>`_
    to view online example.

    """
    from porespy.beta import tau_from_cmap

    if axis > (im.ndim - 1):
        raise Exception(f"'axis' must be <= {im.ndim}")

    eps = im.sum(dtype=np.int64) / im.size

    dC = cL - cR

    L = im.shape[axis]
    A = np.prod(im.shape) / L

    if c_map is None:
        tmp = fickian_diffusion(im = im,
                                axis = axis,
                                cL = cL,
                                cR = cR)
        c_map = tmp.c_map
    
    tau = tau_from_cmap(c = c_map,
                        im = im,
                        axis = axis)

    # Attach useful parameters to Results object
    result = Results()
    result.im = im
    result.tortuosity = tau
    result.porosity = eps
    result.concentration = c_map

    return result

if __name__ == "__main__":
    import porespy as ps
    import numpy as np

    np.random.seed(2)
    im = ps.generators.overlapping_spheres([200, 200], r=10, porosity=0.65)
    axis = 1
    im_trimmed = ps.simulations.check_percolating(im, axis)
    diffusion = ps.simulations.fickian_diffusion(im=im_trimmed, axis=axis)
    result = ps.simulations.tortuosity_fd(im_trimmed, axis)
    print(result)