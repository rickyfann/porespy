import time
from porespy import tools
from porespy.tools import Results
import numpy as np
import openpnm as op
import pandas as pd
import dask
try:
    from pyedt import edt
except:
    from edt import edt


__all__ = [
    'tortuosity_bt',
    'get_block_sizes',
    'df_to_tortuosity',
    'rev_tortuosity',
    'analyze_blocks',
]


def calc_g(im, axis, solver_args={}):
    r"""
    Calculates diffusive conductance of an image in the direction specified

    Parameters
    ----------
    im : ndarray
        The binary image to analyze with ``True`` indicating phase of interest.
    axis : int
        0 for x-axis, 1 for y-axis, 2 for z-axis.
    solver_args : dict
        Dicionary of keyword arguments to pass on to the solver.  The most
        relevant one being `'tol'` which is 1e-6 by default. Using larger values
        might improve speed at the cost of accuracy.

    Returns
    -------
    results : dataclass-like
        An object with the results of the calculation as attributes.

    Notes
    -----
    This is intended to receive blocks of a larger image and is used by
    `tortuosity_bt`.
    """
    from porespy.simulations import tortuosity_fd
    solver_args = {'tol': 1e-6} | solver_args
    solver = solver_args.pop('solver', None)
    try:
        solver = op.solvers.PyamgRugeStubenSolver(**solver_args)
        results = tortuosity_fd(im=im, axis=axis, solver=solver)
    except Exception:
        t0 = time.perf_counter()
        results = Results()
        results.effective_porosity = 0.0
        results.original_porosity = im.sum()/im.size
        results.tortuosity = np.inf
        results.time = time.perf_counter() - t0
    L = im.shape[axis]
    A = np.prod(im.shape)/im.shape[axis]
    g = (results.effective_porosity * A) / (results.tortuosity * (L - 1))
    results.diffusive_conductance = g
    results.volume = np.prod(im.shape)
    results.axis = axis
    return results


def get_block_sizes(shape, block_size_range=[10, 100]):
    """
    Finds all viable block sizes between lower and upper limits

    Parameters
    ----------
    shape : sequence of int
        The [x, y, z] size of the image
    block_size_range : sequence of 2 ints
        The [lower, upper] range of the desired block sizes. Default is [10, 100]

    Returns
    -------
    sizes : ndarray
        All the viable block sizes in the specified range

    Notes
    -----
    This is called by `rev_tortuosity` to determine what size blocks to use.
    """
    Lmin, Lmax = block_size_range
    a = np.ceil(min(shape)/Lmax).astype(int)
    block_sizes = min(shape) // np.arange(a, 9999)  # Generate WAY more than needed
    block_sizes = np.unique(block_sizes[block_sizes >= Lmin])
    return block_sizes


def rev_tortuosity(im, block_sizes=None, use_dask=True):
    """
    Generates the data for creating an REV plot based on tortuosity

    Parameters
    ----------
    im : ndarray
        The binary image to analyze with ``True`` indicating phase of interest
    block_sizes : np.ndarray
        An array containing integers of block sizes to be calculated
    use_dask : bool
        A boolean determining the usage of `dask`.

    Returns
    -------
    df : DataFrame
        A `pandas` data frame with the properties for each block on a given row
    """
    if block_sizes is None:
        block_sizes = get_block_sizes(im.shape)
    block_sizes = np.array(block_sizes, dtype=int)
    tau = []
    for s in block_sizes:
        tau.append(analyze_blocks(im, block_size=s, use_dask=use_dask))
    df = pd.concat(tau)
    return df


def block_size_to_divs(shape, block_size):
    r"""
    Finds the number of blocks in each direction given the size of the blocks

    Parameters
    ----------
    shape : sequence of ints
        The [x, y, z] shape of the image
    block_size : int or sequence of ints
        The size of the blocks

    Returns
    -------
    divs : list of ints
        The number of blocks to divide the image into along each axis. The minimum
        number of blocks is 2.
    """
    shape = np.array(shape)
    divs = shape // np.array(block_size)
    # scraps = shape % np.array(block_size)
    divs = np.clip(divs, a_min=2, a_max=shape)
    return divs


def analyze_blocks(im, block_size, use_dask=True):
    r'''
    Computes structural and transport properties of each block

    Parameters
    ----------
    im : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest
    block_size : int
        The size of the blocks to use. Only cubic blocks are supported so an integer
        must be given, or an exception is raised. If the image is not evenly
        divisible by the given `block_size` any extra voxels are removed from the
        end of each axis before all processing occcurs.
    use_dask : bool
        A boolean determining the usage of `dask`.

    Returns
    -------
    df_out : DataFrame
        A `pandas` data frame with the properties for each block on a given row.
    '''

    if block_size is None:
        scale_factor = 3
        dt = edt(im)
        # TODO: Is the following supposed to be over 2 or over im.ndim?
        x = min(dt.max() * scale_factor, min(np.array(im.shape)/2))
        block_shape = np.ones(im.ndim) * x
    else:
        block_shape = np.int_(np.array(im.shape)//block_size)

    # determines block size, trimmed to fit in the image
    block_size = np.floor(im.shape/np.array(block_shape))

    results = []
    offset = int(block_size[0]/2)

    # create blocks and queues them for calculation
    for ax in range(im.ndim):

        # creates the masked images - removes half of a chunk from both ends of one axis
        tmp = np.swapaxes(im, 0, ax)
        tmp = tmp[offset:-offset, ...]
        tmp = np.swapaxes(tmp, 0, ax)
        slices = tools.subdivide(tmp, block_size=block_size, mode='whole')

        if use_dask:
            for s in slices:
                results.append(dask.delayed(calc_g)(tmp[s], axis=ax))

        # or do it the regular way
        else:
            for s in slices:
                results.append(calc_g(tmp[s], axis=ax))

    # collect all the results and calculate if needed
    results = np.asarray(dask.compute(results), dtype=object).flatten()

    # format results to be returned as a single dataframe
    df_out = pd.DataFrame()

    df_out['eps_orig'] = [r.original_porosity for r in results]
    df_out['eps_perc'] = [r.effective_porosity for r in results]
    df_out['g'] = [r.diffusive_conductance for r in results]
    df_out['tau'] = [r.tortuosity for r in results]
    df_out['volume'] = [r.volume for r in results]
    df_out['length'] = [block_size[0] for r in results]
    df_out['axis'] = [r.axis for r in results]
    df_out['time'] = [r.time for r in results]

    return df_out


def df_to_tortuosity(im, df):
    """
    Compute the tortuosity of a network populated with diffusive conductance values
    from the given dataframe.

    Parameters
    ----------
    df : dataframe
        The dataframe returned by the `blocks_to_dataframe` function
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    block_size : int
        The size of the blocks used to compute the conductance values in `df`

    Returns
    -------
    tau : list of floats
        The tortuosity in all three principal directions
    """

    block_size = df['length'][0]
    divs = block_size_to_divs(shape=im.shape, block_size=block_size)

    net = op.network.Cubic(shape=divs)
    air = op.phase.Phase(network=net)
    gx = df['g'][df['axis']==0]
    gy = df['g'][df['axis']==1]
    gz = df['g'][df['axis']==2]

    g = np.hstack([gz, gy, gx])

    air['throat.diffusive_conductance'] = g

    bcs = {0: {'in': 'left', 'out': 'right'},
           1: {'in': 'front', 'out': 'back'},
           2: {'in': 'top', 'out': 'bottom'}}

    e = np.sum(im, dtype=np.int64) / im.size
    D_AB = 1
    tau = []

    for ax in range(im.ndim):
        fick = op.algorithms.FickianDiffusion(network=net, phase=air)
        fick.set_value_BC(pores=net.pores(bcs[ax]['in']), values=1.0)
        fick.set_value_BC(pores=net.pores(bcs[ax]['out']), values=0.0)
        fick.run()
        rate_inlet = fick.rate(pores=net.pores(bcs[ax]['in']))[0]
        L = (divs[ax] - 1) * block_size
        A = (np.prod(divs) / divs[ax]) * (block_size**2)
        D_eff = rate_inlet * L / (A * (1 - 0))
        tau.append(e * D_AB / D_eff)

    ws = op.Workspace()
    ws.clear()
    return tau


def tortuosity_bt(im, block_size=None, use_dask=True):
    df = analyze_blocks(im, block_size, use_dask)
    tau = df_to_tortuosity(im, df)
    return tau


if __name__ =="__main__":
    import porespy as ps
    import numpy as np
    np.random.seed(1)

    im = ps.generators.blobs([100, 100, 100], blobiness=[1, 2, 1])

    r1 = tortuosity_bt(im, 50, use_dask=True)
    print(r1)
