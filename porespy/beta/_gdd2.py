import numpy as np
import openpnm as op
import pandas as pd
import logging
import dask
from dask.distributed import Client, LocalCluster
from edt import edt
from porespy.tools import Results, get_tqdm
from porespy import settings


__all__ = [
    'tortuosity_by_blocks',
    'rev_tortuosity',
]


tqdm = get_tqdm()
settings.loglevel = 50


def get_block_sizes(shape, block_size_range=[10, 100]):
    r"""
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
    """
    Lmin, Lmax = block_size_range
    a = np.ceil(min(shape)/Lmax).astype(int)
    block_sizes = min(shape) // np.arange(a, 9999)  # Generate WAY more than needed
    block_sizes = np.unique(block_sizes[block_sizes >= Lmin])
    return block_sizes


def rev_tortuosity(im, block_size_range=[10, 100]):
    r"""
    Computes data for a representative element volume plot based on tortuosity

    Parameters
    ----------
    im : ndarray
        A boolean image of the porous media with `True` values indicating the phase
        of interest.
    block_size_range : sequence of 2 ints
        The [lower, upper] range of the desired block sizes. Default is [10, 100]

    Returns
    -------
    df : DataFrame
        A `pandas` DataFrame with the tortuosity and volume for each block, along
        with other useful data like the porosity.
    """
    block_sizes = get_block_sizes(im.shape, block_size_range=[10, 100])
    tau = []
    for s in tqdm(block_sizes):
        tau.append(blocks_to_dataframe(im, block_size=s))
    df = pd.concat(tau)
    del df['Throat Number']
    df = df[df.Tortuosity < np.inf]  # inf values mean block did not percolate
    return df


@dask.delayed
def calc_g(im, axis):
    r"""
    Calculates diffusive conductance of an image

    Parameters
    ----------
    image : ndarray
        The binary image to analyze with ``True`` indicating phase of interest.
    axis : int
        0 for x-axis, 1 for y-axis, 2 for z-axis.
    """
    from porespy.simulations import tortuosity_fd
    try:
        solver = op.solvers.PyamgRugeStubenSolver(tol=1e-6)
        results = tortuosity_fd(im=im, axis=axis, solver=solver)
    except Exception:
        results = Results()
        results.effective_porosity = 0.0
        results.original_porosity = im.sum()/im.size
        results.tortuosity = np.inf
    L = im.shape[axis]
    A = np.prod(im.shape)/im.shape[axis]
    g = (results.effective_porosity * A) / (results.tortuosity * L)
    results.diffusive_conductance = g
    return results


def estimate_block_size(im, scale_factor=3, mode='radial'):
    r"""
    Predicts suitable block size based on properties of the image

    Parameters
    ----------
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    scale_factor : int
        The factor by which to increase the estimating block size to ensure blocks
        are big enough
    mode : str
        Which method to use when estimating the block size. Options are:

        -------- --------------------------------------------------------------------
        mode     description
        -------- --------------------------------------------------------------------
        radial   Uses the maximum of the distance transform
        linear   Uses chords drawn in all three directions, then uses the mean length
        -------- --------------------------------------------------------------------

    Returns
    -------
    block_size : list of ints
        The block size in a 3 directions. If `mode='radial'` these will all be equal.
        If `mode='linear'` they could differ in each direction.  The largest block
        size will be used if passed into another function.
    """
    if mode.startswith('rad'):
        dt = edt(im)
        x = min(dt.max()*scale_factor, min(im.shape)/2)  # Div by 2 is fewest blocks
        size = np.array([x, x, x], dtype=int)
    elif mode.startswith('lin'):
        size = []
        for ax in range(im.ndim):
            crds = ps.filters.apply_chords(im, axis=ax)
            crd_sizes = ps.filters.region_size(crds > 0)
            L = np.mean(crd_sizes[crds > 0])/2  # Divide by 2 to make it a radius
            L = min(L*scale_factor, im.shape[ax]/2)  # Div by 2 is fewest blocks
            size.append(L)
        size = np.array(size, dtype=int)
    else:
        raise Exception('Unsupported mode')
    return size


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
        The number of blocks to divide the image into along each axis
    """
    shape = np.array(shape)
    divs = (shape/np.array(block_size)).astype(int)
    return divs


def blocks_to_dataframe(im, block_size):
    r"""
    Computes the diffusive conductance for each block and returns values in a
    dataframe

    Parameters
    ----------
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    block_size : int or list of ints
        The size of the blocks to use. If a list is given, the largest value is used
        for all directions.

    Returns
    -------
    df : dataframe
        A `pandas` data frame with the properties for each block on a given row.
    """
    if dask.distributed.client._get_global_client() is None:
        client = Client(LocalCluster(silence_logs=logging.CRITICAL))
    else:
        client = dask.distributed.client._get_global_client()
    if not isinstance(block_size, int):  # divs must be equal, so use biggest block
        block_size = max(block_size)
    df = pd.DataFrame()
    offset = int(block_size/2)
    for axis in tqdm(range(im.ndim)):
        im_temp = np.swapaxes(im, 0, axis)
        im_temp = im_temp[offset:-offset, ...]
        divs = block_size_to_divs(shape=im_temp.shape, block_size=block_size)
        slices = ps.tools.subdivide(im_temp, divs)
        queue = []
        for s in slices:
            queue.append(calc_g(im_temp[s], axis=0))
        results = dask.compute(queue, scheduler=client)[0]
        df_temp = pd.DataFrame()
        df_temp['eps_orig'] = [r.original_porosity for r in results]
        df_temp['eps_perc'] = [r.effective_porosity for r in results]
        df_temp['g'] = [r.diffusive_conductance for r in results]
        df_temp['tau'] = [r.tortuosity for r in results]
        df_temp['axis'] = [axis for _ in results]
        df = pd.concat((df, df_temp))
    return df


def network_to_tau(df, im, block_size):
    r"""
    Compute the tortuosity of a network populated with diffusive conductance values
    from the given dataframe.

    Parameters
    ----------
    df : dataframe
        The dataframe returned by the `blocks_to_dataframe` function
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    block_size : int or list of ints
        The size of the blocks to use. If a list is given, the largest value is used
        for all directions.

    Returns
    -------
    tau : list of floats
        The tortuosity in all three principal directions
    """
    if not isinstance(block_size, int):  # divs must be equal, so use biggest block
        block_size = max(block_size)
    divs = block_size_to_divs(shape=im.shape, block_size=block_size)
    net = op.network.Cubic(shape=divs)
    phase = op.phase.Phase(network=net)
    gx = np.array(df['g'])[df['axis'] == 0]
    gy = np.array(df['g'])[df['axis'] == 1]
    gz = np.array(df['g'])[df['axis'] == 2]
    g = np.hstack((gz, gy, gx))  # throat indices in openpnm are in reverse order!
    if np.any(g == 0):
        g += 1e-20
    phase['throat.diffusive_conductance'] = g
    bcs = {0: {'in': 'left', 'out': 'right'},
           1: {'in': 'front', 'out': 'back'},
           2: {'in': 'top', 'out': 'bottom'}}
    im_temp = ps.filters.fill_blind_pores(im, surface=True)
    e = np.sum(im_temp, dtype=np.int64) / im_temp.size
    D_AB = 1
    tau = []
    for ax in range(im.ndim):
        fick = op.algorithms.FickianDiffusion(network=net, phase=phase)
        fick.set_value_BC(pores=net.pores(bcs[ax]['in']), values=1.0)
        fick.set_value_BC(pores=net.pores(bcs[ax]['out']), values=0.0)
        fick.run()
        rate_inlet = fick.rate(pores=net.pores(bcs[ax]['in']))[0]
        L = im.shape[ax] - block_size
        A = np.prod(im.shape) / im.shape[ax]
        D_eff = rate_inlet * L / (A * (1 - 0))
        tau.append(e * D_AB / D_eff)
    ws = op.Workspace()
    ws.clear()
    return tau


def tortuosity_by_blocks(im, block_size=None):
    r"""
    Computes the tortuosity tensor of an image using the geometric domain
    decomposition method

    Parameters
    ----------
    im : ndarray
        The boolean image of the materials with `True` indicating the void space
    block_size : int or list of ints
        The size of the blocks to use. If a list is given, the largest value is used
        for all directions.

    Returns
    -------
    tau : list of floats
        The tortuosity in all three principal directions
    """
    if block_size is None:
        block_size = estimate_block_size(im, scale_factor=3, mode='radial')
    df = blocks_to_dataframe(im, block_size)
    tau = network_to_tau(df=df, im=im, block_size=block_size)
    return tau


if __name__ =="__main__":
    import porespy as ps
    import numpy as np
    import matplotlib.pyplot as plt
    im = ps.generators.cylinders(shape=[200, 200, 200], porosity=0.5, r=3, seed=1)
    block_size = estimate_block_size(im, scale_factor=3, mode='linear')
    tau = tortuosity_by_blocks(im=im, block_size=block_size)
    print(tau)