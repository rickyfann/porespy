import time
from porespy import simulations, tools, settings
from porespy.tools import Results
import numpy as np
import openpnm as op
from pandas import DataFrame
import dask.delayed
import dask
import edt

__all__ = ['tortuosity_gdd',]
settings.loglevel=50


@dask.delayed
def calc_g(image, axis):
    r'''Calculates diffusive conductance of an image.

    Parameters
    ----------
    image : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest.
    axis : int
        0 for x-axis, 1 for y-axis, 2 for z-axis.
    result: int
        0 for diffusive conductance, 1 for both diffusive conductance
        and results object from Porespy.
    '''
    try:
        # if tortuosity_fd fails, throat is closed off from whichever axis was specified
        results = simulations.tortuosity_fd(im=image, axis=axis)

    except Exception:
        # a is diffusive conductance, b is tortuosity
        a, b = (0, np.inf)

        return (a, b)

    L = image.shape[axis]
    A = np.prod(image.shape)/image.shape[axis]

    return ((results.effective_porosity * A) / (results.tortuosity * L), results.tortuosity)


def network_calc(image, block_size, network, phase, bc, axis):
    r'''Calculates the resistor network tortuosity.

    Parameters
    ----------
    image : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest.
    block_size : np.ndarray
        Contains the size of a chunk in each direction.
    bc : tuple
        Contains the first and second boundary conditions.
    axis : int
        The axis to calculate on.

    Returns
    -------
    tau : Tortuosity of the network in the given dimension
    '''
    fd=op.algorithms.FickianDiffusion(network=network, phase=phase)

    fd.set_value_BC(pores=network.pores(bc[0]), values=1)
    fd.set_value_BC(pores=network.pores(bc[1]), values=0)
    fd.run()

    rate_inlet = fd.rate(pores=network.pores(bc[0]))[0]
    L = image.shape[axis] - block_size[axis]
    A = np.prod(image.shape) / image.shape[axis]
    d_eff = rate_inlet * L / (A * (1 - 0))

    e = image.sum() / image.size
    D_AB = 1
    tau = e * D_AB / d_eff

    return tau


def chunking(spacing, divs):
    r'''Returns slices given the number of chunks and chunk sizes.

    Parameters
    ----------
    spacing : float
        Size of each chunk.
    divs : list
        Number of chunks in each direction.

    Returns
    -------
    slices : list
        Contains lists of image slices corresponding to chunks
    '''

    slices = [[
    (int(i*spacing[0]), int((i+1)*spacing[0])),
    (int(j*spacing[1]), int((j+1)*spacing[1])),
    (int(k*spacing[2]), int((k+1)*spacing[2]))]
    for i in range(divs[0])
    for j in range(divs[1])
    for k in range(divs[2])]

    return np.array(slices, dtype=int)

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


def analyze_blocks(im, block_size , use_dask=True):
    r'''Calculates the resistor network tortuosity.

    Parameters
    ----------
    im : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest

    block_shape : list
        Contains the number of chunks to be made in the x,y,z directions.

    Returns
    -------
    results : list
        Contains tau values for three directions, time stamps, tau values for each chunk
    '''
    t0 = time.perf_counter()

    if not block_size:
        scale_factor = 3
        dt = edt.edt(im)
        x = min(dt.max() * scale_factor, min(np.array(im.shape)/2))

        block_shape = np.array([x, x, x], dtype=int)
    
    else:
        block_shape = np.int_(np.array(im.shape)//block_size)
    
    t1 = time.perf_counter() - t0

    # determines chunk size
    block_size = np.floor(im.shape/np.array(block_shape))
    print(fr"Using block shape of {block_shape} and blocking of {block_size}")

    # creates the masked images - removes half of a chunk from both ends of one axis
    x_image = im[int(block_size[0]//2): int(im.shape[0] - block_size[0] //2), :, :]
    y_image = im[:, int(block_size[1]//2): int(im.shape[1] - block_size[1] //2), :]
    z_image = im[:, :, int(block_size[2]//2): int(im.shape[2] - block_size[2] //2)]

    t2 = time.perf_counter()- t0

    # creates the chunks for each masked image
    x_slices = chunking(spacing=block_size,
                        divs=[block_shape[0]-1, block_shape[1], block_shape[2]])
    y_slices = chunking(spacing=block_size,
                        divs=[block_shape[0], block_shape[1]-1, block_shape[2]])
    z_slices = chunking(spacing=block_size,
                        divs=[block_shape[0], block_shape[1], block_shape[2]-1])

    t3 = time.perf_counter()- t0
    # queues up dask delayed function to be run in parallel

    x_gD = [calc_g(x_image[x_slice[0, 0]:x_slice[0, 1],
                           x_slice[1, 0]:x_slice[1, 1],
                           x_slice[2, 0]:x_slice[2, 1],],
                           axis=0) for x_slice in x_slices]

    y_gD = [calc_g(y_image[y_slice[0, 0]:y_slice[0, 1],
                           y_slice[1, 0]:y_slice[1, 1],
                           y_slice[2, 0]:y_slice[2, 1],],
                           axis=1) for y_slice in y_slices]

    z_gD = [calc_g(z_image[z_slice[0, 0]:z_slice[0, 1],
                           z_slice[1, 0]:z_slice[1, 1],
                           z_slice[2, 0]:z_slice[2, 1],],
                           axis=2) for z_slice in z_slices]

    # order of throat creation
    all_values = [z_gD, y_gD, x_gD]

    if use_dask:
        all_results = np.array(dask.compute(all_values), dtype=object).flatten()

    else:
        all_values = np.array(all_values).flatten()
        all_results = []
        for item in all_values:
            all_results.append(item.compute())

        all_results = np.array(all_results).flatten()

    all_gD = [result for result in all_results[::2]]
    all_tau = [result for result in all_results[1::2]]

    # all_tau = [result.tortuosity if type(result)!=int
    #            else result for result in all_tau_unfiltered]

    t4 = time.perf_counter()- t0

    output = Results()
    output.__setitem__('length', np.ones(len(all_gD)) * block_size[0])
    output.__setitem__('time_stamps', [t1, t2, t3, t4])
    output.__setitem__('g', all_gD)
    output.__setitem__('tau', all_tau)

    return output

def df_to_tau(im, df):
    block_size = df['length'][0]
    divs = block_size_to_divs(shape=im.shape, block_size=block_size)
    
    net = op.network.Cubic(shape=divs)
    air = op.phase.Phase(network=net)

    air['throat.diffusive_conductance'] = df['g']

    # calculates throat tau in x, y, z directions
    throat_tau = [
    # x direction
    network_calc(image=im,
                 block_size=df['length'],
                 network=net,
                 phase=air,
                 bc=['left', 'right'],
                 axis=1),

    # y direction
    network_calc(image=im,
                 block_size=df['length'],
                 network=net,
                 phase=air,
                 bc=['front', 'back'],
                 axis=2),

    # z direction
    network_calc(image=im,
                 block_size=df['length'],
                 network=net,
                 phase=air,
                 bc=['top', 'bottom'],
                 axis=0)]

    return throat_tau
    
def tortuosity_gdd(im, block_size=None):
    df = analyze_blocks(im, block_size)
    tau = df_to_tau(im, df)
    return tau

if __name__ =="__main__":
    import porespy as ps
    import numpy as np
    np.random.seed(1)
    im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.7)
    res = tortuosity_gdd(im=im, scale_factor=3, use_dask=True)
    print(res)

    # np.random.seed(2)
    # im = ps.generators.blobs(shape=[100, 100, 100], porosity=0.7)
    # df = ps.simulations.chunks_to_dataframe(im=im, scale_factor=3)
    # print(df)
