import time
from porespy import simulations, tools, settings
import numpy as np
import openpnm as op
from pandas import DataFrame
import dask.delayed
import dask
import edt

__all__ = ['tortuosity_gdd', 'chunks_to_dataframe']
settings.loglevel=50

@dask.delayed
def calc_g (image, axis, result=0):
    r'''Calculates diffusive conductance of an image.

    Parameters
    ----------
    image : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest.
    axis : int
        0 for x-axis, 1 for y-axis, 2 for z-axis.
    result: int
        0 for diffusive conductance, 1 for tortuosity.
    '''
    try:
        # if tortuosity_fd fails, then the throat is closed off from whichever axis was specified
        results = simulations.tortuosity_fd(im=image, axis=axis)

    except:
        return (99,0)
    
    A = np.prod(image.shape)/image.shape[axis]
    L = image.shape[axis]

    if result == 0:

        return( (results.effective_porosity * A) / (results.tortuosity * L))

    else:
        return((results.effective_porosity * A) / (results.tortuosity * L), results)
    
def network_calc (image, chunk_size, network, phase, bc, dimensions):
    r'''Calculates the resistor network tortuosity.
    
    Parameters
    ----------
    image : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest.
    chunk_size : np.ndarray
        Contains the size of a chunk in each direction.
    bc : tuple
        Contains the first and second boundary conditions.
    dimensions : tuple
        Contains the order of axes to calculate on.

    Returns
    -------
    tau : Tortuosity of the network in the given dimension
    '''
    fd=op.algorithms.FickianDiffusion(network=network, phase=phase)

    fd.set_value_BC(pores = network.pores(bc[0]), values = 1)
    fd.set_value_BC(pores = network.pores(bc[1]), values = 0)
    fd.run()

    rate_inlet = fd.rate(pores=network.pores(bc[0]))[0]
    L = image.shape[dimensions[0]] - chunk_size[dimensions[0]]
    A = image.shape[dimensions[1]] * image.shape[dimensions[2]]
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

    return slices

def tortuosity_gdd(im, scale_factor=3,):
    r'''Calculates the resistor network tortuosity.

    Parameters
    ----------
    im : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest

    chunk_shape : list
        Contains the number of chunks to be made in the x,y,z directions.

    subdivide : bool
        Enables the usage of built-in Porespy subdivide function or manually written function for chunking an image
    
    Returns
    -------
    results : list
        Contains tau values for three directions, time stamps, and all tau values for each chunk
    '''
    t0 = time.perf_counter()
    dt = edt.edt(im)
    print(f'Max distance transform found: {dt.max()}')

    # determining the number of chunks in each direction, minimum of 3 is required
    if np.all(im.shape//(scale_factor*dt.max())>np.array([3,3,3])):

        # if the minimum is exceeded, then chunk number is validated
        # integer division is required for defining chunk shapes
        chunk_shape=np.array(im.shape//(dt.max()*scale_factor), dtype=int)
        print(f"{chunk_shape} > [3,3,3], using {(im.shape//chunk_shape)} as chunk size.")
    
    # otherwise, the minimum of 3 in all directions is used
    else:
        chunk_shape=np.array([3,3,3])
        print(f"{np.array(im.shape//(dt.max()*scale_factor), dtype=int)} <= [3,3,3], using {im.shape[0]//3} as chunk size.")

    t1 = time.perf_counter() - t0

    # determines chunk size
    chunk_size = np.floor(im.shape/np.array(chunk_shape))

    # creates the masked images - removes half of a chunk from both ends of one axis
    x_image = im[ int(chunk_size[0]//2): int(im.shape[0] - chunk_size[0] //2), :, :]
    y_image = im[ :, int(chunk_size[1]//2): int(im.shape[1] - chunk_size[1] //2), :]
    z_image = im[ :, :, int(chunk_size[2]//2): int(im.shape[2] - chunk_size[2] //2)]

    t2 = time.perf_counter()- t0

    # creates the chunks for each masked image
    x_slices = chunking(spacing = chunk_size, divs = [chunk_shape[0]-1, chunk_shape[1], chunk_shape[2]])
    y_slices = chunking(spacing = chunk_size, divs = [chunk_shape[0], chunk_shape[1]-1, chunk_shape[2]])
    z_slices = chunking(spacing = chunk_size, divs = [chunk_shape[0], chunk_shape[1], chunk_shape[2]-1])

    t3 = time.perf_counter()- t0
    # queues up dask delayed function to be run in parallel

    x_gD = [calc_g(x_image[x_slice[0][0]:x_slice[0][1], x_slice[1][0]:x_slice[1][1], x_slice[2][0]:x_slice[2][1],], axis = 0, result = 1) for x_slice in x_slices]
    y_gD = [calc_g(y_image[y_slice[0][0]:y_slice[0][1], y_slice[1][0]:y_slice[1][1], y_slice[2][0]:y_slice[2][1],], axis = 0, result = 1) for y_slice in y_slices]
    z_gD = [calc_g(z_image[z_slice[0][0]:z_slice[0][1], z_slice[1][0]:z_slice[1][1], z_slice[2][0]:z_slice[2][1],], axis = 0, result = 1) for z_slice in z_slices]

    # order of throat creation
    all_values = [z_gD, y_gD, x_gD]

    all_results = np.array(dask.compute(all_values), dtype=object).flatten()
    
    all_gD = [result for result in all_results[::2]]
    all_tau_unfiltered = [result for result in all_results[1::2]]
    all_tau = [result.tortuosity if type(result)!=int else result for result in all_tau_unfiltered]
    t4 = time.perf_counter()- t0

    # creates opnepnm network to calculate image tortuosity - order of arrays are incorrect
    net = op.network.Cubic(chunk_shape)
    air = op.phase.Phase(network = net)

    air['throat.diffusive_conductance']=np.array(all_gD).flatten()

    # calculates throat tau in x, y, z directions
    throat_tau = [
    # x direction
    network_calc(image = im,
            chunk_size = chunk_size,
            network = net,
            phase=air,
            bc = ['left', 'right'],
            dimensions = [1,0,2]
    ),

    # y direction
    network_calc(image = im,
            chunk_size = chunk_size,
            network = net,
            phase=air,
            bc = ['front', 'back'],
            dimensions = [2,1,0]
    ),

    # z direction
    network_calc(image = im,
            chunk_size = chunk_size,
            network = net,
            phase=air,
            bc = ['top', 'bottom'],
            dimensions = [0,1,2]
    )
    ]

    t5 = time.perf_counter()- t0

    return [throat_tau[0], throat_tau[1], throat_tau[2], t1, t2, t3, t4, t5, all_tau]

def chunks_to_dataframe(im, scale_factor=3,):
    r'''Calculates the resistor network tortuosity.

    Parameters
    ----------
    im : np.ndarray
        The binary image to analyze with ``True`` indicating phase of interest

    chunk_shape : list
        Contains the number of chunks to be made in the x,y,z directions.

    subdivide : bool
        Enables the usage of built-in Porespy subdivide function or manually written function for chunking an image

    Returns
    -------
    df : pandas.DataFrame
        Contains throat numbers, tau values, diffusive conductance values, and porosity
    
    '''
    t0 = time.perf_counter()
    dt = edt.edt(im)
    print(f'Max distance transform found: {dt.max()}')

    # determining the number of chunks in each direction, minimum of 3 is required
    if np.all(im.shape//(scale_factor*dt.max())>np.array([3,3,3])):

        # if the minimum is exceeded, then chunk number is validated
        # integer division is required for defining chunk shapes
        chunk_shape=np.array(im.shape//(dt.max()*scale_factor), dtype=int)
        print(f"{chunk_shape} > [3,3,3], using {(im.shape//chunk_shape)} as chunk size.")
    
    # otherwise, the minimum of 3 in all directions is used
    else:
        chunk_shape=np.array([3,3,3])
        print(f"{np.array(im.shape//(dt.max()*scale_factor), dtype=int)} <= [3,3,3], using {im.shape[0]//3} as chunk size.")

    t1 = time.perf_counter() - t0

    # determines chunk size
    chunk_size = np.floor(im.shape/np.array(chunk_shape))

    # creates the masked images - removes half of a chunk from both ends of one axis
    x_image = im[ int(chunk_size[0]//2): int(im.shape[0] - chunk_size[0] //2), :, :]
    y_image = im[ :, int(chunk_size[1]//2): int(im.shape[1] - chunk_size[1] //2), :]
    z_image = im[ :, :, int(chunk_size[2]//2): int(im.shape[2] - chunk_size[2] //2)]

    t2 = time.perf_counter()- t0

    # creates the chunks for each masked image
    x_slices = chunking(spacing = chunk_size, divs = [chunk_shape[0]-1, chunk_shape[1], chunk_shape[2]])
    y_slices = chunking(spacing = chunk_size, divs = [chunk_shape[0], chunk_shape[1]-1, chunk_shape[2]])
    z_slices = chunking(spacing = chunk_size, divs = [chunk_shape[0], chunk_shape[1], chunk_shape[2]-1])

    t3 = time.perf_counter()- t0
    # queues up dask delayed function to be run in parallel

    x_gD = [calc_g(x_image[x_slice[0][0]:x_slice[0][1], x_slice[1][0]:x_slice[1][1], x_slice[2][0]:x_slice[2][1],], axis = 0, result = 1) for x_slice in x_slices]
    y_gD = [calc_g(y_image[y_slice[0][0]:y_slice[0][1], y_slice[1][0]:y_slice[1][1], y_slice[2][0]:y_slice[2][1],], axis = 0, result = 1) for y_slice in y_slices]
    z_gD = [calc_g(z_image[z_slice[0][0]:z_slice[0][1], z_slice[1][0]:z_slice[1][1], z_slice[2][0]:z_slice[2][1],], axis = 0, result = 1) for z_slice in z_slices]

    # order of throat creation
    all_values = [z_gD, y_gD, x_gD]

    all_results = np.array(dask.compute(all_values), dtype=object).flatten()
    
    all_gD = [result for result in all_results[::2]]
    all_tau_unfiltered = [result for result in all_results[1::2]]
    all_porosity = [result.effective_porosity if type(result)!=int else result for result in all_tau_unfiltered]
    all_tau = [result.tortuosity if type(result)!=int else result for result in all_tau_unfiltered]
    t4 = time.perf_counter()- t0

    # creates opnepnm network to calculate image tortuosity - order of arrays are incorrect
    net = op.network.Cubic(chunk_shape)

    t5 = time.perf_counter()- t0

    df = DataFrame(list(zip(np.arange(net.Nt), all_tau, all_gD, all_porosity)), 
                        columns = ['Throat Number', 'Tortuosity', 'Diffusive Conductance', 'Porosity'])

    return df

if __name__ =="__main__":
    import porespy as ps
    im = ps.generators.fractal_noise(shape=[100,100,100], seed=1)<0.65
    res = ps.simulations.tortuosity_gdd(im=im, scale_factor=3)
    print(res)

    im = ps.generators.fractal_noise(shape=[100,100,100], seed=2)<0.65
    df = ps.simulations.chunks_to_dataframe(im=im, scale_factor=3)
    print(df)
