import itertools
import numpy as np
from numba import njit, prange
import scipy.ndimage as spim
from skimage.morphology import disk, ball
import pyedt
from porespy.tools import extend_slice
from porespy import settings
from porespy.tools import (
    get_tqdm, 
    make_contiguous, 
    marching_cubes_area_and_volume, 
    create_mc_template_list,
)
from porespy.metrics import region_surface_areas, region_interface_areas
from porespy.metrics import region_volumes
from loguru import logger

tqdm = get_tqdm()

@njit
def decompose_summation(n):
    reduced_n = n
    values = np.zeros(6, dtype=np.uint32)
    for i in range(5, -1, -1):
        if reduced_n >= 2 ** i:
            reduced_n -= 2 ** i
            values[i] = 1
    return values

@njit
def calculate_kernels(voxel_size):
    voxel_size = np.array(voxel_size)
    areas = np.zeros(64, dtype = np.float32)
    face_areas = [
        voxel_size[1] * voxel_size[2],
        voxel_size[0] * voxel_size[2],
        voxel_size[0] * voxel_size[1],
    ]
    kernel = np.zeros(
        (3, 3, 3,),
        dtype = np.uint32,
        )

    for i, x, y, z in (
        (0, 0, 1, 1),
        (1, 2, 1, 1),
        (2, 1, 0, 1),
        (3, 1, 2, 1),
        (4, 1, 1, 0),
        (5, 1, 1, 2),
    ):
        kernel[x, y, z] = 2 ** i

    for i in range(1, 64):
        indexes = decompose_summation(i)
        for j in range(6):
            if indexes[j] == 1:
                areas[i] += face_areas[j//2]

    return kernel, areas


def calculate_throat_geom_properties(vx, sub_dt, voxel_size):
    # Directions used to evaluate geom properties
    dirs = [
        (1, 0, 0),
        (0, 1, 0),
        (0, 0, 1),
        (-1, 0, 0),
        (0, -1, 0),
        (0, 0, -1),
    ]

    voxel_areas = np.array([
        voxel_size[1] * voxel_size[2],
        voxel_size[2] * voxel_size[0],
        voxel_size[0] * voxel_size[1],
    ])

    dx = voxel_size[0]*(max(vx[0]) - min(vx[0]))
    dy = voxel_size[1]*(max(vx[1]) - min(vx[1]))
    dz = voxel_size[2]*(max(vx[2]) - min(vx[2]))
    normal = np.array([dx*dy, dy*dz, dz*dx], dtype=np.float64)
    norm = np.linalg.norm(normal)

    if norm != 0:
        normal /= norm
        t_perimeter_loc = np.sum(sub_dt[vx] < 2*max(voxel_size)) * np.linalg.norm(normal * voxel_size)
        if t_perimeter_loc < 4. * np.linalg.norm(normal * voxel_size):
            t_perimeter_loc = 4. * np.linalg.norm(normal * voxel_size)
        t_area_loc = len(vx[0])*np.sum(normal * voxel_areas)
    else:                        # cases where the throat cross section is aligned in a line of voxels
        t_perimeter_loc = np.sum(sub_dt[vx] < 2*voxel_size[0]) * voxel_size[0]
        if t_perimeter_loc < 4. * voxel_size[0]:
            t_perimeter_loc = 4. * voxel_size[0]
        t_area_loc = len(vx[0])*voxel_areas[0]

    return t_perimeter_loc, t_area_loc

def regions_to_network(
    regions, phases=None, voxel_size=(1, 1, 1), accuracy='standard', porosity_map=None
):
    r"""
    Analyzes an image that has been partitioned into pore regions and extracts
    the pore and throat geometry as well as network connectivity.

    Parameters
    ----------
    regions : ndarray
        An image of the material partitioned into individual regions.
        Zeros in this image are ignored.
    phases : ndarray, optional
        An image indicating to which phase each voxel belongs. The returned
        network contains a 'pore.phase' array with the corresponding value.
        If not given a value of 1 is assigned to every pore.
    voxel_size : tuple (default = (1, 1, 1))
        The resolution of the image, expressed as the length of the sides of a
        voxel, so the volume of a voxel would be the product of **voxel_size** coords.
    accuracy : string
        Controls how accurately certain properties are calculated. Options are:

        'standard' (default)
            Computes the surface areas and perimeters by simply counting
            voxels.  This is *much* faster but does not properly account
            for the rough, voxelated nature of the surfaces.
        'high'
            Computes surface areas using the marching cube method, and
            perimeters using the fast marching method.  These are substantially
            slower but better account for the voxelated nature of the images.

    Returns
    -------
    net : dict
        A dictionary containing all the pore and throat size data, as well as
        the network topological information.  The dictionary names use the
        OpenPNM convention (i.e. 'pore.coords', 'throat.conns').

    Notes
    -----
    The meaning of each of the values returned in ``net`` are outlined below:

    'pore.region_label'
        The region labels corresponding to the watershed extraction. The
        pore indices and regions labels will be offset by 1, so pore 0
        will be region 1.
    'throat.conns'
        An *Nt-by-2* array indicating which pores are connected to each other
    'pore.region_label'
        Mapping of regions in the watershed segmentation to pores in the
        network
    'pore.local_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the pore region in isolation
    'pore.global_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the full image
    'pore.geometric_centroid'
        The center of mass of the pore region as calculated by
        ``skimage.measure.center_of_mass``
    'throat.global_peak'
        The coordinates of the location of the maxima of the distance transform
        performed on the full image
    'pore.region_volume'
        The volume of the pore region computed by summing the voxels
    'pore.volume'
        The volume of the pore found by as volume of a mesh obtained from the
        marching cubes algorithm
    'pore.surface_area'
        The surface area of the pore region as calculated by either counting
        voxels or using the fast marching method to generate a tetramesh (if
        ``accuracy`` is set to ``'high'``.)
    'throat.cross_sectional_area'
        The cross-sectional area of the throat found by either counting
        voxels or using the fast marching method to generate a tetramesh (if
        ``accuracy`` is set to ``'high'``.)
    'throat.perimeter'
        The perimeter of the throat found by counting voxels on the edge of
        the region defined by the intersection of two regions.
    'pore.inscribed_diameter'
        The diameter of the largest sphere inscribed in the pore region. This
        is found as the maximum of the distance transform on the region in
        isolation.
    'pore.extended_diameter'
        The diamter of the largest sphere inscribed in overal image, which
        can extend outside the pore region. This is found as the local maximum
        of the distance transform on the full image.
    'throat.inscribed_diameter'
        The diameter of the largest sphere inscribed in the throat.  This
        is found as the local maximum of the distance transform in the area
        where to regions meet.
    'throat.total_length'
        The length between pore centered via the throat center.
    'throat.direct_length'
        The length between two pore centers on a straight line between them
        that does not pass through the throat centroid.
    'pore.phase'
        Highest phase label in the pore.

    """
    logger.trace('Extracting pore/throat information')
    template_areas, template_volumes = create_mc_template_list(spacing=voxel_size)

    im = make_contiguous(regions)
    struc_elem = disk if im.ndim == 2 else ball
    voxel_size = tuple([float(i) for i in voxel_size])
    if phases is None:
        phases = (im > 0).astype(int)
    if im.size != phases.size:
        raise Exception('regions and phase are different sizes, probably ' +
                        'because boundary regions were not added to phases')
    dt = np.sqrt(pyedt.edt(phases >= 1, scale=voxel_size))

    # Get 'slices' into im for each pore region
    slices = spim.find_objects(im)

    # Initialize arrays
    Ps = np.arange(1, np.amax(im)+1)
    Np = np.size(Ps)
    p_coords_cm = np.zeros((Np, im.ndim), dtype=float)
    p_coords_dt = np.zeros((Np, im.ndim), dtype=float)
    p_coords_dt_global = np.zeros((Np, im.ndim), dtype=float)
    p_volume = np.zeros((Np, ), dtype=float)
    p_dia_local = np.zeros((Np, ), dtype=float)
    p_dia_global = np.zeros((Np, ), dtype=float)
    p_label = np.zeros((Np, ), dtype=int)
    p_area_surf = np.zeros((Np, ), dtype=float)
    p_phase = np.zeros((Np, ), dtype=int)
    p_porosity = np.ones((Np, ), dtype=float)
    # The number of throats is not known at the start, so lists are used
    # which can be dynamically resized more easily.
    t_conns = []
    t_dia_inscribed = []
    t_area = []
    t_perimeter = []
    t_coords = []

    index_kernel, areas = calculate_kernels(voxel_size)

    @njit
    def calculate_pore_area(im, pore):

        p_area_surf = np.zeros(1, dtype=np.float32)

        for i in range(1, im.shape[0] - 1):
            for j in range(1, im.shape[1] - 1):
                for k in range(1, im.shape[2] - 1):
                    pore_label = pore + 1
                    if pore_label != im[i,j,k]:
                        continue

                    kern_im = (im[i-1:i+2,j-1:j+2,k-1:k+2] != pore_label) * index_kernel

                    index = np.sum(kern_im)
                    p_area_surf += areas[index]

        return p_area_surf

    # Start extracting size information for pores and throats
    msg = "Extracting pore and throat properties"
    for i in tqdm(Ps, desc=msg, **settings.tqdm):
        pore = i - 1
        if slices[pore] is None:
            continue
        s = extend_slice(slices[pore], im.shape)
        sub_im = im[s]
        sub_dt = dt[s]
        pore_im = sub_im == i
        padded_mask = np.pad(pore_im, pad_width=1, mode='constant')
        pore_dt = np.sqrt(pyedt.edt(padded_mask, scale=voxel_size, force_method="cpu"))
        s_offset = np.array([i.start for i in s])
        p_label[pore] = i
        p_coords_cm[pore, :] = (spim.center_of_mass(pore_im) + s_offset) * voxel_size
        temp = np.vstack(np.where(pore_dt == pore_dt.max()))[:, 0]
        p_coords_dt[pore, :] = (temp + s_offset) * voxel_size
        p_phase[pore] = (phases[s]*pore_im).max()
        if porosity_map is not None:
            p_porosity[pore] = ((porosity_map[s]*pore_im).sum() / pore_im.sum()) / 100
        else:
            p_porosity[pore] = 1
        p_area_surf[pore], p_volume[pore] = marching_cubes_area_and_volume(
            sub_im, 
            target_label = i, 
            template_areas=template_areas,
            template_volumes=template_volumes,
        )
        temp = np.vstack(np.where(sub_dt == sub_dt.max()))[:, 0]
        p_coords_dt_global[pore, :] = (temp + s_offset) * voxel_size
        p_dia_local[pore] = 2*np.amax(pore_dt)
        p_dia_global[pore] = 2*np.amax(sub_dt)
        im_w_throats = spim.binary_dilation(input=pore_im, structure=struc_elem(1))
        im_w_throats = im_w_throats*sub_im
        Pn = np.unique(im_w_throats)[1:] - 1
        for j in Pn:
            if j > pore:
                t_conns.append([pore, j])
                vx = np.where(im_w_throats == (j + 1))
                t_dia_inscribed.append(2*np.amax(sub_dt[vx]))
                # The following is overwritten if accuracy is set to 'high'
                t_perimeter_loc, t_area_loc = calculate_throat_geom_properties(vx, sub_dt, voxel_size)
                t_perimeter.append(t_perimeter_loc)
                #t_area.append(t_area_surf[j, pore])
                t_area.append(1)
                #t_area.append(t_area_loc)
                #p_area_surf[pore] -= t_area_loc
                t_inds = tuple([i+j for i, j in zip(vx, s_offset)])
                temp = np.where(dt[t_inds] == np.amax(dt[t_inds]))[0][0]
                t_coords.append(tuple([t_inds[k][temp]*voxel_size[k] for k in range(im.ndim)]))

    # Clean up values
    p_coords = p_coords_cm
    Nt = len(t_dia_inscribed)  # Get number of throats
    if im.ndim == 2:  # If 2D, add 0's in 3rd dimension
        p_coords = np.vstack((p_coords_cm.T, np.zeros((Np, )))).T
        t_coords = np.vstack((np.array(t_coords).T, np.zeros((Nt, )))).T

    if len (t_conns) == 0:
        return None

    net = {}
    ND = im.ndim
    # Define all the fundamental stuff
    net['throat.conns'] = np.array(t_conns)
    net['pore.coords'] = np.array(p_coords)
    net['pore.all'] = np.ones_like(net['pore.coords'][:, 0], dtype=bool)
    net['throat.all'] = np.ones_like(net['throat.conns'][:, 0], dtype=bool)
    net['pore.region_label'] = np.array(p_label)
    net['pore.phase'] = np.array(p_phase, dtype=int)
    net['pore.subresolution_porosity'] = p_porosity
    net['throat.phases'] = net['pore.phase'][net['throat.conns']]
    V = np.copy(p_volume)
    net['pore.region_volume'] = V  # This will be an area if image is 2D
    f = 3/4 if ND == 3 else 1.0
    net['pore.equivalent_diameter'] = 2*(V/np.pi * f)**(1/ND)
    # Extract the geometric stuff
    net['pore.local_peak'] = np.copy(p_coords_dt)
    net['pore.global_peak'] = np.copy(p_coords_dt_global)
    net['pore.geometric_centroid'] = np.copy(p_coords_cm)
    net['throat.global_peak'] = np.array(t_coords)
    net['pore.inscribed_diameter'] = np.copy(p_dia_local)
    net['pore.extended_diameter'] = np.copy(p_dia_global)
    net['throat.inscribed_diameter'] = np.array(t_dia_inscribed)
    P12 = net['throat.conns']
    PT1 = np.sqrt(np.sum((p_coords[P12[:, 0]]-t_coords)**2,
                         axis=1))
    PT2 = np.sqrt(np.sum((p_coords[P12[:, 1]]-t_coords)**2,
                         axis=1))
    net['throat.total_length'] = PT1 + PT2
    PT1 = PT1-p_dia_local[P12[:, 0]]/2
    PT2 = PT2-p_dia_local[P12[:, 1]]/2
    dist = (p_coords[P12[:, 0]] - p_coords[P12[:, 1]])
    net['throat.direct_length'] = np.sqrt(np.sum(dist**2, axis=1))
    net['throat.perimeter'] = np.array(t_perimeter)

    net['pore.volume'] = p_volume
    net['pore.surface_area'] = p_area_surf
    interface_area = region_interface_areas(regions=im, areas=p_area_surf, voxel_size=voxel_size)
    A = interface_area.area
    net['throat.cross_sectional_area'] = A
    net['throat.equivalent_diameter'] = (4*A/np.pi)**(1/2)


    return net
