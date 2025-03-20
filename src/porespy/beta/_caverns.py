import porespy as ps
import numpy as np
import matplotlib.pyplot as plt
from porespy.beta import analyze_blocks, df_to_tortuosity
from tqdm import tqdm
import dask
import pandas as pd
import time

plot = False

def plot_slices(im, N):
    step = N
    ims = []
    for i in range(int((im.shape[1]-1)/step)):
        ims.append(im[:, i*step:(i+1)*step])

    fig, axes = plt.subplots(1, len(ims))

    for i, (s, ax) in enumerate(zip(ims, axes)):
        ax.axis(False)
        ax.imshow(s)
    
ps.settings.loglevel=50

def fill_caverns(im, axis=0):
    inlets = ps.generators.faces(im.shape, inlet=axis)
    outlets = ps.generators.faces(im.shape, outlet=axis)
    im2 = ps.filters.trim_nonpercolating_paths(
        im=im, inlets=inlets, outlets=outlets)

    ramp = ps.generators.ramp(im2.shape, inlet=1, outlet=im.shape[axis], axis=axis)
    seq = (im2*ramp).astype(int)
    right_face = ps.generators.faces(im2.shape, outlet=1)
    trapped_1 = ps.filters.find_trapped_regions(
        seq=seq, outlets=right_face)

    ramp = ps.generators.ramp(im2.shape, inlet=im.shape[axis], outlet=1, axis=axis)
    seq = (im2*ramp).astype(int)
    left_face = ps.generators.faces(im2.shape, inlet=1)
    trapped_2 = ps.filters.find_trapped_regions(
        seq=seq, outlets=left_face)
    trapped = trapped_1 + trapped_2

    im3 = im2*~trapped
    return im3

def fill_all_caverns(im, axis=0):
    tmp = [im]
    im_c = im.copy()
    
    run = True
    while run:
        im_filled = fill_caverns(im=im_c, axis=axis)
        if np.sum(im_filled) == 0:
            break

        if (im_c==im_filled).all():
            run = False

        else:
            tmp.append(im_filled)
            im_c = im_filled.copy()
    
    return tmp

def plot_ims(im_list, figsize=[10,7]):
    fig, axes = plt.subplots(1, len(im_list), figsize=figsize)

    axes[0].imshow(im_list[0])
    axes[0].axis(False)

    for i, (ax, im, im_next) in enumerate(zip(axes[1:], im_list, im_list[1:])):
        # ax.imshow(im * ~im_next, alpha=0.5)
        ax.imshow(im)
        ax.imshow(im * np.array(~im!=im_next), alpha=0.7)
        # ax.imshow(im)
        ax.axis(False)

    return fig, axes

def tortuosity_ris(im, N):
    # Now do resistors in series
    step = N
    ims = []
    for i in range(int((im.shape[1]-1)/step)):
        ims.append(im[:, i*step:(i+1)*step])

    taus = []
    Deffs = []
    for i, s in enumerate(ims):
        taus.append(ps.simulations.tortuosity_fd(s, axis=1))
        Deffs.append(taus[i].effective_porosity/taus[i].tortuosity)
    Deff = (im.shape[1]-1)/np.sum(1/(np.array(Deffs)/step))
    return Deff

def porosity_map(im, block_size, dask_on=True):

    slices = ps.tools.subdivide(im, block_size=block_size)

    results = []
    for s in tqdm(slices):
        if dask_on:
            poro_obj = dask.delayed(ps.metrics.porosity)(im[s])
        
        else:
            poro_obj = ps.metrics.porosity(im[s])
        
        results.append(
            {'slice' : s,
             'eps_orig' : poro_obj.compute()}
        )

    df_out = pd.DataFrame()
    df_out['slice'] = [r['slice'] for r in results]
    df_out['eps_orig'] = [r['eps_orig'] for r in results]
    df_out['axis'] = [0 for r in results]

    return df_out

def tortuosity_map(im, block_size, dask_on=True):

    slices = ps.tools.subdivide(im, block_size=block_size)
    tmp = np.zeros(im.shape)

    results = []
    for s in tqdm(slices):
        for axis in range(im.ndim):
            if dask_on:
                tau_obj = dask.delayed(ps.beta.calc_g)(im[s], axis=axis)
            
            else:
                tau_obj = ps.beta.calc_g(im[s], axis=axis)

            tau_obj = tau_obj.compute()
            tau_obj.slice = s

            results.append(tau_obj)

    df_out = pd.DataFrame()

    df_out['eps_orig'] = [r.original_porosity for r in results]
    df_out['eps_perc'] = [r.effective_porosity for r in results]
    df_out['g'] = [r.diffusive_conductance for r in results]
    df_out['tau'] = [r.tortuosity for r in results]
    df_out['volume'] = [r.volume for r in results]
    df_out['length'] = [block_size for r in results]
    df_out['axis'] = [r.axis for r in results]
    df_out['time'] = [r.time for r in results]
    df_out['slice'] = [r.slice for r in results]

    return df_out

def plot_map(im, df, mode='porosity', cmap=None):

    if cmap == None:
        import matplotlib.cm as cm
        cmap = cm.get_cmap('rainbow')

    figs, axs = [], []
    for i, axis in enumerate(np.unique(df['axis'])):
        print(axis)
        tmp_df = df.loc[df['axis']==axis]

        tmp = np.zeros_like(im, np.float64)
        for s, v in zip(tmp_df['slice'], tmp_df[mode]):
            tmp[s] = v
        
        fig, ax = plt.subplots()
        plt.sca(ax)
        plt.imshow(im)
        plt.imshow(tmp, cmap=cmap, alpha=0.5)
        plt.colorbar(label=mode.upper())
        figs.append(fig)
        axs.append(ax)

    return figs, axs

def plot_df(im, df, x, y):
    figs, axes = [], []
    for i, axis in enumerate(np.unique(df['axis'])):

        f = df[y] != 0

        fig, ax = plt.subplots()
        plt.sca(ax)
        plt.plot(df[x][f], df[y][f], '.')
        plt.title(f"{y.capitalize()} vs {x.capitalize()} : Axis {i}")
        plt.xlabel(f"{x.capitalize()}")
        plt.ylabel(f"{y.capitalize()}")
        plt.show()
        
        figs.append(fig)
        axes.append(ax)
    
    return figs, axes

if __name__ == "__main__":
    im = ps.generators.blobs([1000] * 2, porosity=0.7, seed=1)
    # result = tortuosity_map(im, 50,)
    # plots = plot_map(im, result, mode='tau')

    im2 = fill_all_caverns(im)

    a = tortuosity_map(im, 100)
    b = tortuosity_map(im2[-1], 100)

    q = plot_df(im, a, 'eps_orig', 'g')
    f = b['g'] != 0
    plt.sca(q[1][0])
    plt.plot(b['eps_orig'][f], b['g'][f], 'r.', alpha=0.5)
    plt.show()

    # im2[-1]
    # plots = plot_ims(im2)
    # result2 = tortuosity_map(im2[-1], 250,)
    # plots2 = plot_map(im2[-1], result, mode='tau')

    # im2 = ps.generators.lattice_spheres([193, 193], r=10, spacing=32, offset=16)
    # im2 = ps.generators.overlapping_spheres([193, 193], r=10, porosity=0.7)
    # im = ps.generators.blobs([1281] * 2, porosity=0.6, blobiness=1.5, seed=2)
    # im = ps.generators.blobs([100] * 3, porosity=0.6, blobiness=1.5, seed=2)

    # all_ims = fill_all_caverns(im, 0)
    # t = plot_ims(all_ims)

    # for i, image in enumerate(all_ims):
    #     tau2 = ps.simulations.tortuosity_fd(image, axis=1)
    #     Deff2 = 1/tau2.formation_factor
    #     print(f"Effective diffusivity of image {i+1} is {Deff2}")

    # im2 = fill_caverns(im=im, axis=1)

    # t0 = time.perf_counter()
    # tau2 = ps.simulations.tortuosity_fd(im2, axis=1)
    # Deff2 = 1/tau2.formation_factor
    # t1 = time.perf_counter()
    # print(f"Effective diffusivity of original image is {Deff2}")

    # t2 = time.perf_counter()
    # tau3 = ps.simulations.tortuosity_fd(im2, axis=1)
    # Deff3 = 1/tau3.formation_factor
    # t3 = time.perf_counter()

    # print(f"Effective diffusivity of adjusted image is {Deff3}")

    # steps = [64 * i for i in range(1,7)]
    # steps = [64 * i for i in range(1,2)]

    # step_size = []
    # origin = []
    # adjusted = []
    # eta1 = []
    # eta2 = []

    # d = {}
    # d['original'] = Deff2
    # d['adjusted'] = Deff3

    # for step in steps:
    #     step_size.append(step)
    #     t_start = time.perf_counter()
    #     Deff4 = tortuosity_ris(im, N=step)
    #     t_mid = time.perf_counter()
    #     print(f"Effective diffusivity of original images in series with slices of {step} is {Deff4}")

    #     t_mid2 = time.perf_counter()
    #     Deff5 = tortuosity_ris(im2, N=step)
    #     print(f"Effective diffusivity of adjusted images in series with slices of {step} is {Deff5}")

    #     t_end = time.perf_counter()

    #     dt1 = t_mid-t_start
    #     dt2 = t_end-t_mid
    #     origin.append(Deff4)
    #     adjusted.append(Deff5)
    #     eta1.append(t_mid - t_start)
    #     eta2.append(t_end - t_mid2)

    #     # print(dt1)
    #     # print(dt2)

    # step_size.append(im.shape[0])
    # origin.append(Deff2)
    # adjusted.append(Deff3)
    # eta1.append(t1 - t0)
    # eta2.append(t3 - t2)

    # d = {
    #     'step_size' : step_size,
    #     'D_eff - Original' : origin,
    #     'D_eff - Adjusted' : adjusted,
    #     'Time - Original' : eta1,
    #     'Time - Adjusted' : eta2,
    # }
    # df = pd.DataFrame(d)

    # fig, ax = plt.subplots(1,2, figsize=[20,7])
    # ax[0].plot(np.log(step_size), df['D_eff - Original'], label='Original Image')
    # ax[0].plot(np.log(step_size), df['D_eff - Adjusted'], label='Adjusted Image')
    # ax[0].set_xlabel('$log(Step Size)$')
    # ax[0].set_ylabel('$D_{eff}$')
    # ax[0].set_title("$D_{eff}$ vs log(Step Size)")
    # # ax[0].axhline(list(df['D_eff - Adjusted'])[-1], xmin=0, xmax=im.shape[0], label="Adjusted $D_{eff}$", color='r')
    # ax[0].legend()

    # ax[1].plot(np.log(step_size), df['Time - Original'], label='Original Image')
    # ax[1].plot(np.log(step_size), df['Time - Adjusted'], label='Adjusted Image')
    # ax[1].set_xlabel('$log(Step Size)$')
    # ax[1].set_ylabel('Time (s)')
    # ax[1].set_title("Time vs log(Step Size)")
    # ax[1].legend()
