# import pytest
import numpy as np
from edt import edt
import porespy as ps
import scipy.ndimage as spim
from skimage.morphology import disk, ball, skeletonize_3d
from skimage.util import random_noise
from scipy.stats import norm
ps.settings.tqdm['disable'] = True


class SimulationsTest():
    def setup_class(self):
        np.random.seed(0)
        self.im = ps.generators.blobs(shape=[100, 100, 100], blobiness=2)
        # Ensure that im was generated as expeccted
        assert ps.metrics.porosity(self.im) == 0.499829
        self.im_dt = edt(self.im)

    def test_drainage_with_gravity(self):
        np.random.seed(2)
        im = ps.generators.blobs(shape=[100, 100], porosity=0.7)
        dt = edt(im)
        pc = -2*0.072*np.cos(np.deg2rad(180))/dt
        np.testing.assert_approx_equal(pc[im].max(), 0.144)
        drn = ps.simulations.drainage(pc=pc, im=im, voxel_size=1e-5, g=9.81)
        np.testing.assert_approx_equal(drn.im_pc.max(), np.inf)
        drn2 = ps.simulations.drainage(pc=pc, im=im, voxel_size=1e-4, g=0)
        np.testing.assert_approx_equal(drn2.im_pc[im].max(), np.inf)
        im = ps.filters.fill_blind_pores(im)
        drn = ps.simulations.drainage(pc=pc, im=im, voxel_size=1e-5, g=9.81)
        np.testing.assert_approx_equal(drn.im_pc.max(), 10.04657972914)
        drn2 = ps.simulations.drainage(pc=pc, im=im, voxel_size=1e-4, g=0)
        np.testing.assert_approx_equal(drn2.im_pc[im].max(), 0.14622522289864)

    def test_gdd(self):
        im = ps.generators.fractal_noise(shape=[100, 100, 100], seed=1)<0.65
        res = ps.simulations.tortuosity_gdd(im=im, scale_factor=3)
        np.testing.assert_approx_equal(res[0], 1.707800753372152)
        np.testing.assert_approx_equal(res[1], 1.7033469726308779)
        np.testing.assert_approx_equal(res[2], 1.5911705426959204)

    def test_gdd_dataframe(self):
        im = ps.generators.fractal_noise(shape=[100, 100, 100], seed=2)<0.65
        df = ps.simulations.chunks_to_dataframe(im=im, scale_factor=3)
        assert len(df.iloc[:, 0]) == 54
        assert df.columns[0] == 'Throat Number'
        assert df.columns[1] == 'Tortuosity'
        assert df.columns[2] == 'Diffusive Conductance'
        assert df.columns[3] == 'Porosity'

        np.testing.assert_array_almost_equal(np.array(df.iloc[:, 1]),
                                             np.array([1.522598, 1.934481, 1.868785,
                                                       1.636542, 1.560184, 1.392518,
                                                       1.557387, 1.549398, 1.443881,
                                                       1.800068, 1.559147, 1.710932,
                                                       1.863328, 1.663121, 1.539908,
                                                       1.555801, 1.598544, 1.561024,
                                                       1.709926, 1.645250, 1.568915,
                                                       1.446752, 1.592016, 1.468578,
                                                       1.380027, 1.418518, 1.728228,
                                                       1.769209, 1.466767, 1.522099,
                                                       1.556685, 1.530968, 2.534509,
                                                       1.605821, 1.837789, 1.694193,
                                                       1.659488, 1.611752, 1.467382,
                                                       1.425772, 1.689578, 1.608209,
                                                       1.524511, 1.825667, 1.530148,
                                                       1.780772, 2.123051, 1.708457,
                                                       1.417780, 1.329248, 1.790564,
                                                       1.871811, 1.661823, 1.508606,]),
                                                       decimal=4)


if __name__ == '__main__':
    t = SimulationsTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
