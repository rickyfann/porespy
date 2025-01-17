import os
import sys
from os.path import realpath
from pathlib import Path
from platform import system

import numpy as np
import porespy as ps
import pytest

ps.settings.tqdm['disable'] = True


class NetworkExtractionTest():
    def setup_class(self):
        self.im = ps.generators.blobs(shape=[300, 300],
                                      seed=0,
                                      porosity=0.4912888888888889)
        assert self.im.sum()/self.im.size == 0.4912888888888889
        self.snow = ps.filters.snow_partitioning(self.im)
        self.im3d = ps.generators.blobs(shape=[50, 50, 50],
                                        seed=0,
                                        porosity=0.500144)
        assert self.im3d.sum()/self.im3d.size == 0.500144
        self.snow3d = ps.filters.snow_partitioning(self.im3d)

    def test_regions_to_network(self):
        im = self.snow.regions*self.im
        net = ps.networks.regions_to_network(im)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_snow_2d(self):
        a = np.unique(self.snow.peaks*self.im)
        b = np.unique(self.snow.regions*self.im)
        assert len(a) == len(b)

    def test_snow_3d(self):
        a = np.unique(self.snow3d.peaks*self.im3d)
        b = np.unique(self.snow3d.regions*self.im3d)
        assert len(a) == len(b)

    def test_extract_pore_network_3d(self):
        im = self.snow3d.regions*self.im3d
        net = ps.networks.regions_to_network(im)
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_snow(self):
        snow = ps.networks.snow2(self.im3d)
        net = snow.network
        found_nans = False
        for key in net.keys():
            if np.any(np.isnan(net[key])):
                found_nans = True
        assert found_nans is False

    def test_map_to_regions(self):
        im = self.im
        snow = ps.filters.snow_partitioning(im)
        regions = snow.regions
        values = np.random.rand(regions.max())
        mapped = ps.networks.map_to_regions(regions, values)
        assert mapped.max() < 1
        # Some failures
        values = np.random.rand(regions.max()+2)
        with pytest.raises(Exception):
            mapped = ps.networks.map_to_regions(regions, values)

    def test_planar_2d_image(self):
        im1 = ps.generators.blobs([100, 100, 1], seed=1, porosity=0.4998)
        assert im1.sum()/im1.size == 0.4998
        im2 = ps.generators.blobs([100, 1, 100], seed=1, porosity=0.4998)
        assert im2.sum()/im2.size == 0.4998
        im3 = ps.generators.blobs([1, 100, 100], seed=1, porosity=0.4998)
        assert im3.sum()/im3.size == 0.4998
        np.random.seed(1)
        snow_out1 = ps.filters.snow_partitioning(im1)
        pore_map1 = snow_out1.im * snow_out1.regions
        net1 = ps.networks.regions_to_network(regions=pore_map1,
                                              voxel_size=1)
        np.random.seed(1)
        snow_out2 = ps.filters.snow_partitioning(im2)
        pore_map2 = snow_out2.im * snow_out2.regions
        net2 = ps.networks.regions_to_network(regions=pore_map2,
                                              voxel_size=1)
        np.random.seed(1)
        snow_out3 = ps.filters.snow_partitioning(im3)
        pore_map3 = snow_out3.im * snow_out3.regions
        net3 = ps.networks.regions_to_network(regions=pore_map3,
                                              voxel_size=1)
        assert np.allclose(net1['pore.coords'][:, 0], net2['pore.coords'][:, 0])
        assert np.allclose(net1['pore.coords'][:, 1], net2['pore.coords'][:, 2])
        assert np.allclose(net1['pore.coords'][:, 0], net3['pore.coords'][:, 1])

    @pytest.mark.skipif(not sys.platform.startswith("win"), reason="Windows-only!")
    def test_max_ball(self):
        path = Path(realpath(__file__), '../../fixtures/pnextract.exe')
        if system() == 'Windows':
            ps.networks.maximal_ball_wrapper(im=self.im3d,
                                             prefix='test_maxball',
                                             path_to_exe=path,
                                             voxel_size=1e-6)
            assert os.path.isfile("test_maxball_link1.dat")
            assert os.path.isfile("test_maxball_link2.dat")
            assert os.path.isfile("test_maxball_node1.dat")
            assert os.path.isfile("test_maxball_node2.dat")
            os.remove("test_maxball_link1.dat")
            os.remove("test_maxball_link2.dat")
            os.remove("test_maxball_node1.dat")
            os.remove("test_maxball_node2.dat")


if __name__ == '__main__':
    t = NetworkExtractionTest()
    self = t
    t.setup_class()
    for item in t.__dir__():
        if item.startswith('test'):
            print(f'Running test: {item}')
            t.__getattribute__(item)()
