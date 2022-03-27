# MIT License
#
# Copyright (c) 2022 Håkon Wiik Ånes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from copy import deepcopy
import random

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numba as nb
import numpy as np
from scipy import ndimage, spatial


class MapRegions:
    dx = 1
    dy = 1
    scan_unit = "px"
    background_label = -1

    def __init__(
        self,
        label_map,
        dx=1,
        dy=1,
        scan_unit="px",
        background_label=-1,
        intensity_image=None,
    ):
        self._label_map = label_map.astype(int)
        self.dx = dx
        self.dy = dy
        self.scan_unit = scan_unit
        self.background_label = background_label
        self.intensity_image = intensity_image

    def __getitem__(self, key):
        labels_to_keep = self.label[key]
        labels_to_keep_map = np.isin(self.label_map, labels_to_keep)
        new_regions = self.deepcopy()
        new_regions._label_map, _ = ndimage.label(labels_to_keep_map)
        return new_regions

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.label.size}"

    @property
    def _flat_label_map(self):
        return self.label_map.ravel()

    @property
    def _flat_masked_label_map(self):
        return self._flat_label_map[~self.is_background_map_flat]

    @property
    def has_background(self):
        return np.any(self.is_background_map)

    @property
    def _is_background(self):
        all_labels = np.unique(self._flat_label_map)
        return all_labels == self.background_label

    @property
    def is_background_map(self):
        return self.label_map == self.background_label

    @property
    def is_background_map_flat(self):
        return self.is_background_map.ravel()

    @property
    def label_map(self):
        return self._label_map

    @property
    def label(self):
        return np.unique(self._flat_masked_label_map)

    @property
    def map_shape(self):
        return self.label_map.shape

    @property
    def map_size(self):
        return self.label_map.size

    @property
    def _map_row_column_indices(self):
        r, c = np.indices(self.map_shape)
        return np.column_stack([r.ravel(), c.ravel()])

    @property
    def pixel_area(self):
        return self.dx * self.dy

    @property
    def slice(self):
        return ndimage.find_objects(self.label_map)

    @property
    def _slice_coordinates(self):
        n_labels = self.label.size
        slice_start_stop = np.zeros((n_labels, 4), dtype=np.int64)
        slices = self.slice
        for i in range(n_labels):
            sl = slices[i]
            slice_start_stop[i, 0] = sl[0].start
            slice_start_stop[i, 1] = sl[0].stop
            slice_start_stop[i, 2] = sl[1].start
            slice_start_stop[i, 3] = sl[1].stop
        return slice_start_stop

    # ----------------------- Region properties ---------------------- #

    # TODO: Use Numba for looping over regions to get properties when
    #  these cannot be obtained with a vectorized operation

    @property
    def area(self):
        return self.size * self.pixel_area

    @property
    def area_convex(self):
        area_convex = np.zeros(self.label.size)
        qhull = self.convex_hull
        idx = np.arange(self.label.size)[qhull != 0]
        for i in idx:
            area_convex[i] = qhull[i].volume
        return area_convex

    @property
    def centroid(self):
        # Source: https://stackoverflow.com/questions/46840707/efficiently-find-centroid-of-labelled-image-regions
        nr, nc = self.map_shape
        r, c = np.mgrid[:nr, :nc]
        flat_label_map = self._flat_label_map
        count = np.bincount(flat_label_map)
        nonzero = count != 0
        count = count[nonzero]
        centroid_r = np.bincount(flat_label_map, r.ravel())[nonzero] / count
        centroid_c = np.bincount(flat_label_map, c.ravel())[nonzero] / count
        return np.column_stack([centroid_r, centroid_c])[~self._is_background]

    @property
    def circularity(self):
        perimeter = self.perimeter
        return (
            4
            * np.pi
            * np.divide(
                self.area,
                perimeter ** 2,
                where=perimeter != 0,
                out=np.zeros_like(self.perimeter),
            )
        )

    @property
    def coordinates(self):
        return _get_coordinates_from_region(
            label_map=self.label_map.astype(np.int64),
            labels=self.label.astype(np.int64),
            n_max_coordinates=self.size.max(),
            slice_coordinates=self._slice_coordinates,
        )

    @property
    def equivalent_radius(self):
        return np.sqrt(self.area / np.pi)

    @property
    def max_intensity(self):
        image = self.intensity_image
        if image is not None:
            return ndimage.labeled_comprehension(
                input=image,
                labels=self.label_map,
                index=np.arange(1, self.label.size + 1),
                func=np.max,
                out_dtype=float,
                default=0,
            )
        else:
            raise ValueError("No `intensity_image` available")

    @property
    def min_intensity(self):
        image = self.intensity_image
        if image is not None:
            return ndimage.labeled_comprehension(
                input=image,
                labels=self.label_map,
                index=np.arange(1, self.label.size + 1),
                func=np.min,
                out_dtype=float,
                default=0,
            )
        else:
            raise ValueError("No `intensity_image` available")

    @property
    def mean_intensity(self):
        image = self.intensity_image
        if image is not None:
            return ndimage.labeled_comprehension(
                input=image,
                labels=self.label_map,
                index=np.arange(1, self.label.size + 1),
                func=np.mean,
                out_dtype=float,
                default=0,
            )
        else:
            raise ValueError("No `intensity_image` available")

    @property
    def medoid(self):
        rc = self._map_row_column_indices
        centroids = self.centroid
        medoids = np.zeros_like(centroids)
        for i, mask in enumerate(self._iter_mask_flat()):
            node = centroids[i]
            nodes = rc[mask]
            medoids[i] = _medoid(node, nodes)
        return medoids

    @property
    def perimeter(self):
        perimeters = np.zeros(self.label.size)
        binary_map = (~self.is_background_map).astype(np.uint8)
        for i, sl in enumerate(self.slice):
            perimeters[i] = _perimeter(binary_map[sl])
        return perimeters

    @property
    def convex_hull(self):
        qhull = np.zeros(self.label.size, dtype=object)
        coordinates = self.coordinates
        sizes = self.size
        idx = np.arange(self.label.size)[self.size > 2]
        for i in idx:
            coords = coordinates[i]
            coords = coords[coords != -1].reshape((sizes[i], 2))
            try:
                qhull[i] = spatial.ConvexHull(coords)
            except spatial.qhull.QhullError:
                pass
        return qhull

    @property
    def roundness(self):
        # Convex perimeter / perimeter
        roundness = np.zeros(self.label.size)
        qhull = self.convex_hull
        perimeter = self.perimeter
        idx = np.arange(self.label.size)[qhull != 0]
        for i in idx:
            roundness[i] = qhull[i].area / perimeter[i]
        return roundness

    @property
    def size(self):
        _, counts = np.unique(self._flat_masked_label_map, return_counts=True)
        return counts

    @property
    def solidity(self):
        area_convex = self.area_convex
        return np.divide(
            self.area,
            area_convex,
            where=area_convex != 0,
            out=np.zeros_like(area_convex),
        )

    @property
    def std_intensity(self):
        image = self.intensity_image
        if image is not None:
            return ndimage.labeled_comprehension(
                input=image,
                labels=self.label_map,
                index=np.arange(1, self.label.size + 1),
                func=np.std,
                out_dtype=float,
                default=0,
            )
        else:
            raise ValueError("No `intensity_image` available")

    # -------------------- End of region properties ------------------ #

    @classmethod
    def from_image(cls, image, **kwargs):
        label_map, _ = ndimage.label(image)
        kwargs.setdefault("background_label", 0)
        return cls(label_map, **kwargs)

    def deepcopy(self):
        return deepcopy(self)

    def get_map_data(self, data):
        if isinstance(data, str):
            data = self.__getattribute__(data)
        map_data = np.zeros(self.map_shape)
        idx = self._flat_masked_label_map
        if self.has_background:
            idx -= 1
        map_data[~self.is_background_map] = data[idx]
        return map_data

    def _iter_mask(self):
        for l in self.label:
            yield self._label_map == l

    def _iter_mask_flat(self):
        for l in self.label:
            yield self._flat_label_map == l

    def _label2rgb_random(self, background_color=None):
        n = self.label.size
        colors = np.zeros((n, 3))
        for i in range(n):
            hex_id = "#%06x" % random.randint(0, 0xFFFFFF)
            colors[i] = mcolors.hex2color(hex_id)
        if background_color is None:
            background_color = (0.5, 0.5, 0.5)
        rgb = np.ones((self.map_size, 3)) * background_color
        idx = self._flat_masked_label_map
        if self.has_background:
            idx -= 1
        rgb[~self.is_background_map_flat] = colors[idx]
        rgb = rgb.reshape(self.map_shape + (3,))
        return rgb

    def plot(
        self,
        data=None,
        overlay=None,
        background_color=None,
        scalebar=True,
        scalebar_properties=dict(),
        colorbar=False,
        colorbar_label=None,
        colorbar_properties=dict(),
        text=False,
        text_kwargs=dict(),
        remove_padding=False,
        return_figure=False,
        figure_kwargs=dict(),
        **kwargs,
    ):
        # Register "map_region_plot" projection with Matplotlib
        import orix.plot.map_region_plot

        fig = plt.figure(**figure_kwargs)
        ax = fig.add_subplot(projection="map_region_plot")

        if data is None:
            data = self._label2rgb_random(background_color)
        else:
            data = self.get_map_data(data)

        ax.plot_map(
            self,
            data=data,
            scalebar=scalebar,
            scalebar_properties=scalebar_properties,
            **kwargs,
        )
        if overlay is not None:
            ax.add_overlay(overlay)
        if remove_padding:
            ax.remove_padding()
        if colorbar:
            ax.add_colorbar(label=colorbar_label, **colorbar_properties)
        if text:
            ax.add_labels(self, **text_kwargs)
        if return_figure:
            return fig


def _medoid(node, nodes):
    d = spatial.distance.cdist(np.atleast_2d(node), nodes)
    return nodes[np.argmin(d)]


@nb.jit(
    "int64[:, :, :](int64[:, :], int64[:], int64, int64[:, :])",
    cache=True,
    nogil=True,
    nopython=True,
)
def _get_coordinates_from_region(
    label_map, labels, n_max_coordinates, slice_coordinates
):
    n_labels = labels.size
    coordinates = np.full((n_labels, n_max_coordinates, 2), -1, dtype=np.int64)
    for i in nb.prange(n_labels):
        r0, r1, c0, c1 = slice_coordinates[i]
        r, c = np.where(label_map[r0:r1, c0:c1] == labels[i])
        coordinates[i, : r.size, 0] = r
        coordinates[i, : c.size, 1] = c
    return coordinates


def _perimeter(binary_image):
    # Taken from scikit-image:
    # https://github.com/scikit-image/scikit-image/blob/main/skimage/measure/_regionprops_utils.py#L186
    eroded_image = ndimage.binary_erosion(binary_image, border_value=0)
    border_image = binary_image - eroded_image
    perimeter_image = ndimage.convolve(
        border_image,
        np.array([[10, 2, 10], [2, 1, 2], [10, 2, 10]]),
        mode="constant",
        cval=0,
    )

    perimeter_weights = np.zeros(50, dtype=np.double)
    perimeter_weights[[5, 7, 15, 17, 25, 27]] = 1
    perimeter_weights[[21, 33]] = np.sqrt(2)
    perimeter_weights[[13, 23]] = (1 + np.sqrt(2)) / 2

    perimeter_histogram = np.bincount(perimeter_image.ravel(), minlength=50)
    total_perimeter = perimeter_histogram.dot(perimeter_weights)

    return total_perimeter
