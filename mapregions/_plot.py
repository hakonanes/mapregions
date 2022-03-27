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

from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib_scalebar import dimension, scalebar
import numpy as np


class MapRegionsPlot(Axes):
    name = "map_region_plot"
    colorbar = None
    scalebar = None

    def plot_map(
        self, map_region, data, scalebar_properties=dict(), **kwargs,
    ):
        if scalebar:
            _ = self.add_scalebar(map_region, **scalebar_properties)

        return self.imshow(X=data, **kwargs)

    def add_scalebar(self, map_region, **kwargs):
        # Set a reasonable unit dimension
        scan_unit = map_region.scan_unit
        if scan_unit == "px":
            dim = "pixel-length"
        elif scan_unit[-1] == "m":
            dim = "si-length"  # Default
        else:
            dim = dimension._Dimension(scan_unit)

        d = dict(
            pad=0.2,
            sep=3,
            border_pad=0.5,
            location="lower left",
            box_alpha=0.6,
            dimension=dim,
        )
        [kwargs.setdefault(k, v) for k, v in d.items()]

        # Create scalebar
        bar = scalebar.ScaleBar(dx=map_region.dx, units=scan_unit, **kwargs)
        self.axes.add_artist(bar)
        self.scalebar = bar

        return bar

    def add_overlay(self, overlay):
        image = self.images[0]
        image_data = image.get_array()

        # Scale prop to [0, 1] to maximize image contrast
        overlay_min = np.nanmin(overlay)
        rescaled_overlay = (overlay - overlay_min) / (np.nanmax(overlay) - overlay_min)

        n_channels = 3
        for i in range(n_channels):
            image_data[:, :, i] *= rescaled_overlay

        image.set_data(image_data)

    def add_colorbar(self, label=None, **kwargs):
        # Keyword arguments
        d = {"position": "right", "size": "5%", "pad": 0.1}
        [kwargs.setdefault(k, v) for k, v in d.items()]

        # Add colorbar
        divider = make_axes_locatable(self)
        cax = divider.append_axes(**kwargs)
        cbar = self.figure.colorbar(self.images[0], cax=cax)

        # Set label with padding
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(label, rotation=270)

        self.colorbar = cbar

        return cbar

    def add_labels(self, map_region, **kwargs):
        for (r, c), l in zip(map_region.medoid, map_region.label):
            self.text(x=c, y=r, s=l, fontdict=kwargs)

    def remove_padding(self):
        self.set_axis_off()
        self.margins(0, 0)

        # Tune subplot layout
        cbar = self.images[0].colorbar
        if cbar is not None:
            right = self.figure.subplotpars.right
        else:
            right = 1
        self.figure.subplots_adjust(top=1, bottom=0, right=right, left=0)


register_projection(MapRegionsPlot)
