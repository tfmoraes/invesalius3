# --------------------------------------------------------------------------
# Software:     InVesalius - Software de Reconstrucao 3D de Imagens Medicas
# Copyright:    (C) 2001  Centro de Pesquisas Renato Archer
# Homepage:     http://www.softwarepublico.gov.br
# Contact:      invesalius@cti.gov.br
# License:      GNU - GPL 2 (LICENSE.txt/LICENCA.txt)
# --------------------------------------------------------------------------
#    Este programa e software livre; voce pode redistribui-lo e/ou
#    modifica-lo sob os termos da Licenca Publica Geral GNU, conforme
#    publicada pela Free Software Foundation; de acordo com a versao 2
#    da Licenca.
#
#    Este programa eh distribuido na expectativa de ser util, mas SEM
#    QUALQUER GARANTIA; sem mesmo a garantia implicita de
#    COMERCIALIZACAO ou de ADEQUACAO A QUALQUER PROPOSITO EM
#    PARTICULAR. Consulte a Licenca Publica Geral GNU para obter mais
#    detalhes.
# --------------------------------------------------------------------------

import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.gui.dialogs as dlg
import invesalius.gui.widgets.gradient as grad
import invesalius.session as ses
import invesalius.utils as utils
from invesalius.data.slice_ import Slice
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher


class TaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        inner_panel = InnerTaskPanel(self)

        sizer = QHBoxLayout(self)
        sizer.setContentsMargins(7, 0, 7, 7)
        sizer.addWidget(inner_panel)


class InnerTaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.__bind_events()

        self.session = ses.Session()
        self.slc = Slice()
        self.colormaps = [
            "autumn",
            "hot",
            "plasma",
            "cividis",
            "bwr",
            "RdBu",
            "Set3",
            "tab10",
            "twilight",
            "hsv",
        ]
        self.current_colormap = "autumn"
        self.number_colors = 10
        self.cluster_volume = None
        self.zero_value = 0

        line0 = QLabel(_("Select Modalities / File"), self)

        btn_load = QPushButton(_("Load"), self)
        btn_load.setFixedSize(65, 23)
        btn_load.setToolTip(_("Load Nifti image"))
        btn_load.setEnabled(True)
        btn_load.clicked.connect(self.OnLoadFmri)
        self.btn_load = btn_load

        line1 = QHBoxLayout()
        line1.addWidget(btn_load, 1)

        text_thresh = QLabel(_("Select Colormap"), self)

        combo_thresh = QComboBox(self)
        combo_thresh.addItems(self.colormaps)
        combo_thresh.activated.connect(self.OnSelectColormap)
        combo_thresh.setCurrentIndex(self.colormaps.index(self.current_colormap))
        self.combo_thresh = combo_thresh

        cmap = plt.get_cmap(self.current_colormap)
        colors_gradient = self.GenerateColormapColors(cmap)

        self.gradient = grad.GradientDisp(self, -1, -5000, 5000, -5000, 5000, colors_gradient)

        sizer = QVBoxLayout(self)
        sizer.addSpacing(7)
        sizer.addWidget(line0)
        sizer.addSpacing(5)
        sizer.addLayout(line1)
        sizer.addSpacing(5)
        sizer.addWidget(text_thresh)
        sizer.addSpacing(2)
        sizer.addWidget(combo_thresh)
        sizer.addSpacing(5)
        sizer.addWidget(self.gradient, 1)
        sizer.addSpacing(7)

        self.UpdateGradient(self.gradient, colors_gradient)

    def __bind_events(self):
        pass

    def OnSelectColormap(self, event=None):
        self.current_colormap = self.colormaps[self.combo_thresh.currentIndex()]
        colors = self.GenerateColormapColors(self.current_colormap, self.number_colors)

        self.UpdateGradient(self.gradient, colors)

        if isinstance(self.cluster_volume, np.ndarray):
            self.apply_colormap(self.current_colormap, self.cluster_volume, self.zero_value)

    def GenerateColormapColors(self, colormap_name, number_colors=10):
        cmap = plt.get_cmap(colormap_name)
        colors_gradient = [
            (
                int(255 * cmap(i)[0]),
                int(255 * cmap(i)[1]),
                int(255 * cmap(i)[2]),
                int(255 * cmap(i)[3]),
            )
            for i in np.linspace(0, 1, number_colors)
        ]

        return colors_gradient

    def UpdateGradient(self, gradient, colors):
        gradient.SetGradientColours(colors)
        gradient.repaint()
        gradient.update()

        self.repaint()
        self.update()
        self.show()

    def OnLoadFmri(self, event=None):
        filename = dlg.ShowImportOtherFilesDialog(id_type=const.ID_NIFTI_IMPORT)
        filename = utils.decode(filename, const.FS_ENCODE)

        fmri_data = nb.squeeze_image(nb.load(filename))
        fmri_data = nb.as_closest_canonical(fmri_data)
        fmri_data.update_header()

        cluster_volume_original = fmri_data.get_fdata().T[:, ::-1].copy()
        cluster_volume_normalized = (cluster_volume_original - np.min(cluster_volume_original)) / (
            np.max(cluster_volume_original) - np.min(cluster_volume_original)
        )
        self.cluster_volume = (cluster_volume_normalized * 255).astype(np.uint8)

        self.zero_value = int(
            (0.0 - np.min(cluster_volume_original))
            / (np.max(cluster_volume_original) - np.min(cluster_volume_original))
            * 255
        )

        if self.slc.matrix.shape != self.cluster_volume.shape:
            QMessageBox.warning(
                self,
                "InVesalius 3",
                "The overlay volume does not match the underlying structural volume",
            )

        else:
            self.slc.aux_matrices["color_overlay"] = self.cluster_volume
            self.slc.to_show_aux = "color_overlay"
            self.apply_colormap(self.current_colormap, self.cluster_volume, self.zero_value)

    def apply_colormap(self, colormap, cluster_volume, zero_value):
        cmap = plt.get_cmap(colormap)

        cluster_volume_unique = np.unique(cluster_volume)
        colors = cmap(cluster_volume_unique / 255)
        color_dict = {val: color for val, color in zip(cluster_volume_unique, map(tuple, colors))}

        self.slc.aux_matrices_colours["color_overlay"] = color_dict
        if zero_value in self.slc.aux_matrices_colours["color_overlay"]:
            self.slc.aux_matrices_colours["color_overlay"][zero_value] = (0.0, 0.0, 0.0, 0.0)
        else:
            print("Zero value not found in color_overlay. No data is set as transparent.")

        Publisher.sendMessage("Reload actual slice")
