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
import os
import sys

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.data.slice_ as slice_
import invesalius.gui.dialogs as dlg
from invesalius import inv_paths
from invesalius.gui.default_viewers import ColourSelectButton
from invesalius.gui.widgets.inv_spinctrl import InvSpinCtrl
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher

# INTERPOLATION_MODE_LIST = ["Cubic", "Linear", "NearestNeighbor"]
MIN_TRANSPARENCY = 0
MAX_TRANSPARENCY = 100

OP_LIST = [_("Draw"), _("Erase"), _("Threshold")]


class TaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        inner_panel = InnerTaskPanel(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 0, 7, 7)
        layout.addWidget(inner_panel)


class InnerTaskPanel(QScrollArea):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QScrollArea.NoFrame)

        content = QWidget()
        content.setStyleSheet("background-color: white;")

        bmp_path = os.path.join(inv_paths.ICON_DIR, "object_add.png")
        button_new_surface = QPushButton()
        button_new_surface.setIcon(QIcon(QPixmap(bmp_path)))
        button_new_surface.setFlat(True)
        button_new_surface.clicked.connect(self.OnLinkNewSurface)

        tooltip = _("Create 3D surface based on a mask")
        link_new_surface = QPushButton(_("Create new 3D surface"))
        link_new_surface.setFlat(True)
        link_new_surface.setCursor(Qt.PointingHandCursor)
        link_new_surface.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; color: black; border: none; }"
        )
        link_new_surface.setToolTip(tooltip)
        link_new_surface.clicked.connect(self.OnLinkNewSurface)

        Publisher.subscribe(self.OnLinkNewSurface, "Open create surface dialog")

        line_new = QHBoxLayout()
        line_new.addWidget(link_new_surface, 1)
        line_new.addWidget(button_new_surface, 0)

        fold_panel = FoldPanel(content)

        button_next = QPushButton(_("Next step"))
        button_next.clicked.connect(self.OnButtonNextTask)

        main_layout = QVBoxLayout(content)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addLayout(line_new)
        main_layout.addWidget(fold_panel)
        main_layout.addWidget(button_next, 0, Qt.AlignRight)
        main_layout.addStretch()

        self.setWidget(content)

    def OnButtonNextTask(self):
        Publisher.sendMessage("Fold export task")

    def OnLinkNewSurface(self, evt=None):
        try:
            evt = evt.data
            evt = None
        except Exception:
            pass

        sl = slice_.Slice()

        if sl.current_mask is None:
            dlg.InexistentMask()
            return

        from PySide6.QtWidgets import QDialog

        dialog = dlg.SurfaceCreationDialog(
            None, -1, _("New surface"), mask_edited=sl.current_mask.was_edited
        )

        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                ok = 1
            else:
                ok = 0
        except Exception:
            ok = 1

        if ok:
            surface_options = dialog.GetValue()
            Publisher.sendMessage("Create surface from index", surface_parameters=surface_options)
        dialog.Destroy()


class FoldPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        inner_panel = InnerFoldPanel(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(inner_panel)


class InnerFoldPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        tool_box = QToolBox(self)
        tool_box.addItem(SurfaceProperties(tool_box), _("Surface properties"))
        tool_box.addItem(SurfaceTools(tool_box), _("Advanced options"))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tool_box)

        self.tool_box = tool_box


class SurfaceTools(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        link_style = "QPushButton { text-align: left; color: black; border: none; }"

        tooltip = _("Automatically select largest disconnected region and create new surface")
        link_largest = QPushButton(_("Select largest surface"))
        link_largest.setFlat(True)
        link_largest.setCursor(Qt.PointingHandCursor)
        link_largest.setStyleSheet(link_style)
        link_largest.setToolTip(tooltip)
        link_largest.clicked.connect(self.SelectLargest)

        tooltip = _("Automatically select disconnected regions and create a new surface per region")
        link_split_all = QPushButton(_("Split all disconnected surfaces"))
        link_split_all.setFlat(True)
        link_split_all.setCursor(Qt.PointingHandCursor)
        link_split_all.setStyleSheet(link_style)
        link_split_all.setToolTip(tooltip)
        link_split_all.clicked.connect(self.SplitSurface)

        tooltip = _("Manually insert seeds of regions of interest and create a new surface")
        link_seeds = QPushButton(_("Select regions of interest..."))
        link_seeds.setFlat(True)
        link_seeds.setCursor(Qt.PointingHandCursor)
        link_seeds.setStyleSheet(link_style)
        link_seeds.setToolTip(tooltip)
        link_seeds.clicked.connect(self.OnLinkSeed)

        icon_size = QSize(25, 25)

        icon_largest = QIcon(QPixmap(os.path.join(inv_paths.ICON_DIR, "connectivity_largest.png")))
        button_largest = QPushButton()
        button_largest.setIcon(icon_largest)
        button_largest.setIconSize(icon_size)
        button_largest.setFlat(True)
        button_largest.clicked.connect(self.SelectLargest)

        icon_split = QIcon(QPixmap(os.path.join(inv_paths.ICON_DIR, "connectivity_split_all.png")))
        button_split = QPushButton()
        button_split.setIcon(icon_split)
        button_split.setIconSize(icon_size)
        button_split.setFlat(True)
        button_split.clicked.connect(self.SplitSurface)

        icon_seeds = QIcon(QPixmap(os.path.join(inv_paths.ICON_DIR, "connectivity_manual.png")))
        button_seeds = QPushButton()
        button_seeds.setIcon(icon_seeds)
        button_seeds.setIconSize(icon_size)
        button_seeds.setFlat(True)
        button_seeds.setCheckable(True)
        button_seeds.toggled.connect(self.OnToggleSeeds)
        self.button_seeds = button_seeds

        fixed_layout = QGridLayout()
        fixed_layout.setColumnStretch(0, 1)
        fixed_layout.addWidget(link_largest, 0, 0)
        fixed_layout.addWidget(button_largest, 0, 1)
        fixed_layout.addWidget(link_seeds, 1, 0)
        fixed_layout.addWidget(button_seeds, 1, 1)
        fixed_layout.addWidget(link_split_all, 2, 0)
        fixed_layout.addWidget(button_split, 2, 1)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 5, 0, 0)
        main_layout.addLayout(fixed_layout)

    def OnLinkSeed(self):
        self.button_seeds.setChecked(not self.button_seeds.isChecked())

    def OnToggleSeeds(self, checked):
        self.SelectSeed()

    def SelectLargest(self):
        Publisher.sendMessage("Create surface from largest region")

    def SplitSurface(self):
        Publisher.sendMessage("Split surface")

    def SelectSeed(self):
        if self.button_seeds.isChecked():
            self.StartSeeding()
        else:
            self.EndSeeding()

    def StartSeeding(self):
        print("Start Seeding")
        Publisher.sendMessage("Enable style", style=const.VOLUME_STATE_SEED)
        Publisher.sendMessage("Create surface by seeding - start")

    def EndSeeding(self):
        print("End Seeding")
        Publisher.sendMessage("Disable style", style=const.VOLUME_STATE_SEED)
        Publisher.sendMessage("Create surface by seeding - end")


class SurfaceProperties(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.surface_list = []

        combo_surface_name = QComboBox(self)
        combo_surface_name.currentIndexChanged.connect(self.OnComboName)
        self.combo_surface_name = combo_surface_name

        button_colour = ColourSelectButton(self, colour=(0, 0, 255), size=(22, 22))
        button_colour.colour_selected.connect(self.OnSelectColour)
        self.button_colour = button_colour

        line1 = QHBoxLayout()
        line1.addWidget(combo_surface_name, 1)
        line1.addWidget(button_colour, 0)

        text_transparency = QLabel(_("Transparency:"))

        slider_transparency = QSlider(Qt.Horizontal)
        slider_transparency.setMinimum(MIN_TRANSPARENCY)
        slider_transparency.setMaximum(MAX_TRANSPARENCY)
        slider_transparency.setValue(0)
        slider_transparency.valueChanged.connect(self.OnTransparency)
        self.slider_transparency = slider_transparency

        line2 = QHBoxLayout()
        line2.addWidget(text_transparency, 0)
        line2.addWidget(slider_transparency, 1)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.addLayout(line1)
        main_layout.addLayout(line2)

        self.__bind_events()

    def __bind_events(self):
        Publisher.subscribe(self.InsertNewSurface, "Update surface info in GUI")
        Publisher.subscribe(self.ChangeSurfaceName, "Change surface name")
        Publisher.subscribe(self.OnCloseProject, "Close project data")
        Publisher.subscribe(self.OnRemoveSurfaces, "Remove surfaces")

    def OnRemoveSurfaces(self, surface_indexes):
        s = self.combo_surface_name.currentIndex()
        ns = 0

        old_dict = self.surface_list
        new_dict = []
        i = 0
        for n, (name, index) in enumerate(old_dict):
            if n not in surface_indexes:
                new_dict.append([name, i])
                if s == n:
                    ns = i
                i += 1
        self.surface_list = new_dict

        self.combo_surface_name.blockSignals(True)
        self.combo_surface_name.clear()
        self.combo_surface_name.addItems([n[0] for n in self.surface_list])
        self.combo_surface_name.blockSignals(False)

        if self.surface_list:
            self.combo_surface_name.setCurrentIndex(ns)

    def OnCloseProject(self):
        self.CloseProject()

    def CloseProject(self):
        self.combo_surface_name.blockSignals(True)
        self.combo_surface_name.clear()
        self.combo_surface_name.blockSignals(False)
        self.surface_list = []

    def ChangeSurfaceName(self, index, name):
        self.surface_list[index][0] = name
        self.combo_surface_name.setItemText(index, name)

    def InsertNewSurface(self, surface):
        index = surface.index
        name = surface.name
        colour = [int(value * 255) for value in surface.colour]
        i = 0
        try:
            i = self.surface_list.index([name, index])
            overwrite = True
        except ValueError:
            overwrite = False

        if overwrite:
            self.surface_list[i] = [name, index]
        else:
            self.surface_list.append([name, index])
            i = len(self.surface_list) - 1

        self.combo_surface_name.blockSignals(True)
        self.combo_surface_name.clear()
        self.combo_surface_name.addItems([n[0] for n in self.surface_list])
        self.combo_surface_name.setCurrentIndex(i)
        self.combo_surface_name.blockSignals(False)

        transparency = 100 * surface.transparency
        self.button_colour.SetColour(colour)
        self.slider_transparency.setValue(int(transparency))

    def OnComboName(self, index):
        if index < 0 or index >= len(self.surface_list):
            return
        Publisher.sendMessage("Change surface selected", surface_index=self.surface_list[index][1])

    def OnSelectColour(self, colour_values):
        colour = [value / 255.0 for value in colour_values]
        Publisher.sendMessage(
            "Set surface colour",
            surface_index=self.combo_surface_name.currentIndex(),
            colour=colour,
        )

    def OnTransparency(self, value):
        transparency = value / float(MAX_TRANSPARENCY)
        Publisher.sendMessage(
            "Set surface transparency",
            surface_index=self.combo_surface_name.currentIndex(),
            transparency=transparency,
        )


class QualityAdjustment(QWidget):
    def __init__(self, parent):
        import invesalius.constants as const

        super().__init__(parent)

        combo_quality = QComboBox(self)
        combo_quality.addItems(
            list(const.SURFACE_QUALITY.keys())
            or [
                "",
            ]
        )
        combo_quality.setCurrentIndex(3)
        self.combo_quality = combo_quality

        check_decimate = QCheckBox(self)
        text_decimate = QLabel(_("Decimate resolution:"))
        spin_decimate = InvSpinCtrl(self, -1, value=30, min_value=1, max_value=100)

        check_smooth = QCheckBox(self)
        text_smooth = QLabel(_("Smooth iterations:"))
        spin_smooth = InvSpinCtrl(self, -1, value=0, min_value=1, max_value=100)

        fixed_layout = QGridLayout()
        fixed_layout.setColumnStretch(2, 1)
        fixed_layout.addWidget(check_decimate, 0, 0)
        fixed_layout.addWidget(text_decimate, 0, 1)
        fixed_layout.addWidget(spin_decimate, 0, 2)
        fixed_layout.addWidget(check_smooth, 1, 0)
        fixed_layout.addWidget(text_smooth, 1, 1)
        fixed_layout.addWidget(spin_smooth, 1, 2)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addWidget(combo_quality)
        main_layout.addLayout(fixed_layout)

    def OnComboQuality(self, index):
        print(f"TODO: Send Signal - Change surface quality: {self.combo_quality.currentText()}")
