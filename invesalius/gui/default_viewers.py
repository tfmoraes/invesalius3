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

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QActionGroup, QColor, QCursor, QIcon, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QInputDialog,
    QMenu,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.data.viewer_slice as slice_viewer
import invesalius.data.viewer_volume as volume_viewer
import invesalius.gui.widgets.slice_menu as slice_menu_
import invesalius.project as project
import invesalius.session as ses
from invesalius import inv_paths
from invesalius.constants import ID_TO_BMP
from invesalius.gui.widgets.clut_raycasting import CLUTRaycastingWidget
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher


class Panel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._maximized_widget = None
        self.__init_splitter_layout()
        self.__bind_events()

    def __init_splitter_layout(self):
        p1 = slice_viewer.Viewer(self, "AXIAL")
        p2 = slice_viewer.Viewer(self, "CORONAL")
        p3 = slice_viewer.Viewer(self, "SAGITAL")
        p4 = VolumeViewerCover(self)

        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.p4 = p4

        menu = slice_menu_.SliceMenu()
        p1.SetPopupMenu(menu)
        p2.SetPopupMenu(menu)
        p3.SetPopupMenu(menu)

        self.h_splitter_top = QSplitter(Qt.Horizontal)
        self.h_splitter_top.addWidget(p1)
        self.h_splitter_top.addWidget(p2)

        self.h_splitter_bottom = QSplitter(Qt.Horizontal)
        self.h_splitter_bottom.addWidget(p3)
        self.h_splitter_bottom.addWidget(p4)

        self.v_splitter = QSplitter(Qt.Vertical)
        self.v_splitter.addWidget(self.h_splitter_top)
        self.v_splitter.addWidget(self.h_splitter_bottom)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.v_splitter)

        session = ses.Session()
        if session.GetConfig("mode") != const.MODE_NAVIGATOR:
            Publisher.sendMessage("Hide target button")

    def __bind_events(self):
        Publisher.subscribe(self.OnSetTargetMode, "Set target mode")
        Publisher.subscribe(self.OnStartNavigation, "Start navigation")
        Publisher.subscribe(self.UpdateViewerCaption, "Update viewer caption")
        Publisher.subscribe(self._Exit, "Exit")

    def UpdateViewerCaption(self, viewer_name: str, caption: str):
        """Update the caption of a viewer pane.

        QSplitter panes do not have captions, so this is currently a no-op.

        Args:
            viewer_name (str): The name of the viewer pane to update.
            caption (str): The new caption to set for the viewer pane.
        """
        pass

    def OnSetTargetMode(self, enabled=True):
        if enabled:
            self.MaximizeViewerVolume()
        else:
            self.RestoreViewerVolume()

    def OnStartNavigation(self):
        self.MaximizeViewerVolume()

    def RestoreViewerVolume(self):
        if self._maximized_widget is None:
            return
        self.h_splitter_top.show()
        self.p3.show()
        self._maximized_widget = None
        Publisher.sendMessage("Hide raycasting widget")

    def MaximizeViewerVolume(self):
        self.RestoreViewerVolume()
        self.h_splitter_top.hide()
        self.p3.hide()
        self._maximized_widget = self.p4
        Publisher.sendMessage("Show raycasting widget")

    def _Exit(self):
        pass


class VolumeInteraction(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.can_show_raycasting_widget = 0
        self.__init_layout()
        self.__bind_events()

    def __init_layout(self):
        p1 = volume_viewer.Viewer(self)
        self.clut_raycasting = CLUTRaycastingWidget(self)
        self.clut_raycasting.hide()

        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.addWidget(p1)
        self.splitter.addWidget(self.clut_raycasting)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.splitter)

        self.clut_raycasting.clut_point_release.connect(self.OnPointChanged)
        self.clut_raycasting.clut_curve_select.connect(self.OnCurveSelected)
        self.clut_raycasting.clut_curve_wl_change.connect(self.OnChangeCurveWL)

    def __bind_events(self):
        Publisher.subscribe(self.ShowRaycastingWidget, "Show raycasting widget")
        Publisher.subscribe(self.HideRaycastingWidget, "Hide raycasting widget")
        Publisher.subscribe(self.OnSetRaycastPreset, "Update raycasting preset")
        Publisher.subscribe(self.RefreshPoints, "Refresh raycasting widget points")
        Publisher.subscribe(self.LoadHistogram, "Load histogram")
        Publisher.subscribe(self._Exit, "Exit")

    def __update_curve_wwwl_text(self, curve):
        ww, wl = self.clut_raycasting.GetCurveWWWl(curve)
        Publisher.sendMessage("Set raycasting wwwl", ww=ww, wl=wl, curve=curve)

    def ShowRaycastingWidget(self):
        self.can_show_raycasting_widget = 1
        if self.clut_raycasting.to_draw_points:
            self.clut_raycasting.show()

    def HideRaycastingWidget(self):
        self.can_show_raycasting_widget = 0
        self.clut_raycasting.hide()

    def OnPointChanged(self, curve):
        Publisher.sendMessage("Set raycasting refresh")
        Publisher.sendMessage("Set raycasting curve", curve=curve)
        Publisher.sendMessage("Render volume viewer")

    def OnCurveSelected(self, curve):
        Publisher.sendMessage("Set raycasting curve", curve=curve)
        Publisher.sendMessage("Render volume viewer")

    def OnChangeCurveWL(self, curve):
        self.__update_curve_wwwl_text(curve)
        Publisher.sendMessage("Render volume viewer")

    def OnSetRaycastPreset(self):
        preset = project.Project().raycasting_preset
        self.clut_raycasting.SetRaycastPreset(preset)
        if self.clut_raycasting.to_draw_points and self.can_show_raycasting_widget:
            self.clut_raycasting.show()
        else:
            self.clut_raycasting.hide()

    def LoadHistogram(self, histogram, init, end):
        self.clut_raycasting.SetRange((init, end))
        self.clut_raycasting.SetHistogramArray(histogram, (init, end))

    def RefreshPoints(self):
        self.clut_raycasting.CalculatePixelPoints()
        self.clut_raycasting.update()

    def _Exit(self):
        pass


ICON_SIZE = (32, 32)


class ColourSelectButton(QPushButton):
    """A button that displays a colour swatch and opens a QColorDialog when clicked."""

    colour_selected = Signal(list)

    def __init__(self, parent, colour=(0, 0, 0), size=(32, 32)):
        super().__init__(parent)
        self.setFixedSize(size[0], size[1])
        self._colour = QColor(*colour)
        self._update_swatch()
        self.clicked.connect(self._on_clicked)
        self.setFlat(True)

    def _update_swatch(self):
        icon_sz = QSize(self.width() - 4, self.height() - 4)
        self.setIconSize(icon_sz)
        pixmap = QPixmap(icon_sz)
        pixmap.fill(self._colour)
        self.setIcon(QIcon(pixmap))

    def _on_clicked(self):
        colour = QColorDialog.getColor(self._colour, self)
        if colour.isValid():
            self._colour = colour
            self._update_swatch()
            self.colour_selected.emit([colour.red(), colour.green(), colour.blue()])

    def SetColour(self, colour):
        self._colour = QColor(*[int(c) for c in colour])
        self._update_swatch()


class VolumeViewerCover(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(VolumeInteraction(self), 1)
        layout.addWidget(VolumeToolPanel(self), 0)


class VolumeToolPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        icon_raycasting = QIcon(QPixmap(str(inv_paths.ICON_DIR.joinpath("volume_raycasting.png"))))
        icon_slice_plane = QIcon(QPixmap(str(inv_paths.ICON_DIR.joinpath("slice_plane.png"))))
        icon_3d_stereo = QIcon(QPixmap(str(inv_paths.ICON_DIR.joinpath("3D_glasses.png"))))

        self.button_raycasting = QPushButton(self)
        self.button_raycasting.setIcon(icon_raycasting)
        self.button_raycasting.setIconSize(QSize(*ICON_SIZE))
        self.button_raycasting.setFixedSize(*ICON_SIZE)
        self.button_raycasting.setFlat(True)
        self.button_raycasting.setToolTip("Raycasting view")

        self.button_stereo = QPushButton(self)
        self.button_stereo.setIcon(icon_3d_stereo)
        self.button_stereo.setIconSize(QSize(*ICON_SIZE))
        self.button_stereo.setFixedSize(*ICON_SIZE)
        self.button_stereo.setFlat(True)
        self.button_stereo.setToolTip("Real 3D")

        self.button_slice_plane = QPushButton(self)
        self.button_slice_plane.setIcon(icon_slice_plane)
        self.button_slice_plane.setIconSize(QSize(*ICON_SIZE))
        self.button_slice_plane.setFixedSize(*ICON_SIZE)
        self.button_slice_plane.setFlat(True)
        self.button_slice_plane.setToolTip("Slices into 3D")

        icon_front = QIcon(QPixmap(ID_TO_BMP[const.VOL_FRONT][1]))
        self.button_view = QPushButton(self)
        self.button_view.setIcon(icon_front)
        self.button_view.setIconSize(QSize(32, 32))
        self.button_view.setFixedSize(32, 32)
        self.button_view.setFlat(True)
        self.button_view.setToolTip("View plane")

        if sys.platform.startswith("linux"):
            size = (32, 32)
            sp = 2
        else:
            size = (24, 24)
            sp = 5

        self.button_colour = ColourSelectButton(self, colour=(0, 0, 0), size=size)
        self.button_colour.setToolTip("Background Colour")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(sp, sp, sp, sp)
        layout.setSpacing(2)
        layout.addWidget(self.button_colour)
        layout.addWidget(self.button_raycasting)
        layout.addWidget(self.button_view)
        layout.addWidget(self.button_slice_plane)
        layout.addWidget(self.button_stereo)
        layout.addStretch()

        self.navigation_status = False
        self.target_selected = False
        self.track_obj = False

        self.__init_menus()
        self.__bind_events()
        self.__bind_events_wx()

    def __bind_events(self):
        Publisher.subscribe(self.ChangeButtonColour, "Change volume viewer gui colour")
        Publisher.subscribe(self.DisablePreset, "Close project data")
        Publisher.subscribe(self.Uncheck, "Uncheck image plane menu")

    def DisablePreset(self):
        self.off_action.setChecked(True)

    def __bind_events_wx(self):
        self.button_slice_plane.clicked.connect(self.OnButtonSlicePlane)
        self.button_raycasting.clicked.connect(self.OnButtonRaycasting)
        self.button_view.clicked.connect(self.OnButtonView)
        self.button_colour.colour_selected.connect(self.OnSelectColour)
        self.button_stereo.clicked.connect(self.OnButtonStereo)

    def OnButtonRaycasting(self):
        self.menu_raycasting.exec(QCursor.pos())

    def OnButtonStereo(self):
        self.stereo_menu.exec(QCursor.pos())

    def OnButtonView(self):
        self.menu_view.exec(QCursor.pos())

    def OnButtonSlicePlane(self):
        self.slice_plane_menu.exec(QCursor.pos())

    def OnSavePreset(self):
        preset_name, ok = QInputDialog.getText(self, _("Preset name"), _("Preset name"))
        if ok and preset_name:
            Publisher.sendMessage("Save raycasting preset", preset_name=preset_name)

    def __init_menus(self):
        # RAYCASTING TYPES MENU
        menu = self.menu_raycasting = QMenu(self)
        raycasting_group = QActionGroup(self)
        raycasting_group.setExclusive(True)

        for name in const.RAYCASTING_TYPES:
            action = menu.addAction(name)
            action.setCheckable(True)
            raycasting_group.addAction(action)
            if name == const.RAYCASTING_OFF_LABEL:
                self.off_action = action
                action.setChecked(True)
            action.triggered.connect(lambda checked, n=name: self._on_raycasting_type(n))

        menu.addSeparator()

        # RAYCASTING TOOLS SUBMENU
        self.tools_submenu = QMenu(_("Tools"), self)
        self.tools_submenu.setEnabled(False)
        self.tool_actions = {}

        for name in const.RAYCASTING_TOOLS:
            action = self.tools_submenu.addAction(name)
            action.setCheckable(True)
            self.tool_actions[name] = action
            action.triggered.connect(lambda checked, n=name: self._on_raycasting_tool(n, checked))

        menu.addMenu(self.tools_submenu)

        # VOLUME VIEW ANGLE MENU
        self.menu_view = QMenu(self)
        for id_val in ID_TO_BMP:
            icon = QIcon(QPixmap(ID_TO_BMP[id_val][1]))
            action = self.menu_view.addAction(icon, ID_TO_BMP[id_val][0])
            action.setData(id_val)
            action.triggered.connect(lambda checked, iv=id_val: self._on_view(iv))

        # SLICE PLANES MENU
        self.slice_plane_menu = QMenu(self)
        self.slice_plane_actions = {}
        for name in ("Axial", "Coronal", "Sagital"):
            action = self.slice_plane_menu.addAction(name)
            action.setCheckable(True)
            self.slice_plane_actions[name] = action
            action.triggered.connect(lambda checked, n=name: self._on_slice_plane(n, checked))

        # 3D STEREO MENU
        self.stereo_menu = QMenu(self)
        stereo_group = QActionGroup(self)
        stereo_group.setExclusive(True)
        stereo_items = [
            const.STEREO_OFF,
            const.STEREO_RED_BLUE,
            const.STEREO_ANAGLYPH,
            const.STEREO_CRISTAL,
            const.STEREO_INTERLACED,
            const.STEREO_LEFT,
            const.STEREO_RIGHT,
            const.STEREO_DRESDEN,
            const.STEREO_CHECKBOARD,
        ]
        for name in stereo_items:
            action = self.stereo_menu.addAction(name)
            action.setCheckable(True)
            stereo_group.addAction(action)
            action.triggered.connect(lambda checked, n=name: self._on_stereo(n))

    def _on_raycasting_type(self, name):
        Publisher.sendMessage("Load raycasting preset", preset_name=name)
        self.tools_submenu.setEnabled(name != const.RAYCASTING_OFF_LABEL)

    def _on_raycasting_tool(self, name, checked):
        flag = 1 if checked else 0
        Publisher.sendMessage("Enable raycasting tool", tool_name=name, flag=flag)

    def _on_view(self, view_id):
        icon = QIcon(QPixmap(ID_TO_BMP[view_id][1]))
        self.button_view.setIcon(icon)
        Publisher.sendMessage("Set volume view angle", view=view_id)

    def _on_slice_plane(self, label, checked):
        if checked:
            Publisher.sendMessage("Enable plane", plane_label=label)
        else:
            Publisher.sendMessage("Disable plane", plane_label=label)

    def _on_stereo(self, mode):
        Publisher.sendMessage("Set stereo mode", mode=mode)

    def DisableVolumeCutMenu(self):
        self.tools_submenu.setEnabled(False)
        first_action = next(iter(self.tool_actions.values()), None)
        if first_action is not None:
            first_action.setChecked(False)

    def BuildRaycastingMenu(self):
        presets = []
        for folder in const.RAYCASTING_PRESETS_FOLDERS:
            presets += [
                filename.split(".")[0]
                for filename in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, filename))
            ]

    def Uncheck(self):
        for action in self.slice_plane_actions.values():
            if action.isChecked():
                action.setChecked(False)

    def ChangeButtonColour(self, colour):
        colour = [i * 255 for i in colour]
        self.button_colour.SetColour(colour)

    def OnSelectColour(self, colour_values):
        colour = [i / 255.0 for i in colour_values]
        Publisher.sendMessage("Change volume viewer background colour", colour=colour)
