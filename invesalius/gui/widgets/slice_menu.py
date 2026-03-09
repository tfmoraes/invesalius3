# -*- coding: UTF-8 -*-

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
from collections import OrderedDict
from typing import Dict, Optional

from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import QMenu

import invesalius.constants as const
import invesalius.data.slice_ as sl
import invesalius.presets as presets
from invesalius.gui.dialogs import ClutImagedataDialog
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher

PROJECTIONS_ID = OrderedDict(
    (
        (_("Normal"), const.PROJECTION_NORMAL),
        (_("MaxIP"), const.PROJECTION_MaxIP),
        (_("MinIP"), const.PROJECTION_MinIP),
        (_("MeanIP"), const.PROJECTION_MeanIP),
        (_("MIDA"), const.PROJECTION_MIDA),
        (_("Contour MaxIP"), const.PROJECTION_CONTOUR_MIP),
        (_("Contour MIDA"), const.PROJECTION_CONTOUR_MIDA),
    )
)


class SliceMenu(QMenu):
    def __init__(self) -> None:
        QMenu.__init__(self)
        self.cdialog: Optional[ClutImagedataDialog] = None
        self._gen_event = True

        # ------------ Sub menu of the window and level ----------
        submenu_wl = QMenu(_("Window width and level"), self)
        wl_group = QActionGroup(submenu_wl)
        wl_group.setExclusive(True)

        self.wl_default_action = submenu_wl.addAction(_("Default"))
        self.wl_default_action.setCheckable(True)
        self.wl_default_action.setChecked(True)
        wl_group.addAction(self.wl_default_action)

        self.wl_manual_action = submenu_wl.addAction(_("Manual"))
        self.wl_manual_action.setCheckable(True)
        wl_group.addAction(self.wl_manual_action)

        for name in const.WINDOW_LEVEL:
            if not (name == _("Default") or name == _("Manual")):
                action = submenu_wl.addAction(name)
                action.setCheckable(True)
                wl_group.addAction(action)

        # ------------ Sub menu of the pseudo colors ----------------
        submenu_pseudo_colours = QMenu(_("Pseudo color"), self)
        self.pseudo_color_group = QActionGroup(submenu_pseudo_colours)
        self.pseudo_color_group.setExclusive(True)

        self.pseudo_color_default_action = submenu_pseudo_colours.addAction(_("Default "))
        self.pseudo_color_default_action.setCheckable(True)
        self.pseudo_color_default_action.setChecked(True)
        self.pseudo_color_group.addAction(self.pseudo_color_default_action)

        for name in sorted(const.SLICE_COLOR_TABLE):
            if not (name == _("Default ")):
                action = submenu_pseudo_colours.addAction(name)
                action.setCheckable(True)
                self.pseudo_color_group.addAction(action)

        self.plist_presets = presets.get_wwwl_presets()
        for name in sorted(self.plist_presets):
            action = submenu_pseudo_colours.addAction(name)
            action.setCheckable(True)
            self.pseudo_color_group.addAction(action)

        self.custom_color_action = submenu_pseudo_colours.addAction(_("Custom"))
        self.custom_color_action.setCheckable(True)
        self.pseudo_color_group.addAction(self.custom_color_action)

        # --------------- Sub menu of the projection type ---------------------
        self.projection_actions: Dict[int, QAction] = {}
        submenu_projection = QMenu(_("Projection type"), self)
        projection_group = QActionGroup(submenu_projection)
        projection_group.setExclusive(True)
        first_projection = True
        for name in PROJECTIONS_ID:
            action = submenu_projection.addAction(name)
            action.setCheckable(True)
            projection_group.addAction(action)
            self.projection_actions[PROJECTIONS_ID[name]] = action
            if first_projection:
                action.setChecked(True)
                first_projection = False

        # ------------ Sub menu of the image tiling ---------------
        submenu_image_tiling = QMenu(_("Image Tiling"), self)
        tiling_group = QActionGroup(submenu_image_tiling)
        tiling_group.setExclusive(True)
        first_tiling = True
        for name in sorted(const.IMAGE_TILING):
            action = submenu_image_tiling.addAction(name)
            action.setCheckable(True)
            tiling_group.addAction(action)
            if first_tiling:
                action.setChecked(True)
                first_tiling = False

        # Add sub items in the menu
        self.addMenu(submenu_wl)
        self.addMenu(submenu_pseudo_colours)
        self.addMenu(submenu_projection)
        ###self.addMenu(submenu_image_tiling)

        submenu_wl.triggered.connect(self.OnPopup)
        submenu_pseudo_colours.triggered.connect(self.OnPopup)
        submenu_image_tiling.triggered.connect(self.OnPopup)
        submenu_projection.triggered.connect(self.OnPopup)

        self.__bind_events()

    def __bind_events(self) -> None:
        Publisher.subscribe(self.CheckWindowLevelOther, "Check window and level other")
        Publisher.subscribe(self.FirstItemSelect, "Select first item from slice menu")
        Publisher.subscribe(self._close, "Close project data")
        Publisher.subscribe(self._check_projection_menu, "Check projection menu")

    def FirstItemSelect(self) -> None:
        self.wl_default_action.setChecked(True)
        self.pseudo_color_default_action.setChecked(True)

    def CheckWindowLevelOther(self) -> None:
        self.wl_manual_action.setChecked(True)

    def _check_projection_menu(self, projection_id: int) -> None:
        action = self.projection_actions[projection_id]
        action.setChecked(True)

    def OnPopup(self, action: QAction) -> None:
        key = action.text()
        if key in const.WINDOW_LEVEL.keys():
            window, level = const.WINDOW_LEVEL[key]
            Publisher.sendMessage(
                "Bright and contrast adjustment image", window=window, level=level
            )
            Publisher.sendMessage("Update window level value", window=window, level=level)
            Publisher.sendMessage("Update slice viewer")
            Publisher.sendMessage("Render volume viewer")

        elif key in const.SLICE_COLOR_TABLE.keys():
            values = const.SLICE_COLOR_TABLE[key]
            Publisher.sendMessage("Change colour table from background image", values=values)
            Publisher.sendMessage("Update slice viewer")
            self.HideClutDialog()
            self._gen_event = True

        elif key in self.plist_presets:
            values = presets.get_wwwl_preset_colours(self.plist_presets[key])
            Publisher.sendMessage(
                "Change colour table from background image from plist", values=values
            )
            Publisher.sendMessage("Update slice viewer")
            self.HideClutDialog()
            self._gen_event = True

        elif key in const.IMAGE_TILING.keys():
            values = const.IMAGE_TILING[key]
            Publisher.sendMessage("Set slice viewer layout", layout=values)
            Publisher.sendMessage("Update slice viewer")

        elif key in PROJECTIONS_ID:
            pid = PROJECTIONS_ID[key]
            Publisher.sendMessage("Set projection type", projection_id=pid)
            Publisher.sendMessage("Reload actual slice")

        elif key == _("Custom"):
            if self.cdialog is None:
                slc = sl.Slice()
                histogram = slc.histogram
                init = int(slc.matrix.min())
                end = int(slc.matrix.max())
                nodes = slc.nodes
                self.cdialog = ClutImagedataDialog(histogram, init, end, nodes)
                self.cdialog.show()
            else:
                self.cdialog.setVisible(self._gen_event)
            action.setChecked(True)
            self._gen_event = False

    def HideClutDialog(self) -> None:
        if self.cdialog:
            self.cdialog.hide()

    def _close(self) -> None:
        if self.cdialog:
            self.cdialog.close()
            self.cdialog = None
