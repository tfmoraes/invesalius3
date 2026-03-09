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

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

import invesalius.gui.task_efield as efield
import invesalius.gui.task_exporter as exporter
import invesalius.gui.task_fmrisupport as fmrisupport
import invesalius.gui.task_importer as importer
import invesalius.gui.task_slice as slice_
import invesalius.gui.task_surface as surface
import invesalius.gui.task_tractography as tractography
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher


class TaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        inner_panel = InnerTaskPanel(self)

        sizer = QHBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(inner_panel)


class InnerTaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet("background-color: white;")

        fold_panel = FoldPanel(self)

        main_sizer = QVBoxLayout(self)
        main_sizer.setContentsMargins(5, 0, 5, 5)
        main_sizer.addWidget(fold_panel, 1)


class FoldPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        inner_panel = InnerFoldPanel(self)

        sizer = QVBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(inner_panel)


class InnerFoldPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        fold_panel = QToolBox(self)

        self.enable_items = []
        self.overwrite = False

        tasks = [
            (_("Load data"), importer.TaskPanel),
            (_("Select region of interest"), slice_.TaskPanel),
            (_("Configure 3D surface"), surface.TaskPanel),
            (_("Export data"), exporter.TaskPanel),
            (_("Tractography"), tractography.TaskPanel),
            (_("E-Field"), efield.TaskPanel),
            (_("fMRI support"), fmrisupport.TaskPanel),
        ]

        self.__id_slice = -1
        self.__id_surface = -1

        for i, (name, panel_cls) in enumerate(tasks):
            page = panel_cls(fold_panel)
            index = fold_panel.addItem(page, "%d. %s" % (i + 1, name))

            if i != 0:
                self.enable_items.append(index)

            if name == _("Select region of interest"):
                self.__id_slice = index
            elif name == _("Configure 3D surface"):
                self.__id_surface = index

        fold_panel.setCurrentIndex(0)
        self.fold_panel = fold_panel

        sizer = QVBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(fold_panel, 1)
        self.SetStateProjectClose()
        self.__bind_events()

    def __bind_events(self):
        self.fold_panel.currentChanged.connect(self.OnFoldPressCaption)
        Publisher.subscribe(self.OnEnableState, "Enable state project")
        Publisher.subscribe(self.OnOverwrite, "Create surface from index")
        Publisher.subscribe(self.OnFoldSurface, "Fold surface task")
        Publisher.subscribe(self.OnFoldExport, "Fold export task")

    def SetStateProjectClose(self):
        self.fold_panel.setCurrentIndex(0)
        for idx in self.enable_items:
            self.fold_panel.setItemEnabled(idx, False)

    def SetStateProjectOpen(self):
        self.fold_panel.setCurrentIndex(1)
        for idx in self.enable_items:
            self.fold_panel.setItemEnabled(idx, True)

    def OnFoldPressCaption(self, index):
        if index == self.__id_slice:
            Publisher.sendMessage("Retrieve task slice style")
            Publisher.sendMessage("Fold mask page")
        elif index == self.__id_surface:
            Publisher.sendMessage("Fold surface page")
        else:
            Publisher.sendMessage("Disable task slice style")

    def OnOverwrite(self, surface_parameters):
        self.overwrite = surface_parameters["options"]["overwrite"]

    def OnFoldSurface(self):
        if not self.overwrite:
            self.fold_panel.setCurrentIndex(2)

    def OnFoldExport(self):
        self.fold_panel.setCurrentIndex(3)

    def OnEnableState(self, state):
        if state:
            self.SetStateProjectOpen()
        else:
            self.SetStateProjectClose()
