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
import pathlib
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.gui.dialogs as dlgs
import invesalius.project as proj
import invesalius.session as ses
from invesalius import inv_paths
from invesalius.data.slice_ import Slice
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher

WILDCARD_SAVE_3D_FILTERS = [
    "Inventor (*.iv)",
    "PLY (*.ply)",
    "Renderman (*.rib)",
    "STL (*.stl)",
    "STL ASCII (*.stl)",
    "VRML (*.vrml)",
    "VTK PolyData (*.vtp)",
    "Wavefront (*.obj)",
    "X3D (*.x3d)",
]
WILDCARD_SAVE_3D = ";;".join(WILDCARD_SAVE_3D_FILTERS)

INDEX_TO_TYPE_3D = {
    0: const.FILETYPE_IV,
    1: const.FILETYPE_PLY,
    2: const.FILETYPE_RIB,
    3: const.FILETYPE_STL,
    4: const.FILETYPE_STL_ASCII,
    5: const.FILETYPE_VRML,
    6: const.FILETYPE_VTP,
    7: const.FILETYPE_OBJ,
    8: const.FILETYPE_X3D,
}
INDEX_TO_EXTENSION = {
    0: "iv",
    1: "ply",
    2: "rib",
    3: "stl",
    4: "stl",
    5: "vrml",
    6: "vtp",
    7: "obj",
    8: "x3d",
}

WILDCARD_SAVE_2D_FILTERS = [
    "BMP (*.bmp)",
    "JPEG (*.jpg)",
    "PNG (*.png)",
    "PostScript (*.ps)",
    "Povray (*.pov)",
    "TIFF (*.tiff)",
]
WILDCARD_SAVE_2D = ";;".join(WILDCARD_SAVE_2D_FILTERS)

INDEX_TO_TYPE_2D = {
    0: const.FILETYPE_BMP,
    1: const.FILETYPE_JPG,
    2: const.FILETYPE_PNG,
    3: const.FILETYPE_PS,
    4: const.FILETYPE_POV,
    5: const.FILETYPE_OBJ,
}

WILDCARD_SAVE_MASK = "VTK ImageData (*.vti)"


class TaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        inner_panel = InnerTaskPanel(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(7, 0, 7, 7)
        layout.addWidget(inner_panel, 1)


class InnerTaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet("background-color: white;")

        tooltip = _("Export InVesalius screen to an image file")
        link_export_picture = QPushButton(_("Export picture..."))
        link_export_picture.setFlat(True)
        link_export_picture.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; color: black; }"
        )
        link_export_picture.setToolTip(tooltip)
        link_export_picture.clicked.connect(self.OnLinkExportPicture)

        tooltip = _("Export 3D surface")
        link_export_surface = QPushButton(_("Export 3D surface..."))
        link_export_surface.setFlat(True)
        link_export_surface.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; color: black; }"
        )
        link_export_surface.setToolTip(tooltip)
        link_export_surface.clicked.connect(self.OnLinkExportSurface)

        BMP_EXPORT_SURFACE = QPixmap(
            os.path.join(inv_paths.ICON_DIR, "surface_export_original.png")
        ).scaled(25, 25, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        BMP_TAKE_PICTURE = QPixmap(
            os.path.join(inv_paths.ICON_DIR, "tool_photo_original.png")
        ).scaled(25, 25, Qt.KeepAspectRatio, Qt.SmoothTransformation)

        button_picture = QPushButton("")
        button_picture.setIcon(QIcon(BMP_TAKE_PICTURE))
        button_picture.setFlat(True)
        button_picture.clicked.connect(self.OnLinkExportPicture)
        self.button_picture = button_picture

        button_surface = QPushButton("")
        button_surface.setIcon(QIcon(BMP_EXPORT_SURFACE))
        button_surface.setFlat(True)
        button_surface.clicked.connect(self.OnLinkExportSurface)

        fixed_layout = QGridLayout()
        fixed_layout.setHorizontalSpacing(2)
        fixed_layout.setVerticalSpacing(0)
        fixed_layout.setColumnStretch(0, 1)
        fixed_layout.addWidget(link_export_picture, 0, 0)
        fixed_layout.addWidget(button_picture, 0, 1)
        fixed_layout.addWidget(link_export_surface, 1, 0)
        fixed_layout.addWidget(button_surface, 1, 1)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(fixed_layout)

        self.__init_menu()

    def __init_menu(self):
        menu = QMenu(self)
        self.id_to_name = {
            const.AXIAL: _("Axial slice"),
            const.CORONAL: _("Coronal slice"),
            const.SAGITAL: _("Sagittal slice"),
            const.VOLUME: _("Volume"),
        }

        for id_val in self.id_to_name:
            action = menu.addAction(self.id_to_name[id_val])
            action.setData(id_val)

        self.menu_picture = menu
        menu.triggered.connect(self.OnMenuPicture)

    def OnMenuPicture(self, action):
        id_val = action.data()
        value = dlgs.ExportPicture(self.id_to_name[id_val])
        if value:
            filename, filetype = value
            Publisher.sendMessage(
                "Export picture to file", orientation=id_val, filename=filename, filetype=filetype
            )

    def OnLinkExportPicture(self):
        pos = self.button_picture.mapToGlobal(self.button_picture.rect().bottomLeft())
        self.menu_picture.exec_(pos)

    def OnLinkExportMask(self):
        project = proj.Project()
        if sys.platform == "win32":
            project_name = project.name
        else:
            project_name = project.name + ".vti"

        dlg = QFileDialog(None, "Save mask as...", "")
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        dlg.setNameFilters([WILDCARD_SAVE_MASK])
        dlg.selectFile(project_name)

        if dlg.exec() == QFileDialog.Accepted:
            selected_files = dlg.selectedFiles()
            if selected_files:
                filename = selected_files[0]
                extension = "vti"
                if sys.platform != "win32":
                    if filename.split(".")[-1] != extension:
                        filename = filename + "." + extension
                filetype = const.FILETYPE_IMAGEDATA
                Publisher.sendMessage("Export mask to file", filename=filename, filetype=filetype)

    def OnLinkExportSurface(self):
        "OnLinkExportSurface"
        project = proj.Project()
        n_surface = 0

        for index in project.surface_dict:
            if project.surface_dict[index].is_shown:
                n_surface += 1

        if n_surface:
            if sys.platform == "win32":
                project_name = pathlib.Path(project.name).stem
            else:
                project_name = pathlib.Path(project.name).stem + ".stl"

            session = ses.Session()
            last_directory = session.GetConfig("last_directory_3d_surface", "")

            dlg_message = _("Save 3D surface as...")

            convert_to_world = False
            if Slice().has_affine():
                space_dlg = QDialog(self)
                space_dlg.setWindowTitle(_("Export in"))
                space_layout = QVBoxLayout(space_dlg)

                group = QGroupBox(_("Export in"))
                group_layout = QVBoxLayout(group)
                radio_buttons = []
                for i, choice in enumerate(const.SURFACE_SPACE_CHOICES):
                    rb = QRadioButton(choice)
                    if i == 0:
                        rb.setChecked(True)
                    radio_buttons.append(rb)
                    group_layout.addWidget(rb)

                btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
                btn_box.accepted.connect(space_dlg.accept)
                btn_box.rejected.connect(space_dlg.reject)

                space_layout.addWidget(group)
                space_layout.addWidget(btn_box)

                if space_dlg.exec() == QDialog.Accepted:
                    selection = 0
                    for i, rb in enumerate(radio_buttons):
                        if rb.isChecked():
                            selection = i
                            break
                    convert_to_world = selection == const.SURFACE_SPACE_WORLD
                else:
                    return

            dlg = QFileDialog(None, dlg_message, last_directory)
            dlg.setAcceptMode(QFileDialog.AcceptSave)
            dlg.setNameFilters(WILDCARD_SAVE_3D_FILTERS)
            dlg.selectNameFilter(WILDCARD_SAVE_3D_FILTERS[3])
            dlg.selectFile(project_name)

            if dlg.exec() == QFileDialog.Accepted:
                selected_filter = dlg.selectedNameFilter()
                filetype_index = (
                    WILDCARD_SAVE_3D_FILTERS.index(selected_filter)
                    if selected_filter in WILDCARD_SAVE_3D_FILTERS
                    else 3
                )
                filetype = INDEX_TO_TYPE_3D[filetype_index]
                selected_files = dlg.selectedFiles()
                filename = selected_files[0] if selected_files else ""
                extension = INDEX_TO_EXTENSION[filetype_index]
                if sys.platform != "win32":
                    if filename.split(".")[-1] != extension:
                        filename = filename + "." + extension

                if filename:
                    last_directory = os.path.split(filename)[0]
                    session.SetConfig("last_directory_3d_surface", last_directory)

                Publisher.sendMessage(
                    "Export surface to file",
                    filename=filename,
                    filetype=filetype,
                    convert_to_world=convert_to_world,
                )
        else:
            QMessageBox.information(
                None,
                "InVesalius 3",
                _("You need to create a surface and make it ") + _("visible before exporting it."),
            )

    def OnLinkRequestRP(self):
        pass

    def OnLinkReport(self):
        pass
