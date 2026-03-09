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

from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
from invesalius import inv_paths
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher


class TaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        inner_panel = InnerTaskPanel(self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(7, 0, 7, 7)
        layout.addWidget(inner_panel, 1)


class InnerTaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet("background-color: white;")

        self.proj_count = 0
        self.float_hyper_list = []

        tooltip = _("Select DICOM files to be reconstructed")
        link_import_local = QPushButton(_("Import DICOM images..."))
        link_import_local.setFlat(True)
        link_import_local.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; color: black; }"
        )
        link_import_local.setToolTip(tooltip)
        link_import_local.clicked.connect(self.OnLinkImport)

        tooltip = _("Select NIFTI files to be reconstructed")
        link_import_nifti = QPushButton(_("Import NIFTI images..."))
        link_import_nifti.setFlat(True)
        link_import_nifti.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; color: black; }"
        )
        link_import_nifti.setToolTip(tooltip)
        link_import_nifti.clicked.connect(self.OnLinkImportNifti)

        tooltip = _("Open an existing InVesalius project...")
        link_open_proj = QPushButton(_("Open an existing project..."))
        link_open_proj.setFlat(True)
        link_open_proj.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; color: black; }"
        )
        link_open_proj.setToolTip(tooltip)
        link_open_proj.clicked.connect(self.OnLinkOpenProject)

        BMP_IMPORT = QPixmap(str(inv_paths.ICON_DIR.joinpath("file_import_original.png")))
        BMP_OPEN_PROJECT = QPixmap(str(inv_paths.ICON_DIR.joinpath("file_open_original.png")))

        button_import_local = QPushButton("")
        button_import_local.setIcon(QIcon(BMP_IMPORT))
        button_import_local.setFlat(True)
        button_import_local.clicked.connect(self.OnLinkImport)

        button_import_nifti = QPushButton("")
        button_import_nifti.setIcon(QIcon(BMP_IMPORT))
        button_import_nifti.setFlat(True)
        button_import_nifti.clicked.connect(self.OnLinkImportNifti)

        button_open_proj = QPushButton("")
        button_open_proj.setIcon(QIcon(BMP_OPEN_PROJECT))
        button_open_proj.setFlat(True)
        button_open_proj.clicked.connect(self.OnLinkOpenProject)

        fixed_layout = QGridLayout()
        fixed_layout.setHorizontalSpacing(2)
        fixed_layout.setVerticalSpacing(0)
        fixed_layout.setColumnStretch(0, 1)
        fixed_layout.addWidget(link_import_local, 0, 0)
        fixed_layout.addWidget(button_import_local, 0, 1)
        fixed_layout.addWidget(link_import_nifti, 1, 0)
        fixed_layout.addWidget(button_import_nifti, 1, 1)
        fixed_layout.addWidget(link_open_proj, 2, 0)
        fixed_layout.addWidget(button_open_proj, 2, 1)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.addLayout(fixed_layout)

        self.TestLoadProjects2()

    def TestLoadProjects2(self):
        import invesalius.session as ses

        session = ses.Session()
        recent_projects = session.GetConfig("recent_projects")

        for path, filename in recent_projects:
            self.LoadProject(filename, path)

    def TestLoadProjects(self):
        self.LoadProject("test1.inv3", "/Volumes/file/inv3")
        self.LoadProject("test2.inv3", "/Volumes/file/inv3")
        self.LoadProject("test3.inv3", "/Volumes/file/inv3")

    def LoadProject(self, proj_name="Unnamed", proj_dir=""):
        """
        Load into user interface name of invesalius.project into import task panel.
        Can be called 3 times in sequence.
        Call UnloadProjects to empty it.
        """
        proj_path = os.path.join(proj_dir, proj_name)

        if self.proj_count < 3:
            self.proj_count += 1

            label = "     " + str(self.proj_count) + ". " + proj_name

            proj_link = QPushButton(label)
            proj_link.setFlat(True)
            proj_link.setStyleSheet("QPushButton { text-align: left; color: black; }")
            proj_link.clicked.connect(lambda checked=False, p=proj_path: self.OpenProject(p))

            self.main_layout.addWidget(proj_link, 1)
            self.float_hyper_list.append(proj_link)

    def OnLinkImport(self):
        self.ImportDicom()

    def OnLinkImportNifti(self):
        self.ImportNifti()

    def OnLinkImportPACS(self):
        self.ImportPACS()

    def OnLinkOpenProject(self):
        self.OpenProject()

    def ImportPACS(self):
        print("TODO: Send Signal - Import DICOM files from PACS")

    def ImportDicom(self):
        Publisher.sendMessage("Show import directory dialog")

    def ImportNifti(self):
        Publisher.sendMessage("Show import other files dialog", id_type=const.ID_NIFTI_IMPORT)

    def OpenProject(self, path=None):
        if path:
            Publisher.sendMessage("Open recent project", filepath=path)
        else:
            Publisher.sendMessage("Show open project dialog")

    def SaveAsProject(self):
        Publisher.sendMessage("Show save dialog", save_as=True)

    def SaveProject(self):
        Publisher.sendMessage("Show save dialog", save_as=False)

    def CloseProject(self):
        Publisher.sendMessage("Close Project")

    def UnloadProjects(self):
        """
        Unload all projects from interface into import task panel.
        This will be called when the current project is closed.
        """
        for hyper in self.float_hyper_list:
            self.main_layout.removeWidget(hyper)
            hyper.deleteLater()

        self.proj_count = 0
        self.float_hyper_list = []
