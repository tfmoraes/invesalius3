# --------------------------------------------------------------------
# Software:     InVesalius - Software de Reconstrucao 3D de Imagens Medicas
# Copyright:    (C) 2001  Centro de Pesquisas Renato Archer
# Homepage:     http://www.softwarepublico.gov.br
# Contact:      invesalius@cti.gov.br
# License:      GNU - GPL 2 (LICENSE.txt/LICENCA.txt)
# --------------------------------------------------------------------
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
# --------------------------------------------------------------------

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

import invesalius.project as prj
from invesalius import constants as const
from invesalius.gui import utils
from invesalius.i18n import tr as _

ORIENTATION_LABEL = {
    const.AXIAL: _("Axial"),
    const.CORONAL: _("Coronal"),
    const.SAGITAL: _("Sagital"),
}


class ProjectProperties(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle(_("Project Properties"))
        self._init_gui()

    def _init_gui(self):
        project = prj.Project()
        self.name_txt = QLineEdit(project.name)
        self.name_txt.setMinimumWidth(utils.calc_width_needed(self.name_txt, 30))

        modality_txt = QLineEdit(project.modality)
        modality_txt.setReadOnly(True)

        try:
            orientation = ORIENTATION_LABEL[project.original_orientation]
        except KeyError:
            orientation = _("Other")

        orientation_txt = QLineEdit(orientation)
        orientation_txt.setReadOnly(True)

        sx, sy, sz = project.spacing
        spacing_txt_x = QLineEdit(f"{sx:.5}")
        spacing_txt_x.setReadOnly(True)
        spacing_txt_y = QLineEdit(f"{sy:.5}")
        spacing_txt_y.setReadOnly(True)
        spacing_txt_z = QLineEdit(f"{sz:.5}")
        spacing_txt_z.setReadOnly(True)

        name_sizer = QHBoxLayout()
        name_sizer.addWidget(QLabel(_("Name")))
        name_sizer.addWidget(self.name_txt, 1)

        modality_sizer = QHBoxLayout()
        modality_sizer.addWidget(QLabel(_("Modality")))
        modality_sizer.addWidget(modality_txt, 1)

        orientation_sizer = QHBoxLayout()
        orientation_sizer.addWidget(QLabel(_("Orientation")))
        orientation_sizer.addWidget(orientation_txt, 1)

        spacing_sizer = QHBoxLayout()
        spacing_sizer.addWidget(QLabel(_("Spacing")))
        spacing_sizer.addWidget(spacing_txt_x, 1)
        spacing_sizer.addWidget(spacing_txt_y, 1)
        spacing_sizer.addWidget(spacing_txt_z, 1)

        btn_sizer = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_sizer.accepted.connect(self.accept)
        btn_sizer.rejected.connect(self.reject)

        main_sizer = QVBoxLayout(self)
        main_sizer.addLayout(name_sizer)
        main_sizer.addLayout(modality_sizer)
        main_sizer.addLayout(orientation_sizer)
        main_sizer.addLayout(spacing_sizer)
        main_sizer.addWidget(btn_sizer)

        self.setLayout(main_sizer)
