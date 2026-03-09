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

from PySide6.QtCore import QSize
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as constants
from invesalius import inv_paths
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
        self.setStyleSheet("background-color: white;")

        self.proj_count = 0
        self.float_hyper_list = []

        txt_measure = QLabel(_("Measure"), self)
        txt_measure.setToolTip(_("Measure distances"))

        txt_annotation = QPushButton(_("Add text annotations"), self)
        txt_annotation.setFlat(True)
        txt_annotation.setStyleSheet("color: black; text-decoration: none;")
        txt_annotation.setToolTip(_("Add text annotations"))
        txt_annotation.clicked.connect(self.OnTextAnnotation)

        BMP_ANNOTATE = QPixmap(os.path.join(inv_paths.ICON_DIR, "annotation.png"))
        BMP_ANGLE = QPixmap(os.path.join(inv_paths.ICON_DIR, "measure_angle_original.png"))
        BMP_DISTANCE = QPixmap(os.path.join(inv_paths.ICON_DIR, "measure_line_original.png"))

        icon_size = QSize(25, 25)

        button_measure_linear = QPushButton(self)
        button_measure_linear.setIcon(QIcon(BMP_DISTANCE))
        button_measure_linear.setIconSize(icon_size)
        button_measure_linear.setFlat(True)
        button_measure_linear.clicked.connect(self.OnLinkLinearMeasure)

        button_measure_angular = QPushButton(self)
        button_measure_angular.setIcon(QIcon(BMP_ANGLE))
        button_measure_angular.setIconSize(icon_size)
        button_measure_angular.setFlat(True)
        button_measure_angular.clicked.connect(self.OnLinkAngularMeasure)

        button_annotation = QPushButton(self)
        button_annotation.setIcon(QIcon(BMP_ANNOTATE))
        button_annotation.setIconSize(icon_size)
        button_annotation.setFlat(True)
        button_annotation.clicked.connect(self.OnTextAnnotation)

        sizer = QGridLayout()
        sizer.addWidget(txt_measure, 0, 0)
        sizer.addWidget(button_measure_linear, 0, 1)
        sizer.addWidget(button_measure_angular, 0, 2)
        sizer.addWidget(txt_annotation, 1, 0)
        sizer.addWidget(button_annotation, 1, 2, 2, 1)
        sizer.setColumnStretch(0, 1)

        main_sizer = QVBoxLayout(self)
        main_sizer.setContentsMargins(0, 0, 0, 0)
        main_sizer.addLayout(sizer)
        main_sizer.addStretch()

        self.sizer = main_sizer

    def OnTextAnnotation(self):
        print("TODO: Send Signal - Add text annotation (both 2d and 3d)")

    def OnLinkLinearMeasure(self):
        Publisher.sendMessage("Enable style", style=constants.STATE_MEASURE_DISTANCE)

    def OnLinkAngularMeasure(self):
        Publisher.sendMessage("Enable style", style=constants.STATE_MEASURE_ANGLE)
