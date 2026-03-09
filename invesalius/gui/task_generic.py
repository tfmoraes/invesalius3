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


from PySide6.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from invesalius.i18n import tr as _


class TaskPanel(QWidget):
    """
    This panel works as a "frame", drawing a white margin arround
    the panel that really matters (InnerTaskPanel).
    """

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

        self.__init_gui()
        self.__bind_events()
        self.__bind_qt_events()

    def __init_gui(self):
        link_test = QPushButton(_("Testing..."), self)
        link_test.setFlat(True)
        link_test.setStyleSheet("color: black; text-decoration: none;")
        self.link_test = link_test

        sizer = QVBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(link_test)

    def __bind_events(self):
        pass

    def __bind_qt_events(self):
        self.link_test.clicked.connect(self.OnTest)

    def OnTest(self):
        pass
