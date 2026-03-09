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
from functools import partial

from PySide6.QtCore import QSize
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import QDoubleSpinBox, QPushButton

import invesalius.constants as const


def _color_stylesheet(rgb_tuple):
    r, g, b = rgb_tuple
    return f"background-color: rgb({r}, {g}, {b});"


class OrderedFiducialButtons:
    def __init__(
        self,
        parent,
        fiducial_definitions,
        is_fiducial_set,
        get_fiducial_coord=None,
        set_actor_colors=None,
        order=None,
    ):
        """
        Class to initialize fiducials GUI and to keep track of the order to set fiducials.

        :param parent: parent for Qt elements
        :param fiducial_definitions: const.OBJECT_FIDUCIALS or const.TRACKER_FIDUCIALS
        :param is_fiducial_set: Function taking fiducial index as parameter, returning True if that
                                fiducial is set, False otherwise.
        :param get_fiducial_coord: Function to retrieve value for spin boxes. Takes fiducial index and
                                   coordinate index as parameters, returns value.
        :param set_actor_colors: Function taking fiducial index and float color as parameter, changing
                                 the color of relevant actors to match the fiducial index.
        :param order: list of indices representing default order to record fiducials
        """
        count = len(fiducial_definitions)
        self.is_fiducial_set = is_fiducial_set
        self.get_fiducial_coord = get_fiducial_coord
        self.set_actor_colors = set_actor_colors
        self.order: list[int] = order or list(range(count))

        self.buttons: list[QPushButton] = []
        self.numctrls: list[list[QDoubleSpinBox]] = []

        self.focused_index: int | None = None

        self.COLOR_NOT_SET = 0
        self.COLOR_FOCUSED = 1
        self.COLOR_SET = 2

        for n, fiducial in enumerate(fiducial_definitions):
            label = fiducial["label"]
            tip = fiducial["tip"]

            fm = QFontMetrics(parent.font())
            w = fm.horizontalAdvance("M" * len(label))
            h = fm.height()
            ctrl = QPushButton("", parent)
            ctrl.setFixedSize(QSize(55, h + 5))
            ctrl.setText(label)
            ctrl.setToolTip(tip)
            ctrl.clicked.connect(partial(self._OnButton, n=n))
            self.buttons.append(ctrl)

        for n in range(count):
            coords = []
            for coord_index in range(3):
                spinbox = QDoubleSpinBox(parent)
                spinbox.setDecimals(1)
                spinbox.setRange(-9999.9, 9999.9)
                spinbox.hide()
                coords.append(spinbox)
            self.numctrls.append(coords)

        self.Update()

    def __getitem__(self, n):
        return self.buttons[n]

    def __iter__(self):
        return iter(self.buttons)

    @property
    def focused(self):
        if self.focused_index is None:
            return None
        else:
            return self.buttons[self.focused_index]

    @focused.setter
    def focused(self, new_focus):
        if new_focus is None:
            self.focused_index = None
            return

        for n, button in enumerate(self.buttons):
            if new_focus is button:
                self.focused_index = n
                return
        raise ValueError

    def _TrySetActorColors(self, n, color_float):
        if self.set_actor_colors is not None:
            self.set_actor_colors(n, color_float)

    def _SetColor(self, n, color):
        button = self.buttons[n]
        if color == self.COLOR_SET:
            button.setStyleSheet(_color_stylesheet(const.GREEN_COLOR_RGB))
            self._TrySetActorColors(n, const.GREEN_COLOR_FLOAT)
        elif color == self.COLOR_FOCUSED:
            button.setStyleSheet(_color_stylesheet(const.YELLOW_COLOR_RGB))
            self._TrySetActorColors(n, const.YELLOW_COLOR_FLOAT)
        else:
            button.setStyleSheet(_color_stylesheet(const.RED_COLOR_RGB))
            self._TrySetActorColors(n, const.RED_COLOR_FLOAT)

    def _RefreshColors(self):
        for n, button in enumerate(self.buttons):
            if self.is_fiducial_set(n):
                self._SetColor(n, self.COLOR_SET)
            else:
                self._SetColor(n, self.COLOR_NOT_SET)
        if self.focused is not None:
            self._SetColor(self.focused_index, self.COLOR_FOCUSED)

    def _UpdateControls(self):
        if self.get_fiducial_coord is None:
            return

        for n, element in enumerate(self.numctrls):
            for i, numctrl in enumerate(element):
                value = self.get_fiducial_coord(n, i)
                numctrl.setValue(value)

    def _UpdateControl(self, n):
        if self.get_fiducial_coord is None:
            return

        for i, numctrl in enumerate(self.numctrls[n]):
            value = self.get_fiducial_coord(n, i)
            numctrl.setValue(value)

    def Update(self):
        self._UpdateControls()
        self._RefreshColors()

    def FocusNext(self):
        for n in self.order:
            if not self.is_fiducial_set(n):
                self.Focus(n)
                break

    def ClearFocus(self):
        if self.focused is not None:
            self._SetColor(self.focused_index, self.COLOR_NOT_SET)
            self.focused = None

    def _OnButton(self, n):
        self.Focus(n)

    def Focus(self, n):
        self.ClearFocus()
        self.focused = self.buttons[n]
        self._SetColor(self.focused_index, self.COLOR_FOCUSED)

    def SetFocused(self):
        self._SetColor(self.focused_index, self.COLOR_SET)
        self._UpdateControl(self.focused_index)
        self.focused = None
        self.FocusNext()

    def Set(self, n):
        self.Focus(n)
        self.SetFocused()

    def Unset(self, n):
        self._SetColor(n, self.COLOR_NOT_SET)
        self.FocusNext()
