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
import decimal
from typing import Any, Optional, Union

from PySide6.QtCore import QEvent, Qt, Signal
from PySide6.QtGui import QFontMetrics, QWheelEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


class InvSpinCtrl(QWidget):
    valueChanged = Signal()

    def __init__(
        self,
        parent: QWidget,
        id: int = -1,
        value: int = 0,
        min_value: int = 1,
        max_value: int = 100,
        increment: int = 1,
        spin_button: bool = True,
        unit: str = "",
        size=None,
        style: int = 0,
    ):
        super().__init__(parent)

        self._textctrl = QLineEdit(self)
        self._textctrl.setAlignment(Qt.AlignRight)
        self._spinbtn_up: Optional[QToolButton] = None
        self._spinbtn_down: Optional[QToolButton] = None

        if spin_button:
            self._spinbtn_up = QToolButton(self)
            self._spinbtn_up.setArrowType(Qt.UpArrow)
            self._spinbtn_up.setAutoRepeat(True)
            self._spinbtn_down = QToolButton(self)
            self._spinbtn_down.setArrowType(Qt.DownArrow)
            self._spinbtn_down.setAutoRepeat(True)

        self._value = 0
        self._last_value = 0
        self._min_value = 0
        self._max_value = 100
        self._increment = 1

        self.unit = unit

        self.SetMin(min_value)
        self.SetMax(max_value)
        self.SetValue(value)
        self.SetIncrement(increment)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._textctrl, 1)

        if self._spinbtn_up:
            spin_layout = QVBoxLayout()
            spin_layout.setContentsMargins(0, 0, 0, 0)
            spin_layout.setSpacing(0)
            spin_layout.addWidget(self._spinbtn_up)
            spin_layout.addWidget(self._spinbtn_down)
            layout.addLayout(spin_layout)

        self.setLayout(layout)

        self.__bind_events()

    def __bind_events(self) -> None:
        self._textctrl.installEventFilter(self)
        if self._spinbtn_up:
            self._spinbtn_up.clicked.connect(self.OnSpinUp)
            self._spinbtn_down.clicked.connect(self.OnSpinDown)

    def eventFilter(self, obj, event):
        if obj is self._textctrl and event.type() == QEvent.FocusOut:
            self.OnKillFocus()
        return super().eventFilter(obj, event)

    def SetIncrement(self, increment: int) -> None:
        self._increment = increment

    def SetMin(self, min_value: int) -> None:
        self._min_value = min_value
        self.SetValue(self._value)
        self.CalcSizeFromTextSize()

    def SetMax(self, max_value: int) -> None:
        self._max_value = max_value
        self.SetValue(self._value)
        self.CalcSizeFromTextSize()

    def SetRange(self, min_value: int, max_value: int) -> None:
        self.SetMin(min_value)
        self.SetMax(max_value)

    def GetValue(self) -> int:
        return self._value

    def SetValue(self, value: Union[float, str]) -> None:
        try:
            value = int(value)
        except (ValueError, TypeError):
            value = self._last_value

        if value < self._min_value:
            value = self._min_value

        if value > self._max_value:
            value = self._max_value

        self._value = value
        self._textctrl.setText(f"{self._value} {self.unit}")
        self._last_value = self._value

    def GetUnit(self) -> str:
        return self.unit

    def SetUnit(self, unit: str) -> None:
        self.unit = unit
        self.SetValue(self.GetValue())

    def CalcSizeFromTextSize(self, text: Optional[str] = None) -> None:
        if text is None:
            text = "M" * max(len(str(self._max_value)), len(str(self._min_value)), 5)

        fm = QFontMetrics(self.font())
        width = fm.horizontalAdvance(text)
        height = fm.height()

        if self._spinbtn_up:
            btn_width = self._spinbtn_up.sizeHint().width()
            width += btn_width

        self.setMinimumSize(width, height)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self.SetValue(self.GetValue() + self._increment)
        else:
            self.SetValue(self.GetValue() - self._increment)
        self.raise_event()

    def OnKillFocus(self) -> None:
        value = self._textctrl.text()
        self.SetValue(value)
        self.raise_event()

    def OnSpinDown(self) -> None:
        self.SetValue(self.GetValue() - self._increment)
        self.raise_event()

    def OnSpinUp(self) -> None:
        self.SetValue(self.GetValue() + self._increment)
        self.raise_event()

    def raise_event(self) -> None:
        self.valueChanged.emit()


class InvFloatSpinCtrl(QWidget):
    valueChanged = Signal()

    def __init__(
        self,
        parent: QWidget,
        id: int = -1,
        value: float = 0.0,
        min_value: float = 1.0,
        max_value: float = 100.0,
        increment: float = 0.1,
        digits: int = 1,
        spin_button: bool = True,
        size=None,
        style: int = 0,
    ):
        super().__init__(parent)

        self._textctrl = QLineEdit(self)
        self._textctrl.setAlignment(Qt.AlignRight)
        self._spinbtn_up: Optional[QToolButton] = None
        self._spinbtn_down: Optional[QToolButton] = None

        if spin_button:
            self._spinbtn_up = QToolButton(self)
            self._spinbtn_up.setArrowType(Qt.UpArrow)
            self._spinbtn_up.setAutoRepeat(True)
            self._spinbtn_down = QToolButton(self)
            self._spinbtn_down.setArrowType(Qt.DownArrow)
            self._spinbtn_down.setAutoRepeat(True)

        self._digits = digits
        self._dec_context = decimal.Context(prec=digits)

        self._value = decimal.Decimal("0", self._dec_context)
        self._last_value = self._value
        self._min_value = decimal.Decimal("0", self._dec_context)
        self._max_value = decimal.Decimal("100", self._dec_context)
        self._increment = decimal.Decimal("0.1", self._dec_context)

        self.SetIncrement(increment)
        self.SetMin(min_value)
        self.SetMax(max_value)
        self.SetValue(value)

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self._textctrl, 1)

        if self._spinbtn_up:
            spin_layout = QVBoxLayout()
            spin_layout.setContentsMargins(0, 0, 0, 0)
            spin_layout.setSpacing(0)
            spin_layout.addWidget(self._spinbtn_up)
            spin_layout.addWidget(self._spinbtn_down)
            layout.addLayout(spin_layout)

        self.setLayout(layout)

        self.__bind_events()

    def __bind_events(self) -> None:
        self._textctrl.installEventFilter(self)
        if self._spinbtn_up:
            self._spinbtn_up.clicked.connect(self.OnSpinUp)
            self._spinbtn_down.clicked.connect(self.OnSpinDown)

    def eventFilter(self, obj, event):
        if obj is self._textctrl and event.type() == QEvent.FocusOut:
            self.OnKillFocus()
        return super().eventFilter(obj, event)

    def _to_decimal(self, value: Union[decimal.Decimal, float, str]) -> decimal.Decimal:
        if not isinstance(value, str):
            value = "{:.{digits}f}".format(value, digits=self._digits)
        return decimal.Decimal(value, self._dec_context)

    def SetDigits(self, digits: int) -> None:
        self._digits = digits
        self._dec_context = decimal.Context(prec=digits)

        self.SetIncrement(self._increment)
        self.SetMin(self._min_value)
        self.SetMax(self._max_value)
        self.SetValue(self._value)

    def SetIncrement(self, increment: Union[decimal.Decimal, float, str]) -> None:
        self._increment = self._to_decimal(increment)

    def SetMin(self, min_value: Union[decimal.Decimal, float, str]) -> None:
        self._min_value = self._to_decimal(min_value)
        self.SetValue(self._value)

    def SetMax(self, max_value: Union[decimal.Decimal, float, str]) -> None:
        self._max_value = self._to_decimal(max_value)
        self.SetValue(self._value)

    def SetRange(
        self,
        min_value: Union[decimal.Decimal, float, str],
        max_value: Union[decimal.Decimal, float, str],
    ) -> None:
        self.SetMin(min_value)
        self.SetMax(max_value)

    def GetValue(self) -> float:
        return float(self._value)

    def SetValue(self, value: Any) -> None:
        try:
            value = self._to_decimal(value)
        except decimal.InvalidOperation:
            value = self._last_value

        if value < self._min_value:
            value = self._min_value

        if value > self._max_value:
            value = self._max_value

        self._value = value
        self._textctrl.setText(f"{self._value}")
        self._last_value = self._value

    def CalcSizeFromTextSize(self, text: Optional[str] = None) -> None:
        if text is None:
            text = "M" * max(len(str(self._max_value)), len(str(self._min_value)))

        fm = QFontMetrics(self.font())
        width = fm.horizontalAdvance(text)
        height = fm.height()

        if self._spinbtn_up:
            btn_width = self._spinbtn_up.sizeHint().width()
            width += btn_width

        self.setMinimumSize(width, height)

    def wheelEvent(self, event: QWheelEvent) -> None:
        delta = event.angleDelta().y()
        if delta > 0:
            self.SetValue(self._value + self._increment)
        else:
            self.SetValue(self._value - self._increment)
        self.raise_event()

    def OnKillFocus(self) -> None:
        value = self._textctrl.text()
        self.SetValue(value)
        self.raise_event()

    def OnSpinDown(self) -> None:
        self.SetValue(self._value - self._increment)
        self.raise_event()

    def OnSpinUp(self) -> None:
        self.SetValue(self._value + self._increment)
        self.raise_event()

    def raise_event(self) -> None:
        self.valueChanged.emit()
