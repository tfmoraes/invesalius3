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
import sys
from typing import Iterable, List, Literal, Optional, Sequence, SupportsInt, Tuple, Union

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QLinearGradient, QPainter, QPen
from PySide6.QtWidgets import QApplication, QHBoxLayout, QStyle, QStyleOptionButton, QWidget

from invesalius.gui.widgets.inv_spinctrl import InvSpinCtrl

ColourType = Union[Tuple[int, int, int], Tuple[int, int, int, int], List[int]]
try:
    _app = QApplication.instance()
    if _app is not None:
        _fm = QFontMetrics(_app.font())
        PUSH_WIDTH = _fm.horizontalAdvance("M") // 2 + 1
        del _fm
    else:
        PUSH_WIDTH = 7
    del _app
except Exception:
    PUSH_WIDTH = 7


class SliderEvent:
    def __init__(self, minRange: int, maxRange: int, minValue: int, maxValue: int):
        self.min_range = minRange
        self.max_range = maxRange
        self.minimun = minValue
        self.maximun = maxValue


class GradientSlider(QWidget):
    # This widget is formed by a gradient background (black-white), two push
    # buttons change the min and max values respectively and a slider which you can drag to
    # change the both min and max values.
    slider_changed = Signal(object)
    slider_changing = Signal(object)

    def __init__(
        self,
        parent: QWidget,
        id: int,
        minRange: int,
        maxRange: int,
        minValue: int,
        maxValue: int,
        colour: Iterable[SupportsInt],
    ):
        # minRange: the minimal value
        # maxrange: the maximum value
        # minValue: the least value in the range
        # maxValue: the most value in the range
        # colour: colour used in this widget.
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setMouseTracking(True)

        self.min_range = minRange
        self.max_range = maxRange
        self.minimun = minValue
        self.maximun = maxValue
        self.selected = 0
        self.max_position: int = 0
        self._delta = 0

        self._gradient_colours: Optional[Sequence[ColourType]] = None

        self.SetColour(colour)
        self.CalculateControlPositions()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        w = self.width()
        h = self.height()

        painter.fillRect(self.rect(), self.palette().window())

        x_init_push1 = self.min_position - PUSH_WIDTH
        x_init_push2 = self.max_position

        width_transparency = self.max_position - self.min_position

        points: Sequence[Tuple[int, int, ColourType, ColourType]] = (
            (0, PUSH_WIDTH, (0, 0, 0), (0, 0, 0)),
            (PUSH_WIDTH, w - PUSH_WIDTH, (0, 0, 0), (255, 255, 255)),
            (w - PUSH_WIDTH, w, (255, 255, 255), (255, 255, 255)),
        )

        p1: float
        p2: float
        for p1, p2, c1, c2 in points:
            gradient = QLinearGradient(p1, 0, p2, h)
            gradient.setColorAt(0.0, QColor(*c1))
            gradient.setColorAt(1.0, QColor(*c2))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(p1, 0, p2 - p1, h))

        if self._gradient_colours is None:
            color = QColor(*self.colour)
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color))
            painter.drawRect(QRectF(self.min_position, 0, width_transparency, h))
        else:
            n_colors = len(self._gradient_colours)
            for i, (c1, c2) in enumerate(zip(self._gradient_colours, self._gradient_colours[1:])):
                p1 = self.min_position + i * width_transparency / n_colors
                p2 = self.min_position + (i + 1) * width_transparency / n_colors
                gradient = QLinearGradient(p1, 0, p2, h)
                gradient.setColorAt(0.0, QColor(*c1))
                gradient.setColorAt(1.0, QColor(*c2))
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen)
                painter.drawRect(QRectF(p1, 0, p2 - p1, h))

        for x in (x_init_push1, x_init_push2):
            opt = QStyleOptionButton()
            opt.initFrom(self)
            opt.rect.setRect(x, 0, PUSH_WIDTH, h)
            opt.state = QStyle.State_Enabled | QStyle.State_Raised
            self.style().drawPrimitive(QStyle.PE_PanelButtonCommand, opt, painter, self)

        painter.end()

    def mousePressEvent(self, event) -> None:
        x = int(event.position().x())
        self.selected = self._is_over_what(x)
        if self.selected == 1:
            self._delta = x - self.min_position
        elif self.selected == 2:
            self._delta = x - self.max_position
        elif self.selected == 3:
            self._delta = x - self.min_position
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self.selected:
            self.selected = 0
            self._generate_event(self.slider_changed)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        x = int(event.position().x())
        w = self.width()

        if not self.selected:
            if self._is_over_what(x) in (1, 2):
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.unsetCursor()

        elif self.selected == 1:
            x -= self._delta
            if x - PUSH_WIDTH < 0:
                x = PUSH_WIDTH
            elif x >= self.max_position:
                x = self.max_position

            value = self._min_position_to_minimun(x)
            self.minimun = value
            self.min_position = x
            self._generate_event(self.slider_changing)
            self.update()

        elif self.selected == 2:
            x -= self._delta
            if x + PUSH_WIDTH > w:
                x = w - PUSH_WIDTH
            elif x < self.min_position:
                x = self.min_position

            value = self._max_position_to_maximun(x)
            self.maximun = value
            self.max_position = x
            self._generate_event(self.slider_changing)
            self.update()

        elif self.selected == 3:
            x -= self._delta
            slider_size = self.max_position - self.min_position
            diff_values = self.maximun - self.minimun

            if x - PUSH_WIDTH < 0:
                min_x = PUSH_WIDTH
                self.minimun = self._min_position_to_minimun(min_x)
                self.maximun = self.minimun + diff_values
                self.CalculateControlPositions()

            elif x + slider_size + PUSH_WIDTH > w:
                max_x = w - PUSH_WIDTH
                self.maximun = self._max_position_to_maximun(max_x)
                self.minimun = self.maximun - diff_values
                self.CalculateControlPositions()

            else:
                min_x = x
                self.minimun = self._min_position_to_minimun(min_x)
                self.maximun = self.minimun + diff_values
                self.CalculateControlPositions()

            self._generate_event(self.slider_changing)
            self.update()
        event.accept()

    def leaveEvent(self, event) -> None:
        if sys.platform != "win32":
            return

        if self.selected == 0:
            return

        x = self.mapFromGlobal(self.cursor().pos()).x()
        w = self.width()

        if self.selected == 1:
            if x - PUSH_WIDTH < 0:
                x = PUSH_WIDTH
            elif x >= self.max_position:
                x = self.max_position
            value = self._min_position_to_minimun(x)
            self.minimun = value
            self.min_position = x

        elif self.selected == 2:
            if x + PUSH_WIDTH > w:
                x = w - PUSH_WIDTH
            elif x < self.min_position:
                x = self.min_position

            value = self._max_position_to_maximun(x)
            self.maximun = value
            self.max_position = x

        self.selected = 0
        self._generate_event(self.slider_changed)

    def resizeEvent(self, event) -> None:
        self.CalculateControlPositions()
        self.update()
        super().resizeEvent(event)

    def CalculateControlPositions(self) -> None:
        """
        Calculates the Min and Max control position based on the size of this
        widget.
        """
        w = self.width()
        window_width = w - 2 * PUSH_WIDTH
        proportion = window_width / float(self.max_range - self.min_range)

        self.min_position = int(round((self.minimun - self.min_range) * proportion)) + PUSH_WIDTH
        self.max_position = int(round((self.maximun - self.min_range) * proportion)) + PUSH_WIDTH

    def _max_position_to_maximun(self, max_position: int) -> int:
        """
        Calculates the min and max value based on the control positions.
        """
        w = self.width()
        window_width = w - 2 * PUSH_WIDTH
        proportion = window_width / float(self.max_range - self.min_range)

        maximun = int(round((max_position - PUSH_WIDTH) / proportion + self.min_range))

        return maximun

    def _min_position_to_minimun(self, min_position: int) -> int:
        w = self.width()
        window_width = w - 2 * PUSH_WIDTH
        proportion = window_width / float(self.max_range - self.min_range)

        minimun = int(round((min_position - PUSH_WIDTH) / proportion + self.min_range))

        return minimun

    def _is_over_what(self, position_x: int) -> Literal[0, 1, 2, 3]:
        # Test if the given position (x) is over some object. Return 1 to first
        # push, 2 to second push, 3 to slide and 0 to nothing.
        if self.min_position - PUSH_WIDTH <= position_x <= self.min_position:
            return 1
        elif self.max_position <= position_x <= self.max_position + PUSH_WIDTH:
            return 2
        elif self.min_position <= position_x <= self.max_position:
            return 3
        else:
            return 0

    def SetColour(self, colour: Iterable[SupportsInt]) -> None:
        self.colour = [int(i) for i in colour]

    def SetGradientColours(self, colors: Optional[Sequence[ColourType]]) -> None:
        self._gradient_colours = colors

    def SetMinRange(self, min_range: int) -> None:
        self.min_range = min_range
        self.CalculateControlPositions()
        self.update()

    def SetMaxRange(self, max_range: int) -> None:
        self.max_range = max_range
        self.CalculateControlPositions()
        self.update()

    def SetMinimun(self, minimun: int) -> None:
        self.minimun = minimun
        self.CalculateControlPositions()
        self.update()

    def SetMaximun(self, maximun: int) -> None:
        self.maximun = maximun
        self.CalculateControlPositions()
        self.update()

    def GetMaxValue(self) -> int:
        return self.maximun

    def GetMinValue(self) -> int:
        return self.minimun

    def _generate_event(self, signal: Signal) -> None:
        evt = SliderEvent(
            self.min_range,
            self.max_range,
            self.minimun,
            self.maximun,
        )
        signal.emit(evt)


class GradientNoSlide(QWidget):
    # This widget is formed by a gradient background (black-white)
    # Unlike GradientSlide, here the widget is used as a colorbar to display
    # the available colors (used in fmri support)
    slider_changed = Signal(object)
    slider_changing = Signal(object)

    def __init__(
        self,
        parent: QWidget,
        id: int,
        minRange: int,
        maxRange: int,
        minValue: int,
        maxValue: int,
        colours: Sequence[ColourType],
    ):
        # minRange: the minimal value
        # maxrange: the maximum value
        # minValue: the least value in the range
        # maxValue: the most value in the range
        # colour: colour used in this widget.
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent)

        self.min_range = minRange
        self.max_range = maxRange
        self.minimun = minValue
        self.maximun = maxValue
        self.selected = 0
        self._delta = 0

        self.min_position = 0
        self.max_position = 0

        self._gradient_colours = colours

    def paintEvent(self, event) -> None:
        painter = QPainter(self)

        w = self.width()
        h = self.height()

        self.max_position = w
        width_transparency = self.max_position - self.min_position

        lengthcolors = len(self._gradient_colours)
        for i, c1 in enumerate(self._gradient_colours):
            p1 = self.min_position + i * width_transparency / lengthcolors
            p2 = self.min_position + (i + 1) * width_transparency / lengthcolors
            gradient = QLinearGradient(p1, 0, p2, h)
            gradient.setColorAt(0.0, QColor(*c1))
            gradient.setColorAt(1.0, QColor(*c1))
            painter.setBrush(QBrush(gradient))
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(p1, 0, p2 - p1, h))

        painter.end()

    def mousePressEvent(self, event) -> None:
        x = int(event.position().x())
        self.selected = self._is_over_what(x)
        if self.selected == 1:
            self._delta = x - self.min_position
        elif self.selected == 2:
            self._delta = x - self.max_position
        elif self.selected == 3:
            self._delta = x - self.min_position
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self.selected:
            self.selected = 0
            self._generate_event(self.slider_changed)
        event.accept()

    def OnMotion(self, x: int) -> None:
        """Motion handling (not connected by default, preserved for subclass use)."""
        w = self.width()

        if not self.selected:
            if self._is_over_what(x) in (1, 2):
                self.setCursor(Qt.SizeHorCursor)
            else:
                self.unsetCursor()

        elif self.selected == 1:
            x -= self._delta
            if x - PUSH_WIDTH < 0:
                x = PUSH_WIDTH
            elif x >= self.max_position:
                x = self.max_position

            value = self._min_position_to_minimun(x)
            self.minimun = value
            self.min_position = x
            self._generate_event(self.slider_changing)
            self.update()

        elif self.selected == 2:
            x -= self._delta
            if x + PUSH_WIDTH > w:
                x = w - PUSH_WIDTH
            elif x < self.min_position:
                x = self.min_position

            value = self._max_position_to_maximun(x)
            self.maximun = value
            self.max_position = x
            self._generate_event(self.slider_changing)
            self.update()

        elif self.selected == 3:
            x -= self._delta
            slider_size = self.max_position - self.min_position
            diff_values = self.maximun - self.minimun

            if x - PUSH_WIDTH < 0:
                min_x = PUSH_WIDTH
                self.minimun = self._min_position_to_minimun(min_x)
                self.maximun = self.minimun + diff_values
                self.CalculateControlPositions()

            elif x + slider_size + PUSH_WIDTH > w:
                max_x = w - PUSH_WIDTH
                self.maximun = self._max_position_to_maximun(max_x)
                self.minimun = self.maximun - diff_values
                self.CalculateControlPositions()

            else:
                min_x = x
                self.minimun = self._min_position_to_minimun(min_x)
                self.maximun = self.minimun + diff_values
                self.CalculateControlPositions()

            self._generate_event(self.slider_changing)
            self.update()

    def CalculateControlPositions(self) -> None:
        """
        Calculates the Min and Max control position based on the size of this
        widget.
        """
        w = self.width()
        window_width = w - 2 * PUSH_WIDTH
        proportion = window_width / float(self.max_range - self.min_range)

        self.min_position = int(round((self.minimun - self.min_range) * proportion)) + PUSH_WIDTH
        self.max_position = int(round((self.maximun - self.min_range) * proportion)) + PUSH_WIDTH

    def _max_position_to_maximun(self, max_position: int) -> int:
        """
        Calculates the min and max value based on the control positions.
        """
        w = self.width()
        window_width = w - 2 * PUSH_WIDTH
        proportion = window_width / float(self.max_range - self.min_range)

        maximun = int(round((max_position - PUSH_WIDTH) / proportion + self.min_range))

        return maximun

    def _min_position_to_minimun(self, min_position: int) -> int:
        w = self.width()
        window_width = w - 2 * PUSH_WIDTH
        proportion = window_width / float(self.max_range - self.min_range)

        minimun = int(round((min_position - PUSH_WIDTH) / proportion + self.min_range))

        return minimun

    def _is_over_what(self, position_x: int) -> Literal[0, 1, 2, 3]:
        # Test if the given position (x) is over some object. Return 1 to first
        # push, 2 to second push, 3 to slide and 0 to nothing.
        if self.min_position - PUSH_WIDTH <= position_x <= self.min_position:
            return 1
        elif self.max_position <= position_x <= self.max_position + PUSH_WIDTH:
            return 2
        elif self.min_position <= position_x <= self.max_position:
            return 3
        else:
            return 0

    def SetGradientColours(self, colors: Sequence[ColourType]) -> None:
        self._gradient_colours = colors

    def SetMinRange(self, min_range: int) -> None:
        self.min_range = min_range
        self.CalculateControlPositions()
        self.update()

    def SetMaxRange(self, max_range: int) -> None:
        self.max_range = max_range
        self.CalculateControlPositions()
        self.update()

    def SetMinimun(self, minimun: int) -> None:
        self.minimun = minimun
        self.CalculateControlPositions()
        self.update()

    def SetMaximun(self, maximun: int) -> None:
        self.maximun = maximun
        self.CalculateControlPositions()
        self.update()

    def GetMaxValue(self) -> int:
        return self.maximun

    def GetMinValue(self) -> int:
        return self.minimun

    def _generate_event(self, signal: Signal) -> None:
        evt = SliderEvent(
            self.min_range,
            self.max_range,
            self.minimun,
            self.maximun,
        )
        signal.emit(evt)


class GradientCtrl(QWidget):
    threshold_changed = Signal(object)
    threshold_changing = Signal(object)

    def __init__(
        self,
        parent: QWidget,
        id: int,
        minRange: int,
        maxRange: int,
        minValue: int,
        maxValue: int,
        colour: Sequence[float],
    ):
        super().__init__(parent)
        self.min_range = minRange
        self.max_range = maxRange
        self.minimun = minValue
        self.maximun = maxValue
        self.colour = colour[:3]
        self.changed = False
        self._draw_controls()
        self._bind_events()
        self.show()

    def _draw_controls(self) -> None:
        self.gradient_slider = GradientSlider(
            self,
            -1,
            self.min_range,
            self.max_range,
            self.minimun,
            self.maximun,
            self.colour,
        )

        self.spin_min = InvSpinCtrl(
            self,
            value=self.minimun,
            min_value=self.min_range,
            max_value=self.max_range,
            spin_button=False,
        )

        self.spin_max = InvSpinCtrl(
            self,
            value=self.maximun,
            min_value=self.min_range,
            max_value=self.max_range,
            spin_button=False,
        )

        self.spin_min.CalcSizeFromTextSize()
        self.spin_max.CalcSizeFromTextSize()

        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        layout.addWidget(self.spin_min, 0)
        layout.addWidget(self.gradient_slider, 1)
        layout.addWidget(self.spin_max, 0)
        self.setLayout(layout)

    def _bind_events(self) -> None:
        self.gradient_slider.slider_changing.connect(self.OnSliding)
        self.gradient_slider.slider_changed.connect(self.OnSlider)

        self.spin_min.valueChanged.connect(self.OnMinMouseWheel)
        self.spin_max.valueChanged.connect(self.OnMaxMouseWheel)

    def OnSlider(self, evt: SliderEvent) -> None:
        self.spin_min.SetValue(evt.minimun)
        self.spin_max.SetValue(evt.maximun)
        self.minimun = evt.minimun
        self.maximun = evt.maximun
        self._GenerateEvent(self.threshold_changed)

    def OnSliding(self, evt: SliderEvent) -> None:
        self.spin_min.SetValue(evt.minimun)
        self.spin_max.SetValue(evt.maximun)
        self.minimun = evt.minimun
        self.maximun = evt.maximun
        self._GenerateEvent(self.threshold_changing)

    def OnMinMouseWheel(self, *args) -> None:
        """
        When the user wheel the mouse over min texbox
        """
        v = self.spin_min.GetValue()
        self.SetMinValue(v)
        self._GenerateEvent(self.threshold_changed)

    def OnMaxMouseWheel(self, *args) -> None:
        """
        When the user wheel the mouse over max texbox
        """
        v = self.spin_max.GetValue()
        self.SetMaxValue(v)
        self._GenerateEvent(self.threshold_changed)

    def SetColour(self, colour: Sequence[SupportsInt]) -> None:
        colour = list(int(i) for i in colour[:3]) + [90]
        self.colour = colour
        self.gradient_slider.SetColour(colour)
        self.gradient_slider.update()

    def SetGradientColours(self, colors: Sequence[ColourType]) -> None:
        self.gradient_slider.SetGradientColours(colors)

    def SetMaxRange(self, value: int) -> None:
        self.spin_min.SetMax(value)
        self.spin_max.SetMax(value)
        self.spin_min.CalcSizeFromTextSize()
        self.spin_max.CalcSizeFromTextSize()
        self.gradient_slider.SetMaxRange(value)
        self.max_range = value
        if value > self.max_range:
            value = self.max_range

        self.spin_min.CalcSizeFromTextSize()
        self.spin_max.CalcSizeFromTextSize()
        self.updateGeometry()

    def SetMinRange(self, value: int) -> None:
        self.spin_min.SetMin(value)
        self.spin_max.SetMin(value)
        self.spin_min.CalcSizeFromTextSize()
        self.spin_max.CalcSizeFromTextSize()
        self.gradient_slider.SetMinRange(value)
        self.min_range = value
        if value < self.min_range:
            value = self.min_range

        self.spin_min.CalcSizeFromTextSize()
        self.spin_max.CalcSizeFromTextSize()
        self.updateGeometry()

    def SetMaxValue(self, value: Optional[SupportsInt]) -> None:
        if value is not None:
            value = int(value)
            if value > self.max_range:
                value = int(self.max_range)
            if value < self.min_range:
                value = int(self.min_range)
            if value < self.minimun:
                value = int(self.minimun)
            self.spin_max.SetValue(value)
            self.gradient_slider.SetMaximun(value)
            self.maximun = value

    def SetMinValue(self, value: Optional[SupportsInt]) -> None:
        if value is not None:
            value = int(value)
            if value < self.min_range:
                value = int(self.min_range)
            if value > self.max_range:
                value = int(self.max_range)
            if value > self.maximun:
                value = int(self.maximun)
            self.spin_min.SetValue(value)
            self.gradient_slider.SetMinimun(value)
            self.minimun = value

    def GetMaxValue(self) -> int:
        return self.maximun

    def GetMinValue(self) -> int:
        return self.minimun

    def _GenerateEvent(self, signal: Signal) -> None:
        if signal is self.threshold_changing:
            self.changed = True
        elif signal is self.threshold_changed:
            self.changed = False

        evt = SliderEvent(
            self.min_range,
            self.max_range,
            self.minimun,
            self.maximun,
        )
        signal.emit(evt)


class GradientDisp(QWidget):
    # Class for colorbars gradient used in fmri support (showing different colormaps possible)
    threshold_changed = Signal(object)
    threshold_changing = Signal(object)

    def __init__(
        self,
        parent: QWidget,
        id: int,
        minRange: int,
        maxRange: int,
        minValue: int,
        maxValue: int,
        colours: List[ColourType],
    ):
        super().__init__(parent)
        self.min_range = minRange
        self.max_range = maxRange
        self.minimun = minValue
        self.maximun = maxValue
        self.colours = colours
        self.changed = False
        self._draw_controls()
        self.show()

    def _draw_controls(self) -> None:
        self.gradient_slider = GradientNoSlide(
            self, -1, self.min_range, self.max_range, self.minimun, self.maximun, self.colours
        )

        layout = QHBoxLayout()
        layout.setContentsMargins(1, 0, 1, 0)
        layout.setSpacing(0)
        layout.addWidget(self.gradient_slider, 1)
        self.setLayout(layout)
        self.setMinimumHeight(30)

    def OnSlider(self, evt: SliderEvent) -> None:
        self.minimun = evt.minimun
        self.maximun = evt.maximun
        self._GenerateEvent(self.threshold_changed)

    def OnSliding(self, evt: SliderEvent) -> None:
        self.minimun = evt.minimun
        self.maximun = evt.maximun
        self._GenerateEvent(self.threshold_changing)

    def SetGradientColours(self, colors: Sequence[ColourType]) -> None:
        self.gradient_slider.SetGradientColours(colors)

    def GetMaxValue(self) -> int:
        return self.maximun

    def GetMinValue(self) -> int:
        return self.minimun

    def _GenerateEvent(self, signal: Signal) -> None:
        if signal is self.threshold_changing:
            self.changed = True
        elif signal is self.threshold_changed:
            self.changed = False

        evt = SliderEvent(
            self.min_range,
            self.max_range,
            self.minimun,
            self.maximun,
        )
        signal.emit(evt)
