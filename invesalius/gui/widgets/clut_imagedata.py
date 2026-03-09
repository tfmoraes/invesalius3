import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFontMetricsF,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QColorDialog, QWidget

if TYPE_CHECKING:
    import numpy as np

HISTOGRAM_LINE_COLOUR = (128, 128, 128)
HISTOGRAM_FILL_COLOUR = (64, 64, 64)
HISTOGRAM_LINE_WIDTH = 1

DEFAULT_COLOUR = (0, 0, 0)

TEXT_COLOUR = (255, 255, 255)
BACKGROUND_TEXT_COLOUR_RGBA = (255, 0, 0, 128)

GRADIENT_RGBA = 0.75 * 255

LINE_COLOUR = (128, 128, 128)
LINE_WIDTH = 2
RADIUS = 5

PADDING = 2


@dataclass(order=True)
class Node:
    value: float
    colour: Tuple[int, int, int] = field(compare=False)


class CLUTEvent:
    def __init__(self, nodes: List[Node]):
        self.nodes = nodes

    def GetNodes(self) -> List[Node]:
        return self.nodes


class CLUTImageDataWidget(QWidget):
    """
    Widget used to config the Lookup table from imagedata.
    """

    clut_node_changed = Signal(object)

    def __init__(
        self,
        parent: QWidget,
        id: int,
        histogram: "np.ndarray",
        init: float,
        end: float,
        nodes: Optional[List[Node]] = None,
    ):
        super().__init__(parent)

        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumSize(400, 200)

        self.histogram = histogram

        self._init = init
        self._end = end

        self.i_init = init
        self.i_end = end

        self._range = 0.05 * (end - init)
        self._scale = 1.0

        if nodes is None:
            self.wl = (init + end) / 2.0
            self.ww = end - init

            self.nodes = [Node(init, (0, 0, 0)), Node(end, (255, 255, 255))]
        else:
            self.nodes = nodes
            self.nodes.sort()

            n0 = nodes[0]
            nn = nodes[-1]

            self.ww = nn.value - n0.value
            self.wl = (nn.value + n0.value) / 2.0

        self._s_init = init
        self._s_end = end

        self.middle_pressed = False
        self.right_pressed = False
        self.left_pressed = False

        self.selected_node: Optional[Node] = None
        self.last_selected: Optional[Node] = None

        self.first_show = True

        self._d_hist: List[Tuple[float, float]] = []

        self._build_drawn_hist()

    @property
    def window_level(self) -> float:
        self.nodes.sort()
        p0 = self.nodes[0].value
        pn = self.nodes[-1].value
        return (pn + p0) / 2

    @property
    def window_width(self) -> float:
        self.nodes.sort()
        p0 = self.nodes[0].value
        pn = self.nodes[-1].value
        return pn - p0

    def _build_drawn_hist(self) -> None:
        w = self.width()
        h = self.height()

        x_init = self._init
        x_end = self._end

        y_init = 0
        y_end = math.log(self.histogram.max() + 1)

        prop_x = (w) * 1.0 / (x_end - x_init)
        prop_y = (h) * 1.0 / (y_end - y_init)

        self._d_hist = []
        for i in range(w):
            x = i / prop_x + x_init - 1
            if self.i_init <= x < self.i_end:
                try:
                    y = math.log(self.histogram[int(x - self.i_init)] + 1) * prop_y
                except IndexError:
                    pass

                self._d_hist.append((i, y))

    def _interpolation(self, x: float):
        f = math.floor(x)
        c = math.ceil(x)
        h = self.histogram

        if f != c:
            return h[f] + (h[c] - h[f]) / (c - f) * (x - f)
        else:
            return h[int(x)]

    def resizeEvent(self, event) -> None:
        if self.first_show:
            w = self.width()
            init = self.pixel_to_hounsfield(-RADIUS)
            end = self.pixel_to_hounsfield(w + RADIUS)
            self._init = init
            self._end = end
            self._range = 0.05 * (end - init)

            self._s_init = init
            self._s_end = end

            self.first_show = False

        self._build_drawn_hist()
        self.update()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor(0, 0, 0))

        self.draw_histogram(painter)
        self.draw_gradient(painter)

        if self.last_selected is not None:
            self.draw_text(painter, self.last_selected.value)

        painter.end()

    def wheelEvent(self, event) -> None:
        """
        Increase or decrease the range from hounsfield scale showed. It
        doesn't change values in preset, only to visualization.
        """
        direction = event.angleDelta().y() / 120.0
        init = self._init - direction * self._range
        end = self._end + direction * self._range
        self.SetRange(init, end)
        self.update()

    def mousePressEvent(self, event) -> None:
        px = int(event.position().x())
        py = int(event.position().y())

        if event.button() == Qt.MiddleButton:
            self.middle_pressed = True
            self.last_x = self.pixel_to_hounsfield(px)

        elif event.button() == Qt.LeftButton:
            self.left_pressed = True
            self.selected_node = self.get_node_clicked(px, py)
            self.last_selected = self.selected_node
            if self.selected_node is not None:
                self.update()

        elif event.button() == Qt.RightButton:
            w = self.width()
            h = self.height()
            selected_node = self.get_node_clicked(px, py)
            if selected_node:
                self.nodes.remove(selected_node)
                self._generate_event()
                self.update()

        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self.middle_pressed = False
        elif event.button() == Qt.LeftButton:
            self.left_pressed = False
            self.selected_node = None
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return

        px = int(event.position().x())
        py = int(event.position().y())

        selected_node = self.get_node_clicked(px, py)
        if selected_node:
            color = QColorDialog.getColor(QColor(0, 0, 0), self)
            if color.isValid():
                selected_node.colour = (color.red(), color.green(), color.blue())
                self._generate_event()
        else:
            vx = self.pixel_to_hounsfield(px)
            node = Node(vx, DEFAULT_COLOUR)
            self.nodes.append(node)
            self._generate_event()

        self.update()

    def mouseMoveEvent(self, event) -> None:
        if self.middle_pressed:
            x = self.pixel_to_hounsfield(int(event.position().x()))
            dx = x - self.last_x
            init = self._init - dx
            end = self._end - dx
            self.SetRange(init, end)
            self.update()
            self.last_x = x

        elif self.left_pressed and self.selected_node:
            x = self.pixel_to_hounsfield(int(event.position().x()))
            self.selected_node.value = float(x)
            self.update()
            self._generate_event()

    def keyPressEvent(self, event) -> None:
        if self.last_selected is not None:
            key = event.key()

            if key in (Qt.Key_Right,):
                n = self.last_selected
                n.value = self.pixel_to_hounsfield(self.hounsfield_to_pixel(n.value) + 1)
                self.update()
                self._generate_event()

            elif key in (Qt.Key_Left,):
                n = self.last_selected
                n.value = self.pixel_to_hounsfield(self.hounsfield_to_pixel(n.value) - 1)
                self.update()
                self._generate_event()

            elif key in (Qt.Key_Return, Qt.Key_Enter):
                n = self.last_selected
                color = QColorDialog.getColor(QColor(*n.colour), self)
                if color.isValid():
                    n.colour = (color.red(), color.green(), color.blue())
                    self.update()
                    self._generate_event()

            elif key in (Qt.Key_Delete,):
                n = self.last_selected
                self.last_selected = None
                self.nodes.remove(n)
                self.update()
                self._generate_event()

            elif key == Qt.Key_Tab:
                n = self.last_selected
                self.nodes.sort()
                idx = self.nodes.index(n)
                if event.modifiers() & Qt.ShiftModifier:
                    nidx = (idx - 1) % len(self.nodes)
                else:
                    nidx = (idx + 1) % len(self.nodes)
                self.last_selected = self.nodes[nidx]
                self.update()

        super().keyPressEvent(event)

    def draw_histogram(self, painter: QPainter) -> None:
        w = self.width()
        h = self.height()

        if not self._d_hist:
            return

        painter.save()

        hist_pen = QPen(QColor(*HISTOGRAM_LINE_COLOUR), HISTOGRAM_LINE_WIDTH)
        hist_brush = QBrush(QColor(*HISTOGRAM_FILL_COLOUR))

        path = QPainterPath()
        xi, yi = self._d_hist[0]
        path.moveTo(xi, h - yi)
        for x, y in self._d_hist:
            path.lineTo(x, h - y)

        painter.translate(self.hounsfield_to_pixel(self._s_init), 0)
        painter.scale(self._scale, 1.0)

        painter.setPen(hist_pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawPath(path)

        path.lineTo(x, h)
        path.lineTo(xi, h)
        path.lineTo(*self._d_hist[0])

        painter.setPen(Qt.NoPen)
        painter.setBrush(hist_brush)
        painter.drawPath(path)

        painter.restore()

    def draw_gradient(self, painter: QPainter) -> None:
        w = self.width()
        h = self.height()

        knodes = sorted(self.nodes)
        for ni, nj in zip(knodes[:-1], knodes[1:]):
            vi = round(self.hounsfield_to_pixel(ni.value))
            vj = round(self.hounsfield_to_pixel(nj.value))

            ci = QColor(*ni.colour, int(GRADIENT_RGBA))
            cj = QColor(*nj.colour, int(GRADIENT_RGBA))

            gradient = QLinearGradient(vi, h, vj, h)
            gradient.setColorAt(0.0, ci)
            gradient.setColorAt(1.0, cj)

            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(gradient))
            painter.drawRect(QRectF(vi, 0, vj - vi, h))

            self._draw_circle(vi, ni.colour, painter)
            self._draw_circle(vj, nj.colour, painter)

    def _draw_circle(self, px: float, color: Tuple[int, int, int], painter: QPainter) -> None:
        w = self.width()
        h = self.height()

        center = QPointF(px, h / 2)

        painter.setPen(QPen(QColor(255, 255, 255), LINE_WIDTH + 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawEllipse(center, RADIUS, RADIUS)

        painter.setPen(QPen(QColor(*LINE_COLOUR), LINE_WIDTH - 1))
        painter.setBrush(QBrush(QColor(*color)))
        painter.drawEllipse(center, RADIUS, RADIUS)

    def draw_text(self, painter: QPainter, value: float) -> None:
        w = self.width()
        h = self.height()

        x = self.hounsfield_to_pixel(value)
        y = h / 2

        font = painter.font()
        font.setBold(True)
        painter.setFont(font)

        text = "Value: %-6d" % value

        fm = QFontMetricsF(font)
        wt = fm.horizontalAdvance(text)
        ht = fm.height()

        wr, hr = wt + 2 * PADDING, ht + 2 * PADDING
        xr, yr = x + RADIUS, y - RADIUS - hr

        if xr + wr > w:
            xr = x - RADIUS - wr
        if yr < 0:
            yr = y + RADIUS

        xf, yf = xr + PADDING, yr + PADDING

        bg_color = QColor(*BACKGROUND_TEXT_COLOUR_RGBA)
        painter.setBrush(QBrush(bg_color))
        painter.setPen(QPen(bg_color))
        painter.drawRect(QRectF(xr, yr, wr, hr))

        painter.setPen(QPen(QColor(*TEXT_COLOUR)))
        painter.drawText(QPointF(xf, yf + fm.ascent()), text)

    def _generate_event(self) -> None:
        evt = CLUTEvent(self.nodes)
        self.clut_node_changed.emit(evt)

    def hounsfield_to_pixel(self, x: float) -> float:
        w = self.width()
        p = (x - self._init) * w * 1.0 / (self._end - self._init)
        return p

    def pixel_to_hounsfield(self, x: float) -> float:
        w = self.width()
        prop_x = (self._end - self._init) / (w * 1.0)
        p = x * prop_x + self._init
        return p

    def get_node_clicked(self, px: int, py: int) -> Optional[Node]:
        h = self.height()
        for n in self.nodes:
            x = self.hounsfield_to_pixel(n.value)
            y = h / 2

            if ((px - x) ** 2 + (py - y) ** 2) ** 0.5 <= RADIUS:
                return n

        return None

    def SetRange(self, init: float, end: float) -> None:
        """
        Sets the range from hounsfield
        """
        scale = (self._s_end - self._s_init) * 1.0 / (end - init)
        if scale <= 10.0:
            self._scale = scale
            self._init, self._end = init, end
