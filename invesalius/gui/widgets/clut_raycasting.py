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

import bisect
import math
import os
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple, Union

import numpy
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFontMetrics,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QColorDialog, QWidget

import invesalius.gui.dialogs as dialog
from invesalius import inv_paths
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher

if TYPE_CHECKING:
    import numpy as np

    from typings.utils import SupportsGetItem

FONT_COLOUR = (1, 1, 1)
LINE_COLOUR = (128, 128, 128)
LINE_WIDTH = 2
HISTOGRAM_LINE_WIDTH = 1
HISTOGRAM_LINE_COLOUR = (128, 128, 128)
HISTOGRAM_FILL_COLOUR = (64, 64, 64)
BACKGROUND_TEXT_COLOUR_RGBA = (255, 0, 0, 128)
TEXT_COLOUR = (255, 255, 255)
GRADIENT_RGBA = 0.75 * 255
RADIUS = 5
SELECTION_SIZE = 10
TOOLBAR_SIZE = 30
TOOLBAR_COLOUR = (25, 25, 25)
RANGE = 10
PADDING = 2


class Node:
    """
    Represents the points in the raycasting preset. Contains its colour,
    graylevel (hounsfield scale), opacity, x and y position in the widget.
    """

    def __init__(
        self, colour: Tuple[int, int, int], x: int, y: int, graylevel: float, opacity: float
    ):
        self.colour = colour
        self.x = x
        self.y = y
        self.graylevel = graylevel
        self.opacity = opacity


class Curve:
    """
    Represents the curves in the raycasting preset. It contains the point nodes from
    the curve and its window width & level.
    """

    def __init__(self) -> None:
        self.wl: float = 0
        self.ww: float = 0
        self.wl_px: Optional[Tuple[float, int]] = None
        self.nodes: List[Node] = []

    def CalculateWWWl(self) -> None:
        """
        Called when the curve width(ww) or position(wl) is modified.
        """
        self.ww = self.nodes[-1].graylevel - self.nodes[0].graylevel
        self.wl = self.nodes[0].graylevel + self.ww / 2.0


class Histogram:
    def __init__(self) -> None:
        self.init: float = -1024
        self.end: float = 2000
        self.points: List[Tuple[float, float]] = []


class Button:
    """
    The button in the clut raycasting.
    """

    def __init__(self, image: QPixmap) -> None:
        self.image: QPixmap = image
        self.position: Tuple[float, float] = (0, 0)
        self.size: Tuple[int, int] = (24, 24)

    def HasClicked(self, position: Tuple[int, int]) -> bool:
        """
        Test if the button was clicked.
        """
        m_x, m_y = position
        i_x, i_y = self.position
        w, h = self.size
        if i_x < m_x < i_x + w and i_y < m_y < i_y + h:
            return True
        else:
            return False


class CLUTRaycastingWidget(QWidget):
    """
    This class represents the frame where images is showed
    """

    clut_slider = Signal(int)
    clut_slider_change = Signal(int)
    clut_point_move = Signal(int)
    clut_point_release = Signal(int)
    clut_curve_select = Signal(int)
    clut_curve_wl_change = Signal(int)

    def __init__(self, parent: QWidget, id: int = -1):
        """
        Constructor.

        parent -- parent of this frame
        """
        super().__init__(parent)
        self.setAttribute(Qt.WA_OpaquePaintEvent)
        self.points: List[List[Dict[str, float]]] = []
        self.colours: List[List[Dict[str, float]]] = []
        self.curves: List[Curve] = []
        self.init: float = -1024
        self.end: float = 2000
        self.Histogram = Histogram()
        self.padding = 5
        self.previous_wl = 0
        self.to_render = False
        self.dragged = False
        self.middle_drag = False
        self.to_draw_points = False
        self.point_dragged: Optional[Tuple[int, int]] = None
        self.curve_dragged: Optional[int] = None
        self.last_position: int = 0
        self.histogram_array: Union["np.ndarray", List[int]] = [100, 100]
        self.CalculatePixelPoints()
        self._build_buttons()
        self.show()

    def SetRange(self, range: Tuple[float, float]) -> None:
        """
        Se the range from hounsfield
        """
        self.init, self.end = range
        self.CalculatePixelPoints()

    def SetPadding(self, padding: int) -> None:
        self.padding = padding

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self.to_draw_points:
            self.Render(painter)
        painter.end()

    def resizeEvent(self, event) -> None:
        self.CalculatePixelPoints()
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._on_left_click(event)
        elif event.button() == Qt.RightButton:
            self._on_right_click(event)
        elif event.button() == Qt.MiddleButton:
            self._on_middle_click(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._on_double_click(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._on_left_release(event)
        elif event.button() == Qt.MiddleButton:
            self._on_middle_release(event)

    def mouseMoveEvent(self, event) -> None:
        pos = event.position().toPoint()
        x = pos.x()
        y = pos.y()
        if self.dragged and self.point_dragged:
            self._move_node(x, y, self.point_dragged)
        elif self.dragged and self.curve_dragged is not None:
            self._move_curve(x, y)
        elif self.middle_drag:
            d = self.PixelToHounsfield(x) - self.PixelToHounsfield(self.last_position)
            self.SetRange((self.init - d, self.end - d))
            self.last_position = x
            self.update()

    def wheelEvent(self, event) -> None:
        delta = event.angleDelta().y()
        direction = delta / abs(delta) if delta != 0 else 0
        init = self.init - RANGE * direction
        end = self.end + RANGE * direction
        self.SetRange((init, end))
        self.update()

    def _on_left_click(self, event) -> None:
        pos = event.position().toPoint()
        x, y = pos.x(), pos.y()
        if self.save_button.HasClicked((x, y)):
            filename = dialog.ShowSavePresetDialog()
            if filename:
                Publisher.sendMessage("Save raycasting preset", preset_name=filename)
        point = self._has_clicked_in_a_point((x, y))
        if point:
            self.dragged = True
            self.point_dragged = point
            self.update()
            return
        curve = self._has_clicked_in_selection_curve((x, y))
        if curve is not None:
            self.dragged = True
            self.previous_wl = x
            self.curve_dragged = curve
            self.clut_curve_select.emit(curve)
            return
        else:
            point_2 = self._has_clicked_in_line((x, y))
            if point_2:
                n, p = point_2
                self.points[n].insert(p, {"x": 0, "y": 0})
                self.colours[n].insert(p, {"red": 0, "green": 0, "blue": 0})
                self.points[n][p]["x"] = self.PixelToHounsfield(x)
                self.points[n][p]["y"] = self.PixelToOpacity(y)

                node = Node(
                    colour=(0, 0, 0),
                    x=x,
                    y=y,
                    graylevel=self.points[n][p]["x"],
                    opacity=self.points[n][p]["y"],
                )
                self.curves[n].nodes.insert(p, node)

                self.update()
                self.clut_point_release.emit(n)
                return

    def _on_double_click(self, event) -> None:
        """
        Used to change the colour of a point
        """
        pos = event.position().toPoint()
        point = self._has_clicked_in_a_point((pos.x(), pos.y()))
        if point:
            i, j = point
            actual_colour = self.curves[i].nodes[j].colour
            initial = QColor(*actual_colour)
            colour = QColorDialog.getColor(initial, self)
            if colour.isValid():
                r, g, b = colour.red(), colour.green(), colour.blue()

                self.colours[i][j]["red"] = r / 255.0
                self.colours[i][j]["green"] = g / 255.0
                self.colours[i][j]["blue"] = b / 255.0
                self.curves[i].nodes[j].colour = (r, g, b)
                self.update()
                self.clut_point_release.emit(i)

    def _on_right_click(self, event) -> None:
        """
        Used to remove a point
        """
        pos = event.position().toPoint()
        point = self._has_clicked_in_a_point((pos.x(), pos.y()))
        if point:
            i, j = point
            self.RemovePoint(i, j)
            self.update()
            self.clut_point_release.emit(i)
            return
        n_curve = self._has_clicked_in_selection_curve((pos.x(), pos.y()))
        if n_curve is not None:
            self.RemoveCurve(n_curve)
            self.update()
            self.clut_point_release.emit(n_curve)

    def _on_left_release(self, event) -> None:
        """
        Generate a clut_point_release signal indicating that a change has
        been occurred in the preset points.
        """
        if self.to_render:
            self.clut_point_release.emit(0)
        self.dragged = False
        self.curve_dragged = None
        self.point_dragged = None
        self.to_render = False
        self.previous_wl = 0

    def _on_middle_click(self, event) -> None:
        self.middle_drag = True
        self.last_position = event.position().toPoint().x()

    def _on_middle_release(self, event) -> None:
        self.middle_drag = False

    def _has_clicked_in_a_point(
        self, position: "SupportsGetItem[float]"
    ) -> Optional[Tuple[int, int]]:
        """
        returns the index from the selected point
        """
        for i, curve in enumerate(self.curves):
            for j, node in enumerate(curve.nodes):
                if self._calculate_distance((node.x, node.y), position) <= RADIUS:
                    return (i, j)
        return None

    def distance_from_point_line(
        self, p1: Tuple[float, float], p2: Tuple[float, float], pc: Tuple[float, float]
    ) -> float:
        """
        Calculate the distance from point pc to a line formed by p1 and p2.
        """
        A = numpy.array(pc) - numpy.array(p1)
        B = numpy.array(p2) - numpy.array(p1)
        len_A = numpy.linalg.norm(A)
        len_B = numpy.linalg.norm(B)
        theta = math.acos(numpy.dot(A, B) / (len_A * len_B))
        distance = float(math.sin(theta) * len_A)
        return distance

    def _has_clicked_in_selection_curve(self, position: "SupportsGetItem[float]") -> Optional[int]:
        for i, curve in enumerate(self.curves):
            if self._calculate_distance(curve.wl_px, position) <= RADIUS:
                return i
        return None

    def _has_clicked_in_line(self, clicked_point: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Verify if was clicked in a line. If yes, it returns the insertion
        clicked_point in the point list.
        """
        for n, curve in enumerate(self.curves):
            position = bisect.bisect([node.x for node in curve.nodes], clicked_point[0])
            if position != 0 and position != len(curve.nodes):
                p1 = curve.nodes[position - 1].x, curve.nodes[position - 1].y
                p2 = curve.nodes[position].x, curve.nodes[position].y
                if self.distance_from_point_line(p1, p2, clicked_point) <= 5:
                    return (n, position)
        return None

    def _has_clicked_in_save(self, clicked_point: Tuple[int, int]) -> bool:
        x, y = clicked_point
        if self.padding < x < self.padding + 24 and self.padding < y < self.padding + 24:
            return True
        else:
            return False

    def _calculate_distance(
        self, p1: "SupportsGetItem[float]", p2: "SupportsGetItem[float]"
    ) -> float:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

    def _move_node(self, x: int, y: int, point: Tuple[int, int]) -> None:
        self.to_render = True
        i, j = point

        width = self.width()
        height = self.height()

        if y >= height - self.padding:
            y = height - self.padding

        if y <= self.padding:
            y = self.padding

        x = max(x, 0)

        if x > width:
            x = width

        x = max(x, TOOLBAR_SIZE)

        if j > 0 and x <= self.curves[i].nodes[j - 1].x:
            x = self.curves[i].nodes[j - 1].x + 1

        if j < len(self.curves[i].nodes) - 1 and x >= self.curves[i].nodes[j + 1].x:
            x = self.curves[i].nodes[j + 1].x - 1

        graylevel = self.PixelToHounsfield(x)
        opacity = self.PixelToOpacity(y)
        self.points[i][j]["x"] = graylevel
        self.points[i][j]["y"] = opacity
        self.curves[i].nodes[j].x = x
        self.curves[i].nodes[j].y = y
        self.curves[i].nodes[j].graylevel = graylevel
        self.curves[i].nodes[j].opacity = opacity
        for curve in self.curves:
            curve.CalculateWWWl()
            curve.wl_px = (self.HounsfieldToPixel(curve.wl), self.OpacityToPixel(0))
        self.update()

        self.clut_point_move.emit(i)

    def _move_curve(self, x: int, y: int) -> None:
        curve = self.curves[self.curve_dragged]
        curve.wl = self.PixelToHounsfield(x)
        curve.wl_px = x, self.OpacityToPixel(0)
        for node in curve.nodes:
            node.x += x - self.previous_wl
            node.graylevel = self.PixelToHounsfield(node.x)

        self.previous_wl = x
        self.to_draw_points = True
        self.update()

        self.clut_curve_wl_change.emit(self.curve_dragged)

    def RemovePoint(self, i: int, j: int) -> None:
        """
        The point the point in the given i,j index
        """
        self.points[i].pop(j)
        self.colours[i].pop(j)

        self.curves[i].nodes.pop(j)
        if (i, j) == self.point_dragged:
            self.point_dragged = None
        elif self.point_dragged and i == self.point_dragged[0] and j < self.point_dragged[1]:
            new_i = self.point_dragged[0]
            new_j = self.point_dragged[1] - 1
            self.point_dragged = (new_i, new_j)
        if len(self.points[i]) == 1:
            self.RemoveCurve(i)
        else:
            curve = self.curves[i]
            curve.CalculateWWWl()
            curve.wl_px = (self.HounsfieldToPixel(curve.wl), self.OpacityToPixel(0))

    def RemoveCurve(self, n_curve: int) -> None:
        self.points.pop(n_curve)
        self.colours.pop(n_curve)
        self.point_dragged = None

        self.curves.pop(n_curve)

    def _draw_gradient(self, painter: QPainter, height: int) -> None:
        height += self.padding
        for curve in self.curves:
            for nodei, nodej in zip(curve.nodes[:-1], curve.nodes[1:]):
                path = QPainterPath()
                path.moveTo(int(nodei.x), height)
                path.lineTo(int(nodei.x), nodei.y)
                path.lineTo(int(nodej.x), nodej.y)
                path.lineTo(int(nodej.x), height)
                path.closeSubpath()

                gradient = QLinearGradient(int(nodei.x), 0, int(nodej.x), 0)
                colouri = QColor(
                    int(nodei.colour[0]),
                    int(nodei.colour[1]),
                    int(nodei.colour[2]),
                    int(GRADIENT_RGBA),
                )
                colourj = QColor(
                    int(nodej.colour[0]),
                    int(nodej.colour[1]),
                    int(nodej.colour[2]),
                    int(GRADIENT_RGBA),
                )
                gradient.setColorAt(0, colouri)
                gradient.setColorAt(1, colourj)
                painter.setBrush(QBrush(gradient))
                painter.setPen(Qt.NoPen)
                painter.drawPath(path)

    def _draw_curves(self, painter: QPainter) -> None:
        painter.setPen(QPen(QColor(*LINE_COLOUR), LINE_WIDTH))
        painter.setBrush(Qt.NoBrush)
        for curve in self.curves:
            path = QPainterPath()
            path.moveTo(curve.nodes[0].x, curve.nodes[0].y)
            for node in curve.nodes:
                path.lineTo(node.x, node.y)
            painter.drawPath(path)

    def _draw_points(self, painter: QPainter) -> None:
        for curve in self.curves:
            for node in curve.nodes:
                painter.setPen(QPen(QColor(*LINE_COLOUR), LINE_WIDTH))
                painter.setBrush(QBrush(QColor(*node.colour)))
                painter.drawEllipse(
                    int(node.x - RADIUS), int(node.y - RADIUS), RADIUS * 2, RADIUS * 2
                )

    def _draw_selected_point_text(self, painter: QPainter) -> None:
        i, j = self.point_dragged
        node = self.curves[i].nodes[j]
        x, y = node.x, node.y
        value = node.graylevel
        alpha = node.opacity
        widget_width = self.width()

        font = painter.font()
        font.setBold(True)
        painter.setFont(font)
        fm = QFontMetrics(font)

        text1 = _("Value: %-6d" % value)
        text2 = _("Alpha: %-.3f" % alpha)  # noqa: UP031

        wt1 = fm.horizontalAdvance(text1)
        wt2 = fm.horizontalAdvance(text2)
        wt = max(wt1, wt2)
        ht = fm.height()

        wr, hr = wt + 2 * PADDING, ht * 2 + 2 * PADDING
        xr, yr = x + RADIUS, y - RADIUS - hr

        if xr + wr > widget_width:
            xr = x - RADIUS - wr
        if yr < 0:
            yr = y + RADIUS

        xf, yf = xr + PADDING, yr + PADDING

        painter.setBrush(QBrush(QColor(*BACKGROUND_TEXT_COLOUR_RGBA)))
        painter.setPen(QPen(QColor(*BACKGROUND_TEXT_COLOUR_RGBA)))
        painter.drawRect(int(xr), int(yr), int(wr), int(hr))
        painter.setPen(QPen(QColor(*TEXT_COLOUR)))
        painter.drawText(int(xf), int(yf + fm.ascent()), text1)
        painter.drawText(int(xf), int(yf + ht + fm.ascent()), text2)

    def _draw_histogram(self, painter: QPainter, height: int) -> None:
        if not self.Histogram.points:
            return
        x, y = self.Histogram.points[0]

        painter.setPen(QPen(QColor(*HISTOGRAM_LINE_COLOUR), HISTOGRAM_LINE_WIDTH))
        painter.setBrush(Qt.NoBrush)

        stroke_path = QPainterPath()
        stroke_path.moveTo(x, y)
        for x, y in self.Histogram.points:
            stroke_path.lineTo(x, y)

        painter.drawPath(stroke_path)

        fill_path = QPainterPath(stroke_path)
        fill_path.lineTo(x, height + self.padding)
        fill_path.lineTo(self.HounsfieldToPixel(self.Histogram.init), height + self.padding)
        x, y = self.Histogram.points[0]
        fill_path.lineTo(x, y)
        fill_path.closeSubpath()

        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(*HISTOGRAM_FILL_COLOUR)))
        painter.drawPath(fill_path)

    def _draw_selection_curve(self, painter: QPainter, height: int) -> None:
        painter.setPen(QPen(QColor(*LINE_COLOUR), LINE_WIDTH))
        painter.setBrush(QBrush(QColor(0, 0, 0)))
        for curve in self.curves:
            x_center, y_center = curve.wl_px
            painter.drawRect(
                int(x_center - SELECTION_SIZE / 2.0),
                int(y_center),
                SELECTION_SIZE,
                SELECTION_SIZE,
            )

    def _draw_tool_bar(self, painter: QPainter, height: int) -> None:
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor(*TOOLBAR_COLOUR)))
        painter.drawRect(0, 0, TOOLBAR_SIZE, height + self.padding * 2)
        image = self.save_button.image
        w, h = self.save_button.size
        x = (TOOLBAR_SIZE - w) / 2.0
        y = self.padding
        self.save_button.position = (x, y)
        painter.drawPixmap(int(x), int(y), w, h, image)

    def Render(self, painter: QPainter) -> None:
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        height -= self.padding * 2
        width -= self.padding

        self._draw_histogram(painter, height)
        self._draw_gradient(painter, height)
        self._draw_curves(painter)
        self._draw_points(painter)
        self._draw_selection_curve(painter, height)
        self._draw_tool_bar(painter, height)
        if self.point_dragged:
            self._draw_selected_point_text(painter)

    def _build_histogram(self) -> None:
        width = self.width()
        height = self.height()
        width -= self.padding
        height -= self.padding * 2
        x_init = self.Histogram.init
        y_init = 0
        y_end = math.log(max(self.histogram_array))
        proportion_y = height * 1.0 / (y_end - y_init)
        self.Histogram.points = []
        for i in range(0, len(self.histogram_array), 5):
            if self.histogram_array[i]:
                y = math.log(self.histogram_array[i])
            else:
                y = 0
            x = self.HounsfieldToPixel(x_init + i)
            y = height - y * proportion_y + self.padding
            self.Histogram.points.append((x, y))

    def _build_buttons(self) -> None:
        pixmap = QPixmap(os.path.join(inv_paths.ICON_DIR, "Floppy.png"))
        width = pixmap.width()
        height = pixmap.height()
        self.save_button = Button(pixmap)
        self.save_button.size = (width, height)

    def __sort_pixel_points(self) -> None:
        """
        Sort the pixel points (colours and points) maintaining the reference
        between colours and points. It's necessary mainly in negative window
        width when the user interacts with this widgets.
        """
        for n, (point, colour) in enumerate(zip(self.points, self.colours)):
            point_colour: Iterable[Tuple[Dict[str, float], Dict[str, float]]] = zip(point, colour)
            point_colour = sorted(point_colour, key=lambda x: x[0]["x"])
            self.points[n] = [i[0] for i in point_colour]
            self.colours[n] = [i[1] for i in point_colour]

    def CalculatePixelPoints(self) -> None:
        """
        Create a list with points (in pixel x, y coordinate) to draw based in
        the preset points (Hounsfield scale, opacity).
        """
        self.curves = []
        self.__sort_pixel_points()
        for points, colours in zip(self.points, self.colours):
            curve = Curve()
            for point, colour in zip(points, colours):
                x = self.HounsfieldToPixel(point["x"])
                y = self.OpacityToPixel(point["y"])
                node = Node(
                    colour=(
                        int(colour["red"] * 255),
                        int(colour["green"] * 255),
                        int(colour["blue"] * 255),
                    ),
                    x=x,
                    y=y,
                    graylevel=point["x"],
                    opacity=point["y"],
                )
                curve.nodes.append(node)
            curve.CalculateWWWl()
            curve.wl_px = (self.HounsfieldToPixel(curve.wl), self.OpacityToPixel(0))
            self.curves.append(curve)
        self._build_histogram()

    def HounsfieldToPixel(self, graylevel: float) -> int:
        """
        Given a Hounsfield point returns a pixel point in the canvas.
        """
        width = self.width()
        width -= TOOLBAR_SIZE
        proportion = width * 1.0 / (self.end - self.init)
        x = (graylevel - self.init) * proportion + TOOLBAR_SIZE
        return x

    def OpacityToPixel(self, opacity: float) -> int:
        """
        Given a Opacity point returns a pixel point in the canvas.
        """
        height = self.height()
        height -= self.padding * 2
        y = height - (opacity * height) + self.padding
        return y

    def PixelToHounsfield(self, x: int) -> float:
        """
        Translate from pixel point to Hounsfield scale.
        """
        width = self.width()
        width -= TOOLBAR_SIZE
        proportion = width * 1.0 / (self.end - self.init)
        graylevel = (x - TOOLBAR_SIZE) / proportion - abs(self.init)
        return graylevel

    def PixelToOpacity(self, y: int) -> float:
        """
        Translate from pixel point to opacity.
        """
        height = self.height()
        height -= self.padding * 2
        opacity = (height - y + self.padding) * 1.0 / height
        return opacity

    def SetRaycastPreset(self, preset: Dict[str, List[List[Dict[str, float]]]]) -> None:
        if not preset:
            self.to_draw_points = False
        elif preset["advancedCLUT"]:
            self.to_draw_points = True
            self.points = preset["16bitClutCurves"]
            self.colours = preset["16bitClutColors"]
            self.CalculatePixelPoints()
        else:
            self.to_draw_points = False
        self.update()

    def SetHistogramArray(self, h_array: "np.ndarray", range: Tuple[float, float]) -> None:
        self.histogram_array = h_array
        self.Histogram.init = range[0]
        self.Histogram.end = range[1]

    def GetCurveWWWl(self, curve: int) -> Tuple[float, float]:
        return (self.curves[curve].ww, self.curves[curve].wl)


EVT_CLUT_SLIDER = "clut_slider"
EVT_CLUT_SLIDER_CHANGE = "clut_slider_change"
EVT_CLUT_POINT_MOVE = "clut_point_move"
EVT_CLUT_POINT_RELEASE = "clut_point_release"
EVT_CLUT_CURVE_SELECT = "clut_curve_select"
EVT_CLUT_CURVE_WL_CHANGE = "clut_curve_wl_change"
