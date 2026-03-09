import time

import numpy
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QScrollBar,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkImagingColor import vtkImageMapToWindowLevelColors
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkRenderingCore import vtkImageActor, vtkRenderer

import invesalius.constants as const
import invesalius.data.converters as converters
import invesalius.data.vtk_utils as vtku
import invesalius.reader.bitmap_reader as bitmap_reader
import invesalius.utils as utils
from invesalius.gui.widgets.canvas_renderer import CanvasRendererCTX
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher

NROWS = 3
NCOLS = 6
NUM_PREVIEWS = NCOLS * NROWS
PREVIEW_WIDTH = 70
PREVIEW_HEIGTH = 70

PREVIEW_BACKGROUND = (255, 255, 255)  # White

STR_SIZE = _("Image size: %d x %d")
STR_SPC = _("Spacing: %.2f")
STR_LOCAL = _("Location: %.2f")
STR_PATIENT = "%s\n%s"
STR_ACQ = _("%s %s\nMade in InVesalius")

FONTSIZE_SMALL = 2


class BitmapInfo:
    """
    Keep the informations and the image used by preview.
    """

    def __init__(self, data):
        self.id = data[7]
        self.title = data[6]
        self.data = data
        self.pos = data[8]
        self._preview = None
        self.selected = False
        self.thumbnail_path = data[1]

    @property
    def preview(self):
        if not self._preview:
            self._preview = QImage(self.thumbnail_path)
        return self._preview

    def release_thumbnail(self):
        self._preview = None


class DicomPaintPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.image = None
        self.last_size = (10, 10)
        self.pixmap = None

    def _image_resize(self, image):
        new_size = self.size()
        w, h = new_size.width(), new_size.height()
        if w > 0 and h > 0:
            self.last_size = (w, h)
            return image.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        else:
            return image.scaled(
                self.last_size[0],
                self.last_size[1],
                Qt.IgnoreAspectRatio,
                Qt.SmoothTransformation,
            )

    def SetImage(self, image):
        self.image = image
        r_img = self._image_resize(image)
        self.pixmap = QPixmap.fromImage(r_img)
        self.update()

    def paintEvent(self, event):
        if self.image and self.pixmap:
            painter = QPainter(self)
            painter.drawPixmap(0, 0, self.pixmap)
            painter.end()

    def resizeEvent(self, event):
        if self.image:
            r_img = self._image_resize(self.image)
            self.pixmap = QPixmap.fromImage(r_img)
        self.update()
        super().resizeEvent(event)


class Preview(QWidget):
    """
    The little previews.
    """

    preview_clicked = Signal(object, object, bool)
    preview_dblclicked = Signal(object, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.select_on = False
        self.bitmap_info = None
        self._init_ui()

    def _init_ui(self):
        self._set_background(PREVIEW_BACKGROUND)

        self.title = QLabel(_("Image"), self)
        self.title.setAlignment(Qt.AlignHCenter)
        self.subtitle = QLabel(_("Image"), self)
        self.subtitle.setAlignment(Qt.AlignHCenter)
        self.image_viewer = DicomPaintPanel(self)

        self.sizer = QVBoxLayout(self)
        self.sizer.setContentsMargins(0, 0, 0, 0)
        self.sizer.addWidget(self.title, 0, Qt.AlignHCenter)
        self.sizer.addWidget(self.subtitle, 0, Qt.AlignHCenter)
        self.sizer.addWidget(self.image_viewer, 1)

    def _set_background(self, color):
        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(*color))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

    def SetBitmapToPreview(self, bitmap_info):
        if self.bitmap_info:
            self.bitmap_info.release_thumbnail()

        self.bitmap_info = bitmap_info
        self.SetTitle(self.bitmap_info.title[-10:])
        self.SetSubtitle("")

        image = self.bitmap_info.preview

        self.image_viewer.SetImage(image)
        self.select_on = bitmap_info.selected
        self.Select()

    def SetTitle(self, title):
        self.title.setText(title)

    def SetSubtitle(self, subtitle):
        self.subtitle.setText(subtitle)

    def enterEvent(self, event):
        if not self.select_on:
            c = self.palette().color(QPalette.Button)
            self._set_background((c.red(), c.green(), c.blue()))

    def leaveEvent(self, event):
        if not self.select_on:
            self._set_background(PREVIEW_BACKGROUND)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            shift_pressed = bool(event.modifiers() & Qt.ShiftModifier)
            self.select_on = True
            self.bitmap_info.selected = True
            self.Select()

            self.preview_clicked.emit(self.bitmap_info.id, self.bitmap_info.data, shift_pressed)
            Publisher.sendMessage("Set bitmap in preview panel", pos=self.bitmap_info.pos)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.preview_dblclicked.emit(self.bitmap_info.id, self.bitmap_info.data)
        super().mouseDoubleClickEvent(event)

    def Select(self, on=True):
        if self.select_on:
            c = self.palette().color(QPalette.Highlight)
            self._set_background((c.red(), c.green(), c.blue()))
        else:
            self._set_background(PREVIEW_BACKGROUND)
        self.update()


class BitmapPreviewSeries(QWidget):
    """A dicom series preview panel"""

    serie_clicked = Signal(object, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.displayed_position = 0
        self.nhidden_last_display = 0
        self.selected_dicom = None
        self.selected_panel = None
        self._init_ui()

    def _init_ui(self):
        self.scroll = QScrollBar(Qt.Vertical, self)

        self.grid = QGridLayout()
        self.grid.setSpacing(3)

        grid_widget = QWidget(self)
        grid_layout = QHBoxLayout(grid_widget)
        grid_layout.setContentsMargins(2, 2, 2, 2)
        grid_layout.addLayout(self.grid, 1)

        background_sizer = QHBoxLayout(self)
        background_sizer.setContentsMargins(2, 2, 2, 2)
        background_sizer.addWidget(grid_widget, 1)
        background_sizer.addWidget(self.scroll, 0)

        self._Add_Panels_Preview()
        self._bind_events()
        self._bind_pub_sub_events()

    def _Add_Panels_Preview(self):
        self.previews = []
        for i in range(NROWS):
            for j in range(NCOLS):
                p = Preview(self)
                p.preview_clicked.connect(self._on_preview_click)
                self.previews.append(p)
                self.grid.addWidget(p, i, j)

    def _on_preview_click(self, selected_id, item_data, shift_pressed):
        sender = self.sender()
        if self.selected_dicom:
            self.selected_dicom.selected = self.selected_dicom is sender.bitmap_info
            self.selected_panel.select_on = self.selected_panel is sender
            self.selected_panel.Select()
        self.selected_panel = sender
        self.selected_dicom = self.selected_panel.bitmap_info
        self.serie_clicked.emit(selected_id, item_data)

    def _bind_events(self):
        self.scroll.valueChanged.connect(self._on_scroll)

    def _bind_pub_sub_events(self):
        Publisher.subscribe(self.RemovePanel, "Remove preview panel")

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.scroll.setValue(self.scroll.value() - 1)
        elif delta < 0:
            self.scroll.setValue(self.scroll.value() + 1)

    def _on_scroll(self, value):
        if self.displayed_position != value:
            self.displayed_position = value
            self._display_previews()

    def SetBitmapFiles(self, data):
        self.files = []

        bitmap = bitmap_reader.BitmapData()
        bitmap.SetData(data)

        pos = 0
        for d in data:
            d.append(pos)
            info = BitmapInfo(d)
            self.files.append(info)
            pos += 1

        scroll_range = len(self.files) // NCOLS
        if scroll_range * NCOLS < len(self.files):
            scroll_range += 1
        self.scroll.setRange(0, max(0, scroll_range - NROWS))
        self.scroll.setPageStep(NROWS)
        self._display_previews()

    def RemovePanel(self, data):
        for p, f in zip(self.previews, self.files):
            if p.bitmap_info is not None:
                if data in p.bitmap_info.data[0]:
                    self.files.remove(f)
                    p.hide()
                    self._display_previews()
                    Publisher.sendMessage(
                        "Update max of slidebar in single preview image", max_value=len(self.files)
                    )
                    self.update()

        for n, p in enumerate(self.previews):
            if p.bitmap_info is not None:
                if p.isVisible():
                    p.bitmap_info.pos = n

    def _display_previews(self):
        initial = self.displayed_position * NCOLS
        final = initial + NUM_PREVIEWS
        if len(self.files) < final:
            for i in range(final - len(self.files)):
                try:
                    self.previews[-i - 1].hide()
                except IndexError:
                    utils.debug("doesn't exist!")
            self.nhidden_last_display = final - len(self.files)
        else:
            if self.nhidden_last_display:
                for i in range(self.nhidden_last_display):
                    try:
                        self.previews[-i - 1].show()
                    except IndexError:
                        utils.debug("doesn't exist!")
                self.nhidden_last_display = 0

        for f, p in zip(self.files[initial:final], self.previews):
            p.SetBitmapToPreview(f)
            if f.selected:
                self.selected_panel = p

        for f, p in zip(self.files[initial:final], self.previews):
            p.show()


class SingleImagePreview(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.actor = None
        self.__init_gui()
        self.__init_vtk()
        self.__bind_evt_gui()
        self.__bind_pubsub()
        self.dicom_list = []
        self.nimages = 1
        self.current_index = 0
        self.window_width = const.WINDOW_LEVEL[_("Bone")][0]
        self.window_level = const.WINDOW_LEVEL[_("Bone")][1]
        self.ischecked = False

    def __init_vtk(self):
        text_image_size = vtku.TextZero()
        text_image_size.SetPosition(const.TEXT_POS_LEFT_UP)
        text_image_size.SetValue("")
        text_image_size.SetSymbolicSize(FONTSIZE_SMALL)
        self.text_image_size = text_image_size

        text_image_location = vtku.TextZero()
        text_image_location.SetPosition(const.TEXT_POS_LEFT_DOWN)
        text_image_location.SetValue("")
        text_image_location.bottom_pos = True
        text_image_location.SetSymbolicSize(FONTSIZE_SMALL)
        self.text_image_location = text_image_location

        text_patient = vtku.TextZero()
        text_patient.SetPosition(const.TEXT_POS_RIGHT_UP)
        text_patient.SetValue("")
        text_patient.right_pos = True
        text_patient.SetSymbolicSize(FONTSIZE_SMALL)
        self.text_patient = text_patient

        text_acquisition = vtku.TextZero()
        text_acquisition.SetPosition(const.TEXT_POS_RIGHT_DOWN)
        text_acquisition.SetValue("")
        text_acquisition.right_pos = True
        text_acquisition.bottom_pos = True
        text_acquisition.SetSymbolicSize(FONTSIZE_SMALL)
        self.text_acquisition = text_acquisition

        self.renderer = vtkRenderer()
        self.renderer.SetLayer(0)

        cam = self.renderer.GetActiveCamera()

        self.canvas_renderer = vtkRenderer()
        self.canvas_renderer.SetLayer(1)
        self.canvas_renderer.SetActiveCamera(cam)
        self.canvas_renderer.SetInteractive(0)
        self.canvas_renderer.PreserveDepthBufferOn()

        style = vtkInteractorStyleImage()

        self.interactor = QVTKRenderWindowInteractor(self.panel, size=(340, 340))
        if hasattr(self.interactor, "SetRenderWhenDisabled"):
            self.interactor.SetRenderWhenDisabled(True)
        self.interactor.GetRenderWindow().SetNumberOfLayers(2)
        self.interactor.GetRenderWindow().AddRenderer(self.renderer)
        self.interactor.GetRenderWindow().AddRenderer(self.canvas_renderer)
        self.interactor.SetInteractorStyle(style)
        self.interactor.Render()

        self.canvas = CanvasRendererCTX(self, self.renderer, self.canvas_renderer)
        self.canvas.draw_list.append(self.text_image_size)
        self.canvas.draw_list.append(self.text_image_location)
        self.canvas.draw_list.append(self.text_patient)
        self.canvas.draw_list.append(self.text_acquisition)

        panel_sizer = QVBoxLayout(self.panel)
        panel_sizer.addWidget(self.interactor, 1)

    def __init_gui(self):
        self.panel = QWidget(self)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 99)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)

        self.checkbox = QCheckBox(_("Auto-play"), self)

        in_sizer = QHBoxLayout()
        in_sizer.addWidget(self.slider, 1)
        in_sizer.addWidget(self.checkbox, 0)

        sizer = QVBoxLayout(self)
        sizer.addWidget(self.panel, 20)
        sizer.addLayout(in_sizer, 1)

    def __bind_evt_gui(self):
        self.slider.valueChanged.connect(self.OnSlider)
        self.checkbox.stateChanged.connect(self.OnCheckBox)

    def __bind_pubsub(self):
        Publisher.subscribe(self.ShowBitmapByPosition, "Set bitmap in preview panel")
        Publisher.subscribe(
            self.UpdateMaxValueSliderBar, "Update max of slidebar in single preview image"
        )
        Publisher.subscribe(self.ShowBlackSlice, "Show black slice in single preview image")

    def ShowBitmapByPosition(self, pos):
        if pos is not None:
            self.ShowSlice(int(pos))

    def OnSlider(self, pos):
        self.ShowSlice(pos)

    def OnCheckBox(self, state):
        self.ischecked = state == Qt.Checked.value if hasattr(Qt.Checked, "value") else bool(state)
        if self.ischecked:
            QTimer.singleShot(0, self.OnRun)

    def OnRun(self):
        pos = self.slider.value()
        pos += 1
        if not (self.nimages - pos):
            pos = 0
        self.slider.setValue(pos)
        self.ShowSlice(pos)
        time.sleep(0.2)
        if self.ischecked:
            QApplication.processEvents()
            QTimer.singleShot(0, self.OnRun)

    def SetBitmapFiles(self, data):
        self.bitmap_list = data
        self.current_index = 0
        self.nimages = len(data)
        self.slider.setMaximum(self.nimages - 1)
        self.slider.setValue(0)
        self.ShowSlice()

    def UpdateMaxValueSliderBar(self, max_value):
        self.slider.setMaximum(max_value - 1)

    def ShowBlackSlice(self, pub_sub):
        n_array = numpy.zeros((100, 100))

        self.text_image_size.SetValue("")

        image = converters.to_vtk(n_array, spacing=(1, 1, 1), slice_number=1, orientation="AXIAL")

        colorer = vtkImageMapToWindowLevelColors()
        colorer.SetInputData(image)
        colorer.Update()

        if self.actor is None:
            self.actor = vtkImageActor()
            self.renderer.AddActor(self.actor)

        self.actor.SetInputData(colorer.GetOutput())
        self.renderer.ResetCamera()
        self.interactor.Render()

        self.slider.setValue(0)

    def ShowSlice(self, index=0):
        bitmap = self.bitmap_list[index]

        value = STR_SIZE % (bitmap[3], bitmap[4])
        self.text_image_size.SetValue(value)

        value1 = ""
        value2 = ""

        value = f"{value1}\n{value2}"
        self.text_image_location.SetValue(value)

        self.text_patient.SetValue("")
        self.text_acquisition.SetValue("")

        n_array = bitmap_reader.ReadBitmap(bitmap[0])

        image = converters.to_vtk(n_array, spacing=(1, 1, 1), slice_number=1, orientation="AXIAL")

        window_level = n_array.max() / 2
        window_width = n_array.max()

        colorer = vtkImageMapToWindowLevelColors()
        colorer.SetInputData(image)
        colorer.SetWindow(float(window_width))
        colorer.SetLevel(float(window_level))
        colorer.Update()

        if self.actor is None:
            self.actor = vtkImageActor()
            self.renderer.AddActor(self.actor)

        self.actor.SetInputData(colorer.GetOutput())
        self.renderer.ResetCamera()
        self.interactor.Render()

        self.slider.setValue(index)
