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

# -*- coding: UTF-8 -*-

import sys
import time

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
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkImagingColor import vtkImageMapToWindowLevelColors
from vtkmodules.vtkImagingCore import vtkImageFlip
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkIOImage import vtkPNGReader
from vtkmodules.vtkRenderingCore import vtkImageActor, vtkRenderer

import invesalius.constants as const
import invesalius.data.vtk_utils as vtku
import invesalius.reader.dicom_reader as dicom_reader
import invesalius.utils as utils
from invesalius.data import converters, imagedata_utils
from invesalius.gui.widgets.canvas_renderer import CanvasRendererCTX
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher

if sys.platform == "win32":
    try:
        import win32api

        _has_win32api = True
    except ImportError:
        _has_win32api = False
else:
    _has_win32api = False

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


class DicomInfo:
    """
    Keep the informations and the image used by preview.
    """

    def __init__(self, id, dicom, title, subtitle, n=0):
        self.id = id
        self.dicom = dicom
        self.title = title
        self.subtitle = subtitle
        self._preview = None
        self.selected = False
        self.filename = ""
        self._slice = n

    @property
    def preview(self):
        if not self._preview:
            if isinstance(self.dicom.image.thumbnail_path, list):
                self._preview = QImage(self.dicom.image.thumbnail_path[self._slice])
            else:
                self._preview = QImage(self.dicom.image.thumbnail_path)
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
        self.dicom_info = None
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

    def SetDicomToPreview(self, dicom_info):
        if self.dicom_info:
            self.dicom_info.release_thumbnail()

        self.dicom_info = dicom_info
        self.SetTitle(dicom_info.title)
        self.SetSubtitle(dicom_info.subtitle)
        self.ID = dicom_info.id
        image = dicom_info.preview
        self.image_viewer.SetImage(image)
        self.data = dicom_info.id
        self.select_on = dicom_info.selected
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
            self.dicom_info.selected = True
            self.Select()
            self.preview_clicked.emit(self.dicom_info.id, self.dicom_info.dicom, shift_pressed)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.preview_dblclicked.emit(self.dicom_info.id, self.dicom_info.dicom)
        super().mouseDoubleClickEvent(event)

    def Select(self, on=True):
        if self.select_on:
            c = self.palette().color(QPalette.Highlight)
            self._set_background((c.red(), c.green(), c.blue()))
        else:
            self._set_background(PREVIEW_BACKGROUND)
        self.update()


class DicomPreviewSeries(QWidget):
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
            self.selected_dicom.selected = self.selected_dicom is sender.dicom_info
            self.selected_panel.select_on = self.selected_panel is sender
            self.selected_panel.Select()
        self.selected_panel = sender
        self.selected_dicom = self.selected_panel.dicom_info
        self.serie_clicked.emit(selected_id, item_data)

    def _bind_events(self):
        self.scroll.valueChanged.connect(self._on_scroll)

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

    def SetPatientGroups(self, patient):
        self.files = []
        self.displayed_position = 0
        self.nhidden_last_display = 0
        group_list = patient.GetGroups()
        self.group_list = group_list
        n = 0
        for group in group_list:
            info = DicomInfo(
                (group.dicom.patient.id, group.dicom.acquisition.serie_number),
                group.dicom,
                group.title,
                _("%d images") % (group.nslices),
            )
            self.files.append(info)
            n += 1

        scroll_range = len(self.files) // NCOLS
        if scroll_range * NCOLS < len(self.files):
            scroll_range += 1
        self.scroll.setRange(0, max(0, scroll_range - NROWS))
        self.scroll.setPageStep(NROWS)
        self._display_previews()

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
            p.SetDicomToPreview(f)
            if f.selected:
                self.selected_panel = p

        for f, p in zip(self.files[initial:final], self.previews):
            p.show()


class DicomPreviewSlice(QWidget):
    """A dicom preview panel"""

    slice_clicked = Signal(object, object)

    def __init__(self, parent):
        super().__init__(parent)
        self.displayed_position = 0
        self.nhidden_last_display = 0
        self.selected_dicom = None
        self.selected_panel = None
        self.first_selection = None
        self.last_selection = None
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

    def _Add_Panels_Preview(self):
        self.previews = []
        for i in range(NROWS):
            for j in range(NCOLS):
                p = Preview(self)
                p.preview_clicked.connect(self.OnPreviewClick)
                self.previews.append(p)
                self.grid.addWidget(p, i, j)

    def _bind_events(self):
        self.scroll.valueChanged.connect(self._on_scroll)

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

    def SetDicomDirectory(self, directory):
        utils.debug(f"Setting Dicom Directory {directory}")
        self.directory = directory
        self.series = dicom_reader.GetSeries(directory)[0]

    def SetPatientGroups(self, patient):
        self.group_list = patient.GetGroups()

    def SetDicomSerie(self, pos):
        self.files = []
        self.displayed_position = 0
        self.nhidden_last_display = 0
        group = self.group_list[pos]
        self.group = group
        dicom_files = group.GetHandSortedList()
        n = 0
        for dicom in dicom_files:
            if isinstance(dicom.image.thumbnail_path, list):
                _slice = 0
                for thumbnail in dicom.image.thumbnail_path:
                    print(thumbnail)
                    info = DicomInfo(
                        n, dicom, _("Image %d") % (n), f"{dicom.image.position[2]:.2f}", _slice
                    )
                    self.files.append(info)
                    n += 1
                    _slice += 1
            else:
                info = DicomInfo(
                    n,
                    dicom,
                    _("Image %d") % (dicom.image.number),
                    f"{dicom.image.position[2]:.2f}",
                )
                self.files.append(info)
                n += 1

        scroll_range = len(self.files) / NCOLS
        if scroll_range * NCOLS < len(self.files):
            scroll_range += 1
        self.scroll.setRange(0, max(0, int(scroll_range) - NROWS))
        self.scroll.setPageStep(NROWS)

        self._display_previews()

    def SetDicomGroup(self, group):
        self.files = []
        self.displayed_position = 0
        self.nhidden_last_display = 0
        dicom_files = group.GetHandSortedList()
        n = 0
        for dicom in dicom_files:
            if isinstance(dicom.image.thumbnail_path, list):
                _slice = 0
                for thumbnail in dicom.image.thumbnail_path:
                    print(thumbnail)
                    info = DicomInfo(
                        n, dicom, _("Image %d") % int(n), f"{dicom.image.position[2]:.2f}", _slice
                    )
                    self.files.append(info)
                    n += 1
                    _slice += 1
            else:
                info = DicomInfo(
                    n,
                    dicom,
                    _("Image %d") % int(dicom.image.number),
                    f"{dicom.image.position[2]:.2f}",
                )
                self.files.append(info)
                n += 1

        scroll_range = len(self.files) // NCOLS
        if scroll_range * NCOLS < len(self.files):
            scroll_range += 1
        self.scroll.setRange(0, max(0, scroll_range - NROWS))
        self.scroll.setPageStep(NROWS)

        self._display_previews()

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
            p.SetDicomToPreview(f)
            if f.selected:
                self.selected_panel = p

        for f, p in zip(self.files[initial:final], self.previews):
            p.show()

    def OnPreviewClick(self, dicom_id, item_data, shift_pressed):
        sender = self.sender()

        if self.first_selection is None:
            self.first_selection = dicom_id

        if self.last_selection is None:
            self.last_selection = dicom_id

        if shift_pressed:
            if dicom_id < self.first_selection and dicom_id < self.last_selection:
                self.first_selection = dicom_id
            else:
                self.last_selection = dicom_id
        else:
            self.first_selection = dicom_id
            self.last_selection = dicom_id

            for i in range(len(self.files)):
                if i == dicom_id:
                    self.files[i].selected = True
                else:
                    self.files[i].selected = False

        if self.selected_dicom:
            self.selected_dicom.selected = self.selected_dicom is sender.dicom_info
            self.selected_panel.select_on = self.selected_panel is sender

            if self.first_selection != self.last_selection:
                for i in range(len(self.files)):
                    if i >= self.first_selection and i <= self.last_selection:
                        self.files[i].selected = True
                    else:
                        self.files[i].selected = False
            else:
                self.selected_panel.Select()

        self._display_previews()
        self.selected_panel = sender
        self.selected_dicom = self.selected_panel.dicom_info
        self.slice_clicked.emit(dicom_id, item_data)

        Publisher.sendMessage(
            "Selected Import Images", selection=(self.first_selection, self.last_selection)
        )


class SingleImagePreview(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.actor = None
        self.__init_gui()
        self.__bind_evt_gui()
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

    def __init_gui(self):
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 99)
        self.slider.setValue(0)
        self.slider.setTickPosition(QSlider.TicksBelow)
        self.slider.setTickInterval(1)

        self.checkbox = QCheckBox(_("Auto-play"), self)

        self.interactor = QVTKRenderWindowInteractor(self)
        if hasattr(self.interactor, "SetRenderWhenDisabled"):
            self.interactor.SetRenderWhenDisabled(True)

        in_sizer = QHBoxLayout()
        in_sizer.addWidget(self.slider, 1)
        in_sizer.addWidget(self.checkbox, 0)

        sizer = QVBoxLayout(self)
        sizer.addWidget(self.interactor, 1)
        sizer.addLayout(in_sizer, 0)

    def __bind_evt_gui(self):
        self.slider.valueChanged.connect(self.OnSlider)
        self.checkbox.stateChanged.connect(self.OnCheckBox)

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

    def SetDicomGroup(self, group):
        self.dicom_list = group.GetHandSortedList()
        self.current_index = 0
        if len(self.dicom_list) > 1:
            self.nimages = len(self.dicom_list)
        else:
            self.nimages = self.dicom_list[0].image.number_of_frames
        self.slider.setMaximum(self.nimages - 1)
        self.slider.setValue(0)
        self.slider.setTickInterval(1)
        self.ShowSlice()

    def ShowSlice(self, index=0):
        try:
            dicom = self.dicom_list[index]
        except IndexError:
            dicom = self.dicom_list[0]

        if self.actor is None:
            self.__init_vtk()

        value = STR_SIZE % (dicom.image.size[0], dicom.image.size[1])
        self.text_image_size.SetValue(value)

        if not (dicom.image.spacing):
            value1 = ""
        else:
            value1 = STR_SPC % (dicom.image.spacing[2])

        if dicom.image.orientation_label == "AXIAL":
            value2 = STR_LOCAL % (dicom.image.position[2])
        elif dicom.image.orientation_label == "CORONAL":
            value2 = STR_LOCAL % (dicom.image.position[1])
        elif dicom.image.orientation_label == "SAGITTAL":
            value2 = STR_LOCAL % (dicom.image.position[0])
        else:
            value2 = ""

        value = f"{value1}\n{value2}"
        self.text_image_location.SetValue(value)

        value = STR_PATIENT % (dicom.patient.id, dicom.acquisition.protocol_name)
        self.text_patient.SetValue(value)

        value = STR_ACQ % (dicom.acquisition.date, dicom.acquisition.time)
        self.text_acquisition.SetValue(value)

        if isinstance(dicom.image.thumbnail_path, list):
            reader = vtkPNGReader()
            if _has_win32api:
                reader.SetFileName(
                    win32api.GetShortPathName(dicom.image.thumbnail_path[index]).encode(
                        const.FS_ENCODE
                    )
                )
            else:
                reader.SetFileName(dicom.image.thumbnail_path[index])
            reader.Update()

            image = reader.GetOutput()
        else:
            filename = dicom.image.file
            if _has_win32api:
                filename = win32api.GetShortPathName(filename).encode(const.FS_ENCODE)

            np_image = imagedata_utils.read_dcm_slice_as_np2(filename)
            vtk_image = converters.to_vtk(np_image, dicom.image.spacing, 0, "AXIAL")

            window_level = dicom.image.level
            window_width = dicom.image.window
            colorer = vtkImageMapToWindowLevelColors()
            colorer.SetInputData(vtk_image)
            colorer.SetWindow(float(window_width))
            colorer.SetLevel(float(window_level))
            colorer.Update()

            image = colorer.GetOutput()

        flip = vtkImageFlip()
        flip.SetInputData(image)
        flip.SetFilteredAxis(1)
        flip.FlipAboutOriginOn()
        flip.ReleaseDataFlagOn()
        flip.Update()

        if self.actor is None:
            self.actor = vtkImageActor()
            self.renderer.AddActor(self.actor)

        self.canvas.modified = True

        self.actor.SetInputData(flip.GetOutput())
        self.renderer.ResetCamera()
        self.interactor.Render()

        self.slider.setValue(index)
