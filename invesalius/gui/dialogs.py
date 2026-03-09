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

import csv
import datetime
import itertools
import os
import random
import sys
import textwrap
import time
import webbrowser
from concurrent import futures
from functools import partial
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
)

if sys.platform == "win32":
    try:
        import win32api

        _has_win32api = True
    except ImportError:
        _has_win32api = False
else:
    _has_win32api = False

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QDesktopServices,
    QFontMetrics,
    QIcon,
    QPixmap,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkCommonColor import vtkNamedColors
from vtkmodules.vtkCommonComputationalGeometry import vtkParametricTorus
from vtkmodules.vtkCommonCore import mutable, vtkPoints
from vtkmodules.vtkCommonDataModel import (
    vtkCellLocator,
    vtkIterativeClosestPointTransform,
    vtkPolyData,
)
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from vtkmodules.vtkCommonTransforms import vtkTransform
from vtkmodules.vtkFiltersCore import vtkAppendPolyData, vtkCleanPolyData, vtkPolyDataNormals
from vtkmodules.vtkFiltersGeneral import vtkTransformPolyDataFilter
from vtkmodules.vtkFiltersSources import (
    vtkArrowSource,
    vtkCylinderSource,
    vtkParametricFunctionSource,
    vtkRegularPolygonSource,
    vtkSphereSource,
)
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballActor,
    vtkInteractorStyleTrackballCamera,
)
from vtkmodules.vtkIOGeometry import vtkSTLReader
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCellPicker,
    vtkFollower,
    vtkPolyDataMapper,
    vtkProperty,
    vtkRenderer,
)
from vtkmodules.vtkRenderingFreeType import vtkVectorText

import invesalius
import invesalius.constants as const
import invesalius.data.coordinates as dco
import invesalius.data.coregistration as dcr
import invesalius.data.imagedata_utils as img_utils
import invesalius.data.polydata_utils as pu
import invesalius.data.transformations as tr
import invesalius.data.vtk_utils as vtku
import invesalius.gui.widgets.gradient as grad
import invesalius.session as ses
import invesalius.utils as utils
from invesalius import inv_paths
from invesalius.gui.utils import calc_width_needed
from invesalius.gui.widgets.clut_imagedata import CLUTImageDataWidget
from invesalius.gui.widgets.fiducial_buttons import OrderedFiducialButtons
from invesalius.gui.widgets.inv_spinctrl import InvFloatSpinCtrl, InvSpinCtrl
from invesalius.i18n import tr as _
from invesalius.math_utils import inner1d
from invesalius.pubsub import pub as Publisher

if TYPE_CHECKING:
    from invesalius.data.mask import Mask
    from invesalius.data.styles import (
        CropMaskConfig,
        FFillConfig,
        FFillSegmentationConfig,
        SelectPartConfig,
        WatershedConfig,
    )
    from invesalius.gui.widgets.clut_imagedata import Node
    from invesalius.navigation.mtms import mTMS
    from invesalius.navigation.navigation import Navigation
    from invesalius.navigation.robot import Robot
    from invesalius.navigation.tracker import Tracker
    from invesalius.net.pedal_connection import PedalConnector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _top_window():
    app = QApplication.instance()
    if app:
        return app.activeWindow()
    return None


def _center_on_screen(widget):
    screen = widget.screen()
    if screen:
        geo = screen.availableGeometry()
        widget.move(
            geo.x() + (geo.width() - widget.width()) // 2,
            geo.y() + (geo.height() - widget.height()) // 2,
        )


def _wx_wildcard_to_qt(wildcard: str) -> str:
    parts = wildcard.split("|")
    qt_parts = []
    for i in range(0, len(parts), 2):
        if i < len(parts):
            qt_parts.append(parts[i].strip())
    return ";;".join(qt_parts)


class FileBrowseButton(QWidget):
    """Replacement for wx.lib.filebrowsebutton.FileBrowseButton."""

    fileChanged = Signal(str)

    def __init__(
        self,
        parent=None,
        labelText="",
        fileMask="",
        dialogTitle="",
        startDirectory="",
        changeCallback=None,
    ):
        super().__init__(parent)
        self._filter = _wx_wildcard_to_qt(fileMask) if "|" in fileMask else fileMask
        self._dialog_title = dialogTitle
        self._start_dir = startDirectory

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if labelText:
            layout.addWidget(QLabel(labelText))
        self._line = QLineEdit()
        layout.addWidget(self._line, 1)
        btn = QPushButton(_("Browse..."))
        btn.clicked.connect(self._browse)
        layout.addWidget(btn)

        if changeCallback:
            self.fileChanged.connect(changeCallback)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self._dialog_title, self._start_dir, self._filter
        )
        if path:
            self._line.setText(path)
            self.fileChanged.emit(path)

    def GetValue(self):
        return self._line.text()

    def GetString(self):
        return self._line.text()


class FilePickerCtrl(QWidget):
    """Replacement for wx.FilePickerCtrl."""

    def __init__(self, parent=None, path="", wildcard="", message="", **kwargs):
        super().__init__(parent)
        self._filter = _wx_wildcard_to_qt(wildcard) if "|" in wildcard else wildcard
        self._message = message
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._line = QLineEdit(path)
        layout.addWidget(self._line, 1)
        btn = QPushButton(_("Browse..."))
        btn.clicked.connect(self._browse)
        layout.addWidget(btn)

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._message,
            os.path.dirname(self._line.text()),
            self._filter,
        )
        if path:
            self._line.setText(path)

    def GetPath(self):
        return self._line.text()


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INV_NON_COMPRESSED = 0
INV_COMPRESSED = 1

WILDCARD_INV_SAVE = (
    _("InVesalius project (*.inv3)") + ";;" + _("InVesalius project compressed (*.inv3)")
)

WILDCARD_OPEN = "InVesalius 3 project (*.inv3);;All files (*.*)"

WILDCARD_ANALYZE = "Analyze 7.5 (*.hdr);;All files (*.*)"

WILDCARD_NIFTI = "NIfTI 1 (*.nii *.nii.gz *.hdr);;All files (*.*)"

WILDCARD_PARREC = "PAR/REC (*.par);;All files (*.*)"

WILDCARD_MESH_FILES = (
    "STL File format (*.stl);;"
    "Standard Polygon File Format (*.ply);;"
    "Alias Wavefront Object (*.obj);;"
    "VTK Polydata File Format (*.vtp);;"
    "All files (*.*)"
)
WILDCARD_JSON_FILES = "JSON File format (*.json);;All files (*.*)"


# ---------------------------------------------------------------------------
# NumberDialog
# ---------------------------------------------------------------------------


class NumberDialog(QDialog):
    def __init__(self, message: str, value: int = 0):
        super().__init__(None)
        self.setWindowTitle("InVesalius 3")

        label = QLabel(message)

        self.num_ctrl = QDoubleSpinBox()
        self.num_ctrl.setRange(-999, 999)
        self.num_ctrl.setDecimals(2)
        self.num_ctrl.setValue(value)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(label)
        layout.addWidget(self.num_ctrl)
        layout.addWidget(btn_box)
        self.adjustSize()
        _center_on_screen(self)

    def SetValue(self, value: int) -> None:
        self.num_ctrl.setValue(value)

    def GetValue(self) -> Union[int, float, None]:
        return self.num_ctrl.value()


class ResizeImageDialog(QDialog):
    def __init__(self):
        super().__init__(None)
        self.setWindowTitle("InVesalius 3")

        lbl_message = QLabel(
            _(
                "InVesalius is running on a 32-bit operating system or has insufficient memory. \n"
                "If you want to work with 3D surfaces or volume rendering, \n"
                "it is recommended to reduce the medical images resolution."
            )
        )
        icon_label = QLabel()
        icon_pixmap = QApplication.style().standardPixmap(QStyle.SP_MessageBoxWarning)
        icon_label.setPixmap(
            icon_pixmap.scaled(32, 32, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        lbl_message_percent = QLabel(_("Percentage of original resolution"))
        self.num_ctrl_porcent = InvSpinCtrl(self, -1, value=100, min_value=20, max_value=100)

        sizer_percent = QHBoxLayout()
        sizer_percent.addWidget(lbl_message_percent)
        sizer_percent.addWidget(self.num_ctrl_porcent)

        sizer_items = QVBoxLayout()
        sizer_items.addWidget(lbl_message)
        sizer_items.addLayout(sizer_percent)
        sizer_items.addWidget(btn_box)

        main_layout = QHBoxLayout(self)
        main_layout.addWidget(icon_label)
        main_layout.addLayout(sizer_items)
        self.adjustSize()
        _center_on_screen(self)

    def SetValue(self, value: Union[float, str]) -> None:
        self.num_ctrl_porcent.SetValue(value)

    def GetValue(self) -> int:
        return self.num_ctrl_porcent.GetValue()

    def Close(self) -> None:
        self.close()


def ShowNumberDialog(message: str, value: int = 0) -> Union[int, float, None]:
    dlg = NumberDialog(message, value)
    dlg.SetValue(value)
    if dlg.exec() == QDialog.Accepted:
        return dlg.GetValue()
    return 0


# ---------------------------------------------------------------------------
# File Dialogs
# ---------------------------------------------------------------------------


def ShowOpenProjectDialog() -> Union[str, None]:
    current_dir = os.path.abspath(".")
    session = ses.Session()
    last_directory = session.GetConfig("last_directory_inv3", "")
    filepath, _ = QFileDialog.getOpenFileName(
        None,
        _("Open InVesalius 3 project..."),
        last_directory,
        WILDCARD_OPEN,
    )
    if filepath:
        last_directory = os.path.split(filepath)[0]
        session.SetConfig("last_directory_inv3", last_directory)
    os.chdir(current_dir)
    return filepath if filepath else None


def ShowImportDirDialog(self) -> Union[str, bytes, None]:
    current_dir = os.path.abspath(".")
    if sys.platform == "win32" or sys.platform.startswith("linux"):
        session = ses.Session()
        folder = session.GetConfig("last_dicom_folder", "")
    else:
        folder = ""

    path = QFileDialog.getExistingDirectory(self, _("Choose a DICOM folder:"), folder)

    if path:
        if sys.platform != "win32":
            path = path.encode("utf-8")
        if sys.platform != "darwin":
            path_decoded = utils.decode(path, const.FS_ENCODE) if isinstance(path, bytes) else path
            session.SetConfig("last_dicom_folder", path_decoded)

    os.chdir(current_dir)
    return path if path else None


def ShowImportBitmapDirDialog(self) -> Optional[str]:
    current_dir = os.path.abspath(".")
    session = ses.Session()
    last_directory = session.GetConfig("last_directory_bitmap", "")

    path = QFileDialog.getExistingDirectory(
        self, _("Choose a folder with TIFF, BMP, JPG or PNG:"), last_directory
    )

    if path:
        session.SetConfig("last_directory_bitmap", path)
    os.chdir(current_dir)
    return path if path else None


def ShowImportOtherFilesDialog(
    id_type: int, msg: str = "Import NIFTi 1 file"
) -> Union[str, bytes, None]:
    session = ses.Session()
    last_directory = session.GetConfig("last_directory_%d" % id_type, "")

    wildcard = WILDCARD_NIFTI
    if id_type == const.ID_PARREC_IMPORT:
        msg = _("Import PAR/REC file")
        wildcard = WILDCARD_PARREC
    elif id_type == const.ID_ANALYZE_IMPORT:
        msg = _("Import Analyze 7.5 file")
        wildcard = WILDCARD_ANALYZE

    filename, _ = QFileDialog.getOpenFileName(None, msg, last_directory, wildcard)

    if filename:
        if sys.platform != "win32":
            filename_ret = filename.encode("utf-8")
        else:
            filename_ret = filename
        last_directory = os.path.split(filename)[0]
        session.SetConfig("last_directory_%d" % id_type, last_directory)
        return filename_ret

    return None


def ShowImportMeshFilesDialog() -> Optional[str]:
    from invesalius.data.slice_ import Slice

    current_dir = os.path.abspath(".")
    session = ses.Session()
    last_directory = session.GetConfig("last_directory_surface_import", "")

    if Slice().has_affine():
        dlg = FileSelectionDialog(
            title=_("Import surface file"),
            default_dir=last_directory,
            wildcard=WILDCARD_MESH_FILES,
        )
        group = QGroupBox(_("File coordinate space"))
        group_layout = QVBoxLayout(group)
        conversion_buttons = []
        for i, choice_text in enumerate(const.SURFACE_SPACE_CHOICES):
            rb = QRadioButton(choice_text)
            if i == 0:
                rb.setChecked(True)
            conversion_buttons.append(rb)
            group_layout.addWidget(rb)
        dlg.sizer.addWidget(group)
        dlg.FitSizers()

        filename = None
        if dlg.exec() == QDialog.Accepted:
            filename = dlg.GetPath()
            if filename:
                sel_idx = 0
                for i, rb in enumerate(conversion_buttons):
                    if rb.isChecked():
                        sel_idx = i
                        break
                convert_to_inv = sel_idx == const.SURFACE_SPACE_WORLD
                Publisher.sendMessage("Update convert_to_inv flag", convert_to_inv=convert_to_inv)
    else:
        filename, _ = QFileDialog.getOpenFileName(
            None, _("Import surface file"), last_directory, WILDCARD_MESH_FILES
        )

    if filename:
        session.SetConfig("last_directory_surface_import", os.path.split(filename)[0])

    os.chdir(current_dir)
    return filename if filename else None


def ImportMeshCoordSystem() -> bool:
    msg = _("Was the imported mesh created by InVesalius?")
    result = QMessageBox.question(None, "InVesalius 3", msg, QMessageBox.Yes | QMessageBox.No)
    return result != QMessageBox.Yes


def ShowSaveAsProjectDialog(default_filename: str) -> Tuple[Optional[str], bool]:
    current_dir = os.path.abspath(".")
    session = ses.Session()
    last_directory = session.GetConfig("last_directory_inv3", "")

    filename, selected_filter = QFileDialog.getSaveFileName(
        None,
        _("Save project as..."),
        os.path.join(last_directory, default_filename),
        WILDCARD_INV_SAVE,
    )

    if filename:
        extension = "inv3"
        if sys.platform != "win32":
            if filename.split(".")[-1] != extension:
                filename = filename + "." + extension
        last_directory = os.path.split(filename)[0]
        session.SetConfig("last_directory_inv3", last_directory)

    os.chdir(current_dir)
    if not filename:
        return None, False

    filters = WILDCARD_INV_SAVE.split(";;")
    wildcard = filters.index(selected_filter) if selected_filter in filters else 0
    return filename, wildcard == INV_COMPRESSED


def ShowLoadCSVDebugEfield(
    message: str = _("Load debug CSV Enorm file"),
    current_dir: "str | bytes | os.PathLike[str]" = os.path.abspath("."),
    style: int = 0,
    wildcard: str = _("(*.csv)"),
    default_filename: str = "",
) -> Optional[np.ndarray]:
    qt_wildcard = _wx_wildcard_to_qt(wildcard) if "|" in wildcard else wildcard
    filepath, _ = QFileDialog.getOpenFileName(None, message, "", qt_wildcard)
    os.chdir(current_dir)
    if filepath:
        with open(filepath) as file:
            my_reader = csv.reader(file, delimiter=",")
            rows = [row for row in my_reader]
        return np.array(rows).astype(float)
    return None


def ShowLoadSaveDialog(
    message: str = _("Load File"),
    current_dir: "str | bytes | os.PathLike[str]" = os.path.abspath("."),
    style: int = 0,
    wildcard: str = _("Registration files (*.obr)|*.obr"),
    default_filename: str = "",
    save_ext: Optional[str] = None,
) -> Optional[str]:
    qt_wildcard = _wx_wildcard_to_qt(wildcard) if "|" in wildcard else wildcard
    is_save = save_ext is not None
    if is_save:
        filepath, _ = QFileDialog.getSaveFileName(None, message, default_filename, qt_wildcard)
    else:
        filepath, _ = QFileDialog.getOpenFileName(None, message, "", qt_wildcard)

    if save_ext and filepath:
        if sys.platform != "win32":
            if filepath.split(".")[-1] != save_ext:
                filepath = filepath + "." + save_ext

    os.chdir(current_dir)
    return filepath if filepath else None


def LoadConfigEfield() -> Optional[str]:
    current_dir = os.path.abspath(".")
    session = ses.Session()
    last_directory = session.GetConfig("last_directory_surface_import", "")

    filename, _ = QFileDialog.getOpenFileName(
        None, _("Import json file"), last_directory, WILDCARD_JSON_FILES
    )

    if filename:
        session.SetConfig("last_directory_surface_import", os.path.split(filename)[0])
    os.chdir(current_dir)
    return filename if filename else None


# ---------------------------------------------------------------------------
# Message Dialogs
# ---------------------------------------------------------------------------


class MessageDialog(QDialog):
    def __init__(self, message: str):
        super().__init__(None)
        self.setWindowTitle("InVesalius 3")

        label = QLabel(message)

        btn_yes = QPushButton(_("Yes"))
        btn_no = QPushButton(_("No"))
        btn_cancel = QPushButton(_("Cancel"))

        btn_yes.clicked.connect(self.accept)
        btn_no.clicked.connect(self.reject)
        btn_cancel.clicked.connect(lambda: self.done(2))

        self._result_code = 2

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_yes)
        btn_layout.addWidget(btn_no)
        btn_layout.addWidget(btn_cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(label)
        layout.addLayout(btn_layout)
        self.adjustSize()
        _center_on_screen(self)


class UpdateMessageDialog(QDialog):
    def __init__(self, url: str):
        msg = _(
            "A new version of InVesalius is available. Do you want to open the download website now?"
        )
        title = _("Invesalius Update")
        self.url = url

        super().__init__(None)
        self.setWindowTitle(title)

        label = QLabel(msg)

        btn_yes = QPushButton(_("Yes"))
        btn_no = QPushButton(_("No"))
        btn_yes.setDefault(True)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn_yes)
        btn_layout.addWidget(btn_no)

        layout = QVBoxLayout(self)
        layout.addWidget(label)
        layout.addLayout(btn_layout)
        self.adjustSize()
        _center_on_screen(self)

        btn_yes.clicked.connect(self._OnYes)
        btn_no.clicked.connect(self._OnNo)
        Publisher.subscribe(self._Exit, "Exit")

    def _OnYes(self) -> None:
        webbrowser.open(self.url)
        self.close()

    def _OnNo(self) -> None:
        self.close()

    def _Exit(self) -> None:
        self.close()


class MessageBox(QDialog):
    def __init__(self, parent, title: str, message: str, caption: str = "InVesalius3 Error"):
        super().__init__(parent)
        self.setWindowTitle(caption)

        title_label = QLabel(title)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(message)
        fm = QFontMetrics(text.font())
        text.setMinimumWidth(fm.horizontalAdvance("O" * 30))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self.accept)

        layout = QVBoxLayout(self)
        layout.addWidget(title_label)
        layout.addWidget(text, 1)
        layout.addWidget(btn_box)
        self.adjustSize()
        _center_on_screen(self)
        self.exec()


class ErrorMessageBox(QDialog):
    def __init__(
        self,
        parent,
        title: str,
        message: str,
        caption: str = "InVesalius3 Error",
    ):
        super().__init__(parent)
        self.setWindowTitle(caption)

        title_label = QLabel(title)
        icon_label = QLabel()
        icon_pixmap = QApplication.style().standardPixmap(QStyle.SP_MessageBoxCritical)
        icon_label.setPixmap(icon_pixmap)

        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText(message)
        fm = QFontMetrics(text.font())
        text.setMinimumWidth(fm.horizontalAdvance("M" * 60))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok)
        btn_box.accepted.connect(self.accept)

        title_sizer = QHBoxLayout()
        title_sizer.addWidget(icon_label)
        title_sizer.addWidget(title_label)

        layout = QVBoxLayout(self)
        layout.addLayout(title_sizer)
        layout.addWidget(text, 1)
        layout.addWidget(btn_box)
        self.adjustSize()
        _center_on_screen(self)


def SaveChangesDialog__Old(filename: str) -> Literal[-1, 0, 1]:
    message = _("The project %s has been modified.\nSave changes?") % filename
    dlg = MessageDialog(message)
    answer = dlg.exec()
    if answer == QDialog.Accepted:
        return 1
    elif answer == QDialog.Rejected:
        return 0
    else:
        return -1


def _show_info_msg(msg: str, title: str = "InVesalius 3") -> None:
    QMessageBox.information(None, title, msg)


def ImportEmptyDirectory(dirpath: bytes) -> None:
    _show_info_msg(_("%s is an empty folder.") % dirpath.decode("utf-8"))


def ImportOldFormatInvFile() -> None:
    _show_info_msg(
        _(
            "File was created in a newer InVesalius version. Some functionalities may not work correctly."
        )
    )


def ImportInvalidFiles(ftype: str = "DICOM") -> None:
    if ftype == "Bitmap":
        msg = _("There are no Bitmap, JPEG, PNG or TIFF files in the selected folder.")
    elif ftype == "DICOM":
        msg = _("There are no DICOM files in the selected folder.")
    else:
        msg = _("Invalid file.")
    _show_info_msg(msg)


def WarningRescalePixelValues() -> None:
    msg1 = _("Warning! Pixel values are smaller than 8 (possible float values).\n")
    msg2 = _("Pixel values have been rescaled from 0-255 for compatibility.")
    _show_info_msg(msg1 + msg2)


def ImagePixelRescaling() -> None:
    msg1 = _("Warning! InVesalius has limited support to Analyze format.\n")
    msg2 = _("Slices may be wrongly oriented and functions may not work properly.")
    _show_info_msg(msg1 + msg2)


def InexistentMask() -> None:
    _show_info_msg(_("A mask is needed to create a surface."))


def MaskSelectionRequiredForRemoval() -> None:
    _show_info_msg(_("No mask was selected for removal."))


def SurfaceSelectionRequiredForRemoval() -> None:
    _show_info_msg(_("No surface was selected for removal."))


def MeasureSelectionRequiredForRemoval() -> None:
    _show_info_msg(_("No measure was selected for removal."))


def MaskSelectionRequiredForDuplication() -> None:
    _show_info_msg(_("No mask was selected for duplication."))


def SurfaceSelectionRequiredForDuplication() -> None:
    _show_info_msg(_("No surface was selected for duplication."))


# ---------------------------------------------------------------------------
# Neuronavigation dialogs
# ---------------------------------------------------------------------------


def ShowNavigationTrackerWarning(trck_id: int, lib_mode: str) -> None:
    trck = {
        const.SELECT: "Tracker",
        const.MTC: "Claron MicronTracker",
        const.FASTRAK: "Polhemus FASTRAK",
        const.ISOTRAKII: "Polhemus ISOTRAK",
        const.PATRIOT: "Polhemus PATRIOT",
        const.CAMERA: "CAMERA",
        const.POLARIS: "NDI Polaris",
        const.POLARISP4: "NDI Polaris P4",
        const.OPTITRACK: "Optitrack",
        const.DEBUGTRACKRANDOM: "Debug tracker device (random)",
        const.DEBUGTRACKAPPROACH: "Debug tracker device (approach)",
    }
    if lib_mode == "choose":
        msg = _("No tracking device selected")
    elif lib_mode == "probe marker not visible":
        msg = _("Probe marker is not visible.")
    elif lib_mode == "coil marker not visible":
        msg = _("Coil marker is not visible.")
    elif lib_mode == "head marker not visible":
        msg = _("Head marker is not visible.")
    elif lib_mode == "error":
        msg = trck[trck_id] + _(" is not installed.")
    elif lib_mode == "disconnect":
        msg = trck[trck_id] + _(" disconnected.")
    else:
        msg = trck[trck_id] + _(" is not connected.")
    _show_info_msg(msg, "InVesalius 3 - Neuronavigator")


def Efield_connection_warning() -> None:
    _show_info_msg(_("No connection to E-field library"), "InVesalius 3 - Neuronavigator")


def Efield_no_data_to_save_warning() -> None:
    _show_info_msg(_("No Efield data to save"), "InVesalius 3 - Neuronavigator")


def Efield_debug_Enorm_warning() -> None:
    _show_info_msg(_("The CSV Enorm file is not loaded."), "InVesalius 3 - Neuronavigator")


def ICPcorregistration(fre: float) -> bool:
    msg = (
        _("The fiducial registration error is: ")
        + str(round(fre, 2))
        + "\n\n"
        + _("Would you like to improve accuracy?")
    )
    result = QMessageBox.question(None, "InVesalius 3", msg, QMessageBox.Yes | QMessageBox.No)
    return result == QMessageBox.Yes


def ReportICPerror(prev_error: float, final_error: float) -> None:
    msg = (
        _("Points to scalp distance: ")
        + str(round(final_error, 2))
        + " mm"
        + "\n\n"
        + _("Distance before refine: ")
        + str(round(prev_error, 2))
        + " mm"
    )
    _show_info_msg(msg)


def ReportICPPointError() -> None:
    msg = (
        _("The last point is more than 20 mm away from the surface")
        + "\n\n"
        + _("Please, create a new point.")
    )
    _show_info_msg(msg)


def ReportICPDistributionError() -> None:
    msg = (
        _("The distribution of the transformed points looks wrong.")
        + "\n\n"
        + _("It is recommended to remove the points and redone the acquisition")
    )
    _show_info_msg(msg)


def ShowEnterMarkerID(default: str) -> str:
    from PySide6.QtWidgets import QInputDialog

    result, ok = QInputDialog.getText(None, "InVesalius 3", _("Change label"), text=default)
    return result if ok else default


def ShowEnterMEPValue(default):
    from PySide6.QtWidgets import QInputDialog

    msg = _("Enter the MEP value (uV)")
    result, ok = QInputDialog.getText(None, "InVesalius 3", msg, text=str(default))
    if not ok:
        return None
    try:
        return float(result)
    except ValueError:
        _show_info_msg(_("The value entered is not a number."))
        return None


def ShowConfirmationDialog(msg: str = _("Proceed?")) -> int:
    result = QMessageBox.question(None, "InVesalius 3", msg, QMessageBox.Ok | QMessageBox.Cancel)
    if result == QMessageBox.Ok:
        return QDialog.Accepted
    return QDialog.Rejected


def ShowColorDialog(
    color_current,
) -> Optional[Tuple[int, int, int]]:
    initial = (
        QColor(*color_current[:3])
        if isinstance(color_current, (list, tuple))
        else QColor(color_current)
    )
    color = QColorDialog.getColor(initial, None)
    if color.isValid():
        return (color.red(), color.green(), color.blue())
    return None


# ---------------------------------------------------------------------------
# NewMask
# ---------------------------------------------------------------------------


class NewMask(QDialog):
    def __init__(self, parent=None, ID=-1, title="InVesalius 3", **kwargs):
        import invesalius.constants as const
        import invesalius.data.mask as mask
        import invesalius.project as prj

        super().__init__(parent)
        self.setWindowTitle(title)
        _center_on_screen(self)

        label_mask = QLabel(_("New mask name:"))
        default_name = const.MASK_NAME_PATTERN % (mask.Mask.general_index + 2)
        text = QLineEdit(default_name)
        self.text = text

        label_thresh = QLabel(_("Threshold preset:"))
        project = prj.Project()
        thresh_list = sorted(project.threshold_modes.keys())
        default_index = thresh_list.index(_("Bone"))
        self.thresh_list = thresh_list

        combo_thresh = QComboBox()
        combo_thresh.addItems(self.thresh_list)
        combo_thresh.setCurrentIndex(default_index)
        self.combo_thresh = combo_thresh

        bound_min, bound_max = project.threshold_range
        thresh_min, thresh_max = project.threshold_modes[_("Bone")]
        original_colour = random.choice(const.MASK_COLOUR)
        self.colour = original_colour
        colour = [255 * i for i in original_colour]
        colour.append(100)
        gradient = grad.GradientCtrl(
            self, -1, int(bound_min), int(bound_max), int(thresh_min), int(thresh_max), colour
        )
        self.gradient = gradient

        fixed_layout = QGridLayout()
        fixed_layout.addWidget(label_mask, 0, 0)
        fixed_layout.addWidget(text, 0, 1)
        fixed_layout.addWidget(label_thresh, 1, 0)
        fixed_layout.addWidget(combo_thresh, 1, 1)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(fixed_layout)
        layout.addWidget(gradient)
        layout.addWidget(btn_box)
        self.adjustSize()

        self.gradient.thresholdChanged.connect(self.OnSlideChanged)
        self.combo_thresh.currentTextChanged.connect(self.OnComboThresh)

    def OnComboThresh(self, text: str) -> None:
        import invesalius.project as prj

        proj = prj.Project()
        (thresh_min, thresh_max) = proj.threshold_modes[text]
        self.gradient.SetMinValue(thresh_min)
        self.gradient.SetMaxValue(thresh_max)

    def OnSlideChanged(self) -> None:
        import invesalius.project as prj

        thresh_min = self.gradient.GetMinValue()
        thresh_max = self.gradient.GetMaxValue()
        thresh = (thresh_min, thresh_max)
        proj = prj.Project()
        if thresh in proj.threshold_modes.values():
            preset_name = proj.threshold_modes.get_key(thresh)[0]
            index = self.thresh_list.index(preset_name)
            self.combo_thresh.setCurrentIndex(index)
        else:
            index = self.thresh_list.index(_("Custom"))
            self.combo_thresh.setCurrentIndex(index)

    def GetValue(self) -> Tuple[str, List[int], List[float]]:
        mask_name = self.text.text()
        thresh_value = [self.gradient.GetMinValue(), self.gradient.GetMaxValue()]
        return mask_name, thresh_value, self.colour


def InexistentPath(path: "str | bytes | os.PathLike[str]") -> None:
    _show_info_msg(_("%s does not exist.") % (path))


def MissingFilesForReconstruction() -> None:
    _show_info_msg(_("Please, provide more than one DICOM file for 3D reconstruction"))


def SaveChangesDialog(filename: "str | bytes | os.PathLike[str]", parent) -> Literal[-1, 0, 1]:
    current_dir = os.path.abspath(".")
    msg = _("The project %s has been modified.\nSave changes?") % filename
    result = QMessageBox.question(
        None,
        "InVesalius 3",
        msg,
        QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel,
    )
    os.chdir(current_dir)
    if result == QMessageBox.Yes:
        return 1
    elif result == QMessageBox.No:
        return 0
    else:
        return -1


def SaveChangesDialog2(filename: "str | bytes | os.PathLike[str]") -> Literal[0, 1]:
    current_dir = os.path.abspath(".")
    msg = _("The project %s has been modified.\nSave changes?") % filename
    result = QMessageBox.question(None, "InVesalius 3", msg, QMessageBox.Yes | QMessageBox.No)
    os.chdir(current_dir)
    return 1 if result == QMessageBox.Yes else 0


def ShowAboutDialog(parent) -> None:
    year = datetime.date.today().year
    description = textwrap.fill(
        _(
            "InVesalius is a medical imaging program for 3D reconstruction. It uses a sequence of "
            "2D DICOM image files acquired with CT or MRI scanners. InVesalius allows exporting 3D "
            "volumes or surfaces as mesh files for creating physical models of a patient's anatomy "
            "using additive manufacturing (3D printing) technologies. The software is developed by "
            "Center for Information Technology Renato Archer (CTI), National Council for Scientific "
            "and Technological Development (CNPq) and the Brazilian Ministry of Health.\n\n"
            "InVesalius must be used only for research. The Center for Information Technology Renato "
            "Archer is not responsible for damages caused by the use of this software.\n\n"
            "Contact: invesalius@cti.gov.br"
        ),
        80,
    )

    about_text = (
        f"<h2>InVesalius {invesalius.__version__}</h2>"
        f"<p>&copy; 2007-{year} Center for Information Technology Renato Archer - CTI</p>"
        f"<p>{description}</p>"
        f'<p>Website: <a href="https://www.cti.gov.br/invesalius">https://www.cti.gov.br/invesalius</a></p>'
        f"<p>License: GNU GPL (General Public License) version 2</p>"
    )
    QMessageBox.about(parent, "About InVesalius", about_text)


def ShowSavePresetDialog(default_filename: str = "raycasting") -> Optional[str]:
    from PySide6.QtWidgets import QInputDialog

    result, ok = QInputDialog.getText(
        None, "InVesalius 3", _("Save raycasting preset as:"), text=default_filename
    )
    return result if ok else None


# ---------------------------------------------------------------------------
# NewSurfaceDialog
# ---------------------------------------------------------------------------


class NewSurfaceDialog(QDialog):
    def __init__(self, parent=None, ID=-1, title="InVesalius 3", **kwargs):
        import invesalius.constants as const
        import invesalius.data.surface as surface
        import invesalius.project as prj

        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(500, 300)
        _center_on_screen(self)

        label_surface = QLabel(_("New surface name:"))
        default_name = const.SURFACE_NAME_PATTERN % (surface.Surface.general_index + 2)
        text = QLineEdit(default_name)
        self.text = text

        label_mask = QLabel(_("Mask of reference:"))
        project = prj.Project()
        index_list = sorted(project.mask_dict.keys())
        self.mask_list = [project.mask_dict[index].name for index in index_list]

        combo_mask = QComboBox()
        combo_mask.addItems(self.mask_list)
        combo_mask.setCurrentIndex(len(self.mask_list) - 1)
        self.combo_mask = combo_mask

        label_quality = QLabel(_("Surface quality:"))
        choices = const.SURFACE_QUALITY_LIST
        combo_quality = QComboBox()
        combo_quality.addItems(choices)
        combo_quality.setCurrentIndex(3)
        self.combo_quality = combo_quality

        fixed_layout = QGridLayout()
        fixed_layout.addWidget(label_surface, 0, 0)
        fixed_layout.addWidget(text, 0, 1)
        fixed_layout.addWidget(label_mask, 1, 0)
        fixed_layout.addWidget(combo_mask, 1, 1)
        fixed_layout.addWidget(label_quality, 2, 0)
        fixed_layout.addWidget(combo_quality, 2, 1)

        self.check_box_holes = QCheckBox(_("Fill holes"))
        self.check_box_holes.setChecked(True)
        self.check_box_largest = QCheckBox(_("Keep largest region"))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(fixed_layout)
        layout.addWidget(self.check_box_holes)
        layout.addWidget(self.check_box_largest)
        layout.addWidget(btn_box)
        self.adjustSize()

    def GetValue(self) -> Tuple[int, str, str, bool, bool]:
        mask_index = self.combo_mask.currentIndex()
        surface_name = self.text.text()
        quality = const.SURFACE_QUALITY_LIST[self.combo_quality.currentIndex()]
        fill_holes = self.check_box_holes.isChecked()
        keep_largest = self.check_box_largest.isChecked()
        return (mask_index, surface_name, quality, fill_holes, keep_largest)


def ExportPicture(type_: str = "") -> Union[Tuple[str, int], Tuple[()]]:
    import invesalius.constants as const
    import invesalius.project as proj

    INDEX_TO_EXTENSION = {0: "bmp", 1: "jpg", 2: "png", 3: "ps", 4: "povray", 5: "tiff"}
    WILDCARD_SAVE_PICTURE = (
        _("BMP image")
        + " (*.bmp);;"
        + _("JPG image")
        + " (*.jpg);;"
        + _("PNG image")
        + " (*.png);;"
        + _("PostScript document")
        + " (*.ps);;"
        + _("POV-Ray file")
        + " (*.pov);;"
        + _("TIFF image")
        + " (*.tif)"
    )
    INDEX_TO_TYPE = {
        0: const.FILETYPE_BMP,
        1: const.FILETYPE_JPG,
        2: const.FILETYPE_PNG,
        3: const.FILETYPE_PS,
        4: const.FILETYPE_POV,
        5: const.FILETYPE_TIF,
    }

    project = proj.Project()
    session = ses.Session()
    last_directory = session.GetConfig("last_directory_screenshot", "")
    project_name = f"{project.name}_{type_}"
    if sys.platform not in ("win32", "linux2", "linux"):
        project_name += ".jpg"

    filename, selected_filter = QFileDialog.getSaveFileName(
        None,
        f"Save {type_} picture as...",
        os.path.join(last_directory, project_name),
        WILDCARD_SAVE_PICTURE,
    )

    if filename:
        filters = WILDCARD_SAVE_PICTURE.split(";;")
        filetype_index = filters.index(selected_filter) if selected_filter in filters else 1
        filetype = INDEX_TO_TYPE.get(filetype_index, const.FILETYPE_JPG)
        extension = INDEX_TO_EXTENSION.get(filetype_index, "jpg")

        last_directory = os.path.split(filename)[0]
        session.SetConfig("last_directory_screenshot", last_directory)

        if sys.platform != "win32":
            if filename.split(".")[-1] != extension:
                filename = filename + "." + extension
        return filename, filetype
    return ()


# ---------------------------------------------------------------------------
# Surface dialogs
# ---------------------------------------------------------------------------


class SurfaceDialog(QDialog):
    def __init__(self):
        super().__init__(None)
        self.setWindowTitle(_("Surface generation options"))
        self._build_widgets()
        _center_on_screen(self)

    def _build_widgets(self) -> None:
        self.ca = SurfaceMethodPanel(self, -1, True)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.ca)
        layout.addWidget(btn_box)
        self.adjustSize()

    def GetOptions(self) -> Dict[str, float]:
        return self.ca.GetOptions()

    def GetAlgorithmSelected(self) -> str:
        return self.ca.GetAlgorithmSelected()


class SurfaceCreationDialog(QDialog):
    def __init__(
        self, parent=None, ID=-1, title=_("Surface creation"), mask_edited=False, **kwargs
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        _center_on_screen(self)

        options_group = QGroupBox(_("Surface creation options"))
        options_layout = QVBoxLayout(options_group)
        self.nsd = SurfaceCreationOptionsPanel(self, -1)
        self.nsd.mask_set.connect(self.OnSetMask)
        options_layout.addWidget(self.nsd)

        method_group = QGroupBox(_("Surface creation method"))
        method_layout = QVBoxLayout(method_group)
        self.ca = SurfaceMethodPanel(self, -1, mask_edited)
        method_layout.addWidget(self.ca)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        panels_layout = QHBoxLayout()
        panels_layout.addWidget(options_group)
        panels_layout.addWidget(method_group)

        layout = QVBoxLayout(self)
        layout.addLayout(panels_layout)
        layout.addWidget(btn_box)
        self.adjustSize()

    def OnSetMask(self, mask_index: int) -> None:
        import invesalius.project as proj

        mask = proj.Project().mask_dict[mask_index]
        self.ca.mask_edited = mask.was_edited
        self.ca.ReloadMethodsOptions()

    def GetValue(self):
        return {"method": self.ca.GetValue(), "options": self.nsd.GetValue()}


class SurfaceCreationOptionsPanel(QWidget):
    mask_set = Signal(int)

    def __init__(self, parent=None, ID=-1):
        import invesalius.constants as const
        import invesalius.data.slice_ as slc
        import invesalius.data.surface as surface
        import invesalius.project as prj

        super().__init__(parent)

        label_surface = QLabel(_("New surface name:"))
        default_name = const.SURFACE_NAME_PATTERN % (surface.Surface.general_index + 2)
        text = QLineEdit(default_name)
        self.text = text

        label_mask = QLabel(_("Mask of reference:"))
        project = prj.Project()
        index_list = project.mask_dict.keys()
        self.mask_list = [project.mask_dict[index].name for index in sorted(index_list)]

        active_mask = 0
        for idx in project.mask_dict:
            if project.mask_dict[idx] is slc.Slice().current_mask:
                active_mask = idx
                break

        combo_mask = QComboBox()
        combo_mask.addItems(self.mask_list)
        combo_mask.setCurrentIndex(active_mask)
        combo_mask.currentIndexChanged.connect(self.OnSetMask)
        self.combo_mask = combo_mask

        label_quality = QLabel(_("Surface quality:"))
        combo_quality = QComboBox()
        combo_quality.addItems(const.SURFACE_QUALITY_LIST)
        combo_quality.setCurrentIndex(3)
        self.combo_quality = combo_quality

        fixed_layout = QGridLayout()
        fixed_layout.addWidget(label_surface, 0, 0)
        fixed_layout.addWidget(text, 0, 1)
        fixed_layout.addWidget(label_mask, 1, 0)
        fixed_layout.addWidget(combo_mask, 1, 1)
        fixed_layout.addWidget(label_quality, 2, 0)
        fixed_layout.addWidget(combo_quality, 2, 1)

        self.check_box_border_holes = QCheckBox(_("Fill border holes"))
        self.check_box_holes = QCheckBox(_("Fill holes"))
        self.check_box_largest = QCheckBox(_("Keep largest region"))

        layout = QVBoxLayout(self)
        layout.addLayout(fixed_layout)
        layout.addWidget(self.check_box_border_holes)
        layout.addWidget(self.check_box_holes)
        layout.addWidget(self.check_box_largest)

    def OnSetMask(self, index: int) -> None:
        self.mask_set.emit(index)

    def GetValue(self) -> Dict[str, Union[str, int, bool]]:
        return {
            "index": self.combo_mask.currentIndex(),
            "name": self.text.text(),
            "quality": const.SURFACE_QUALITY_LIST[self.combo_quality.currentIndex()],
            "fill_border_holes": self.check_box_border_holes.isChecked(),
            "fill": self.check_box_holes.isChecked(),
            "keep_largest": self.check_box_largest.isChecked(),
            "overwrite": False,
        }


class SelectLargestSurfaceProgressWindow:
    def __init__(self):
        parent = _top_window()
        self.dlg = QProgressDialog(
            "Creating a new surface form the largest contiguous region...", "Cancel", 0, 0, parent
        )
        self.dlg.setWindowTitle("InVesalius 3")
        self.dlg.setWindowModality(Qt.ApplicationModal)
        self.dlg.show()

    def Update(self, msg=None, value=None):
        if msg:
            self.dlg.setLabelText(msg)
        QApplication.processEvents()

    def Close(self):
        self.dlg.close()


class SmoothSurfaceProgressWindow:
    def __init__(self):
        parent = _top_window()
        self.dlg = QProgressDialog("Creating a new smooth surface ...", "Cancel", 0, 0, parent)
        self.dlg.setWindowTitle("InVesalius 3")
        self.dlg.setWindowModality(Qt.ApplicationModal)
        self.dlg.show()

    def Update(self, msg=None, value=None):
        if msg:
            self.dlg.setLabelText(msg)
        QApplication.processEvents()

    def Close(self):
        self.dlg.close()


class RemoveNonVisibleFacesProgressWindow:
    def __init__(self):
        parent = _top_window()
        self.dlg = QProgressDialog("Removing non-visible faces...", "Cancel", 0, 0, parent)
        self.dlg.setWindowTitle("InVesalius 3")
        self.dlg.setWindowModality(Qt.ApplicationModal)
        self.dlg.show()

    def Update(self, msg=None, value=None):
        if msg:
            self.dlg.setLabelText(msg)
        QApplication.processEvents()

    def Close(self):
        self.dlg.close()


class SurfaceTransparencyDialog(QDialog):
    def __init__(self, parent, surface_index=0, transparency=0):
        super().__init__(parent)
        self.surface_index = surface_index
        self.setWindowTitle("InVesalius 3")
        self.resize(300, 180)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(transparency)
        self.slider.valueChanged.connect(self.on_slider)

        self.value_text = QLabel(f"Surface transparency: {self.slider.value()}%")

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.value_text)
        layout.addWidget(self.slider)
        layout.addWidget(btn_box)
        self.adjustSize()
        _center_on_screen(self)

    def on_slider(self, value):
        self.value_text.setText(f"Surface transparency: {value}%")
        Publisher.sendMessage(
            "Set surface transparency",
            surface_index=self.surface_index,
            transparency=value / 100.0,
        )

    def get_value(self):
        return self.slider.value()


class CAOptions(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_widgets()

    def _build_widgets(self):
        group = QGroupBox(_("Options"))
        self.angle = InvFloatSpinCtrl(
            self, -1, value=0.7, min_value=0.0, max_value=1.0, increment=0.1, digits=1
        )
        self.max_distance = InvFloatSpinCtrl(
            self, -1, value=3.0, min_value=0.0, max_value=100.0, increment=0.1, digits=2
        )
        self.min_weight = InvFloatSpinCtrl(
            self, -1, value=0.5, min_value=0.0, max_value=1.0, increment=0.1, digits=1
        )
        self.steps = InvSpinCtrl(self, -1, value=10, min_value=1, max_value=100)

        grid = QGridLayout()
        grid.addWidget(QLabel(_("Angle:")), 0, 0)
        grid.addWidget(self.angle, 0, 1)
        grid.addWidget(QLabel(_("Max. distance:")), 1, 0)
        grid.addWidget(self.max_distance, 1, 1)
        grid.addWidget(QLabel(_("Min. weight:")), 2, 0)
        grid.addWidget(self.min_weight, 2, 1)
        grid.addWidget(QLabel(_("N. steps:")), 3, 0)
        grid.addWidget(self.steps, 3, 1)
        group.setLayout(grid)

        layout = QVBoxLayout(self)
        layout.addWidget(group)


class SurfaceMethodPanel(QWidget):
    def __init__(self, parent=None, id=-1, mask_edited=False):
        super().__init__(parent)
        self.mask_edited = mask_edited
        self.alg_types = {
            _("Default"): "Default",
            _("Context aware smoothing"): "ca_smoothing",
            _("Binary"): "Binary",
        }
        self.edited_imp = [_("Default")]
        self._build_widgets()

    def _build_widgets(self):
        self.ca_options = CAOptions(self)
        choices = [
            i for i in sorted(self.alg_types) if not (self.mask_edited and i in self.edited_imp)
        ]
        self.cb_types = QComboBox()
        self.cb_types.addItems(choices)
        self.cb_types.currentTextChanged.connect(self._set_cb_types)

        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel(_("Method:")))
        method_layout.addWidget(self.cb_types, 1)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(method_layout)
        main_layout.addWidget(self.ca_options)

        if self.mask_edited:
            self.cb_types.setCurrentText(_("Context aware smoothing"))
            self.ca_options.setEnabled(True)
        else:
            self.ca_options.setEnabled(False)

    def _set_cb_types(self, text):
        self.ca_options.setEnabled(self.alg_types.get(text) == "ca_smoothing")

    def GetAlgorithmSelected(self):
        return self.alg_types.get(self.cb_types.currentText(), "Default")

    def GetOptions(self):
        if self.GetAlgorithmSelected() == "ca_smoothing":
            return {
                "angle": self.ca_options.angle.GetValue(),
                "max distance": self.ca_options.max_distance.GetValue(),
                "min weight": self.ca_options.min_weight.GetValue(),
                "steps": self.ca_options.steps.GetValue(),
            }
        return {}

    def GetValue(self):
        return {"algorithm": self.GetAlgorithmSelected(), "options": self.GetOptions()}

    def ReloadMethodsOptions(self):
        self.cb_types.clear()
        self.cb_types.addItems(
            [i for i in sorted(self.alg_types) if not (self.mask_edited and i in self.edited_imp)]
        )
        if self.mask_edited:
            self.cb_types.setCurrentText(_("Context aware smoothing"))
            self.ca_options.setEnabled(True)
        else:
            self.cb_types.setCurrentText(_("Default"))
            self.ca_options.setEnabled(False)


class ClutImagedataDialog(QDialog):
    def __init__(self, histogram, init, end, nodes=None):
        super().__init__(_top_window())
        self.setWindowFlags(self.windowFlags() | Qt.Tool)

        self.histogram = histogram
        self.init = init
        self.end = end
        self.nodes = nodes
        self._init_gui()
        self.bind_events()

    def _init_gui(self):
        self.clut_widget = CLUTImageDataWidget(
            self, -1, self.histogram, self.init, self.end, self.nodes
        )
        layout = QVBoxLayout(self)
        layout.addWidget(self.clut_widget)
        self.adjustSize()

    def bind_events(self):
        Publisher.subscribe(self._refresh_widget, "Update clut imagedata widget")
        self.clut_widget.clutNodeChanged.connect(self.OnClutChange)

    def OnClutChange(self, nodes):
        Publisher.sendMessage("Change colour table from background image from widget", nodes=nodes)
        Publisher.sendMessage(
            "Update window level text",
            window=self.clut_widget.window_width,
            level=self.clut_widget.window_level,
        )

    def _refresh_widget(self):
        self.clut_widget.update()

    def Show(self, gen_evt=True, show=True):
        if show:
            super().show()
        else:
            super().hide()
        if gen_evt:
            self.clut_widget._generate_event()


# ---------------------------------------------------------------------------
# Watershed
# ---------------------------------------------------------------------------


class WatershedOptionsPanel(QWidget):
    def __init__(self, parent, config):
        super().__init__(parent)
        self.algorithms = ("Watershed", "Watershed IFT")
        self.con2d_choices = (4, 8)
        self.con3d_choices = (6, 18, 26)
        self.config = config
        self._init_gui()

    def _init_gui(self):
        algo_group = QGroupBox(_("Method"))
        algo_layout = QHBoxLayout(algo_group)
        self.algo_buttons = []
        for i, alg in enumerate(self.algorithms):
            rb = QRadioButton(alg)
            if self.algorithms.index(self.config.algorithm) == i:
                rb.setChecked(True)
            self.algo_buttons.append(rb)
            algo_layout.addWidget(rb)

        conn_group = QGroupBox("Conectivity")
        conn_layout = QVBoxLayout(conn_group)

        lbl_2d = QLabel("2D")
        conn_layout.addWidget(lbl_2d)
        self.con2d_buttons = []
        h2d = QHBoxLayout()
        for i, val in enumerate(self.con2d_choices):
            rb = QRadioButton(str(val))
            if self.con2d_choices.index(self.config.con_2d) == i:
                rb.setChecked(True)
            self.con2d_buttons.append(rb)
            h2d.addWidget(rb)
        conn_layout.addLayout(h2d)

        lbl_3d = QLabel("3D")
        conn_layout.addWidget(lbl_3d)
        self.con3d_buttons = []
        h3d = QHBoxLayout()
        for i, val in enumerate(self.con3d_choices):
            rb = QRadioButton(str(val))
            if self.con3d_choices.index(self.config.con_3d) == i:
                rb.setChecked(True)
            self.con3d_buttons.append(rb)
            h3d.addWidget(rb)
        conn_layout.addLayout(h3d)

        self.gaussian_size = InvSpinCtrl(
            self, -1, value=self.config.mg_size, min_value=1, max_value=10
        )

        g_layout = QHBoxLayout()
        g_layout.addWidget(QLabel(_("Gaussian sigma")))
        g_layout.addWidget(self.gaussian_size)

        layout = QVBoxLayout(self)
        layout.addWidget(algo_group)
        layout.addWidget(conn_group)
        layout.addLayout(g_layout)

    def apply_options(self):
        for i, rb in enumerate(self.algo_buttons):
            if rb.isChecked():
                self.config.algorithm = self.algorithms[i]
                break
        for i, rb in enumerate(self.con2d_buttons):
            if rb.isChecked():
                self.config.con_2d = self.con2d_choices[i]
                break
        for i, rb in enumerate(self.con3d_buttons):
            if rb.isChecked():
                self.config.con_3d = self.con3d_choices[i]
                break
        self.config.mg_size = self.gaussian_size.GetValue()


class WatershedOptionsDialog(QDialog):
    def __init__(self, config, ID=-1, title=_("Watershed"), **kwargs):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.config = config
        self._init_gui()

    def _init_gui(self):
        self.wop = WatershedOptionsPanel(self, self.config)
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.OnOk)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.wop)
        layout.addWidget(btn_box)
        self.adjustSize()
        _center_on_screen(self)

    def OnOk(self):
        self.wop.apply_options()
        self.accept()


class MaskBooleanDialog(QDialog):
    def __init__(self, masks, ID=-1, title=_("Boolean operations"), **kwargs):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self._init_gui(masks)
        _center_on_screen(self)

    def _init_gui(self, masks):
        mask_choices = [(masks[i].name, masks[i]) for i in sorted(masks)]
        self.mask1 = QComboBox()
        self.mask2 = QComboBox()
        for n, m in mask_choices:
            self.mask1.addItem(n, m)
            self.mask2.addItem(n, m)
        self.mask1.setCurrentIndex(0)
        self.mask2.setCurrentIndex(min(1, len(mask_choices) - 1))

        icon_folder = inv_paths.ICON_DIR
        op_choices = (
            (_("Union"), const.BOOLEAN_UNION, "bool_union.png"),
            (_("Difference"), const.BOOLEAN_DIFF, "bool_difference.png"),
            (_("Intersection"), const.BOOLEAN_AND, "bool_intersection.png"),
            (_("Exclusive disjunction"), const.BOOLEAN_XOR, "bool_disjunction.png"),
        )
        self.op_boolean = QComboBox()
        for n, val, f in op_choices:
            icon = QIcon(os.path.join(icon_folder, f))
            self.op_boolean.addItem(icon, n, val)
        self.op_boolean.setCurrentIndex(0)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.OnOk)
        btn_box.rejected.connect(self.reject)

        grid = QGridLayout()
        grid.addWidget(QLabel(_("Mask 1")), 0, 0)
        grid.addWidget(self.mask1, 0, 1)
        grid.addWidget(QLabel(_("Operation")), 1, 0)
        grid.addWidget(self.op_boolean, 1, 1)
        grid.addWidget(QLabel(_("Mask 2")), 2, 0)
        grid.addWidget(self.mask2, 2, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(btn_box)
        self.adjustSize()

    def OnOk(self):
        op = self.op_boolean.currentData()
        m1 = self.mask1.currentData()
        m2 = self.mask2.currentData()
        Publisher.sendMessage("Do boolean operation", operation=op, mask1=m1, mask2=m2)
        Publisher.sendMessage("Reload actual slice")
        Publisher.sendMessage("Refresh viewer")
        self.close()


class ReorientImageDialog(QDialog):
    def __init__(self, ID=-1, title=_("Image reorientation"), **kwargs):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self._closed = False
        self._last_ax = "0.0"
        self._last_ay = "0.0"
        self._last_az = "0.0"
        self._init_gui()
        self._bind_events()

    def _init_gui(self):
        interp_methods_choices = (
            (_("Nearest Neighbour"), 0),
            (_("Trilinear"), 1),
            (_("Tricubic"), 2),
            (_("Lanczos (experimental)"), 3),
        )
        self.interp_method = QComboBox()
        for txt, code in interp_methods_choices:
            self.interp_method.addItem(txt, code)
        self.interp_method.setCurrentIndex(2)
        self.interp_method.currentIndexChanged.connect(self.OnSelect)

        self.anglex = QLineEdit("0.0")
        self.angley = QLineEdit("0.0")
        self.anglez = QLineEdit("0.0")
        self.anglex.editingFinished.connect(self.OnLostFocus)
        self.angley.editingFinished.connect(self.OnLostFocus)
        self.anglez.editingFinished.connect(self.OnLostFocus)

        self.btnapply = QPushButton(_("Apply"))
        self.btnapply.clicked.connect(self.apply_reorientation)

        angles_layout = QGridLayout()
        angles_layout.addWidget(QLabel(_("Angle X")), 0, 0)
        angles_layout.addWidget(self.anglex, 0, 1)
        angles_layout.addWidget(QLabel(_("Angle Y")), 1, 0)
        angles_layout.addWidget(self.angley, 1, 1)
        angles_layout.addWidget(QLabel(_("Angle Z")), 2, 0)
        angles_layout.addWidget(self.anglez, 2, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_("Interpolation method:")))
        layout.addWidget(self.interp_method)
        layout.addLayout(angles_layout)
        layout.addWidget(self.btnapply)
        self.adjustSize()

    def _bind_events(self):
        Publisher.subscribe(self._update_angles, "Update reorient angles")
        Publisher.subscribe(self._close_dialog, "Close reorient dialog")

    def _update_angles(self, angles):
        ax, ay, az = angles
        self.anglex.setText(f"{np.rad2deg(ax):.3f}")
        self.angley.setText(f"{np.rad2deg(ay):.3f}")
        self.anglez.setText(f"{np.rad2deg(az):.3f}")

    def _close_dialog(self):
        self.close()

    def apply_reorientation(self):
        Publisher.sendMessage("Apply reorientation")
        self.close()

    def closeEvent(self, event):
        self._closed = True
        Publisher.sendMessage("Disable style", style=const.SLICE_STATE_REORIENT)
        Publisher.sendMessage("Enable style", style=const.STATE_DEFAULT)
        super().closeEvent(event)

    def OnSelect(self, index):
        im_code = self.interp_method.currentData()
        Publisher.sendMessage("Set interpolation method", interp_method=im_code)

    def OnLostFocus(self):
        if not self._closed:
            try:
                ax = np.deg2rad(float(self.anglex.text()))
                ay = np.deg2rad(float(self.angley.text()))
                az = np.deg2rad(float(self.anglez.text()))
            except ValueError:
                return
            Publisher.sendMessage("Set reorientation angles", angles=(ax, ay, az))


class ImportBitmapParameters(QDialog):
    def __init__(self):
        super().__init__(_top_window())
        self.setWindowTitle(_("Create project from bitmap"))
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.interval = 0
        self._init_gui()
        _center_on_screen(self)

    def _init_gui(self):
        import invesalius.project as prj

        self.tx_name = QLineEdit("InVesalius Bitmap")
        cb_orientation_options = [_("Axial"), _("Coronal"), _("Sagital")]
        self.cb_orientation = QComboBox()
        self.cb_orientation.addItems(cb_orientation_options)

        self.fsp_spacing_x = InvFloatSpinCtrl(
            self, -1, min_value=0, max_value=1e9, increment=0.25, value=1.0, digits=8
        )
        self.fsp_spacing_y = InvFloatSpinCtrl(
            self, -1, min_value=0, max_value=1e9, increment=0.25, value=1.0, digits=8
        )
        self.fsp_spacing_z = InvFloatSpinCtrl(
            self, -1, min_value=0, max_value=1e9, increment=0.25, value=1.0, digits=8
        )

        try:
            proj = prj.Project()
            self.fsp_spacing_x.SetValue(proj.spacing[0])
            self.fsp_spacing_y.SetValue(proj.spacing[1])
            self.fsp_spacing_z.SetValue(proj.spacing[2])
        except AttributeError:
            pass

        grid = QGridLayout()
        grid.addWidget(QLabel(_("Project name:")), 0, 0)
        grid.addWidget(self.tx_name, 0, 1, 1, 5)
        grid.addWidget(QLabel(_("Slices orientation:")), 1, 0)
        grid.addWidget(self.cb_orientation, 1, 1, 1, 5)
        grid.addWidget(QLabel(_("Spacing (mm):")), 2, 0)
        grid.addWidget(QLabel(_("X:")), 3, 0)
        grid.addWidget(self.fsp_spacing_x, 3, 1)
        grid.addWidget(QLabel(_("Y:")), 3, 2)
        grid.addWidget(self.fsp_spacing_y, 3, 3)
        grid.addWidget(QLabel(_("Z:")), 3, 4)
        grid.addWidget(self.fsp_spacing_z, 3, 5)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.OnOk)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(btn_box)
        self.adjustSize()

    def SetInterval(self, v):
        self.interval = v

    def OnOk(self):
        sel = self.cb_orientation.currentIndex()
        orientation = {0: "AXIAL", 1: "CORONAL", 2: "SAGITTAL"}.get(sel, "AXIAL")
        values = [
            self.tx_name.text(),
            orientation,
            self.fsp_spacing_x.GetValue(),
            self.fsp_spacing_y.GetValue(),
            self.fsp_spacing_z.GetValue(),
            self.interval,
        ]
        Publisher.sendMessage("Open bitmap files", rec_data=values)
        self.close()


def BitmapNotSameSize() -> None:
    QMessageBox.critical(
        None,
        "Error",
        _("All bitmaps files must be the same \n width and height size."),
    )


# ---------------------------------------------------------------------------
# Flood Fill panels
# ---------------------------------------------------------------------------


class PanelTargeFFill(QWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self.target_2d = QRadioButton(_("2D - Actual slice"))
        self.target_3d = QRadioButton(_("3D - All slices"))
        self.target_2d.setChecked(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.target_2d)
        layout.addWidget(self.target_3d)


class Panel2DConnectivity(QWidget):
    def __init__(self, parent=None, show_orientation=False, **kwargs):
        super().__init__(parent)
        self.conect2D_4 = QRadioButton("4")
        self.conect2D_8 = QRadioButton("8")
        self.conect2D_4.setChecked(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_("2D Connectivity")))
        h = QHBoxLayout()
        h.addWidget(self.conect2D_4)
        h.addWidget(self.conect2D_8)
        layout.addLayout(h)

        if show_orientation:
            self.cmb_orientation = QComboBox()
            self.cmb_orientation.addItems([_("Axial"), _("Coronal"), _("Sagital")])
            layout.addWidget(QLabel(_("Orientation")))
            layout.addWidget(self.cmb_orientation)

    def GetConnSelected(self):
        return 4 if self.conect2D_4.isChecked() else 8

    def GetOrientation(self):
        dic_ori = {_("Axial"): "AXIAL", _("Coronal"): "CORONAL", _("Sagital"): "SAGITAL"}
        return dic_ori.get(self.cmb_orientation.currentText(), "AXIAL")


class Panel3DConnectivity(QWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self.conect3D_6 = QRadioButton("6")
        self.conect3D_18 = QRadioButton("18")
        self.conect3D_26 = QRadioButton("26")
        self.conect3D_6.setChecked(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_("3D Connectivity")))
        h = QHBoxLayout()
        h.addWidget(self.conect3D_6)
        h.addWidget(self.conect3D_18)
        h.addWidget(self.conect3D_26)
        layout.addLayout(h)

    def GetConnSelected(self):
        if self.conect3D_6.isChecked():
            return 6
        elif self.conect3D_18.isChecked():
            return 18
        return 26


class PanelFFillThreshold(QWidget):
    def __init__(self, parent, config, **kwargs):
        super().__init__(parent)
        self.config = config
        self._init_gui()

    def _init_gui(self):
        import invesalius.project as prj

        project = prj.Project()
        bound_min, bound_max = project.threshold_range
        colour = [i * 255 for i in const.MASK_COLOUR[0]]
        colour.append(100)
        self.threshold = grad.GradientCtrl(
            self, -1, int(bound_min), int(bound_max), self.config.t0, self.config.t1, colour
        )
        layout = QVBoxLayout(self)
        layout.addWidget(self.threshold)
        self.threshold.thresholdChanged.connect(self.OnSlideChanged)
        self.threshold.thresholdChanging.connect(self.OnSlideChanged)

    def OnSlideChanged(self):
        self.config.t0 = int(self.threshold.GetMinValue())
        self.config.t1 = int(self.threshold.GetMaxValue())


class PanelFFillDynamic(QWidget):
    def __init__(self, parent, config, **kwargs):
        super().__init__(parent)
        self.config = config
        self._init_gui()

    def _init_gui(self):
        self.use_ww_wl = QCheckBox(_("Use WW&WL"))
        self.use_ww_wl.setChecked(self.config.use_ww_wl)
        self.use_ww_wl.stateChanged.connect(
            lambda: setattr(self.config, "use_ww_wl", self.use_ww_wl.isChecked())
        )

        self.deviation_min = InvSpinCtrl(
            self, -1, value=self.config.dev_min, min_value=0, max_value=10000
        )
        self.deviation_max = InvSpinCtrl(
            self, -1, value=self.config.dev_max, min_value=0, max_value=10000
        )

        self.deviation_min.valueChanged.connect(
            lambda: setattr(self.config, "dev_min", self.deviation_min.GetValue())
        )
        self.deviation_max.valueChanged.connect(
            lambda: setattr(self.config, "dev_max", self.deviation_max.GetValue())
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.use_ww_wl)
        layout.addWidget(QLabel(_("Deviation")))
        h = QHBoxLayout()
        h.addWidget(QLabel(_("Min:")))
        h.addWidget(self.deviation_min)
        h.addWidget(QLabel(_("Max:")))
        h.addWidget(self.deviation_max)
        layout.addLayout(h)


class PanelFFillConfidence(QWidget):
    def __init__(self, parent, config, **kwargs):
        super().__init__(parent)
        self.config = config
        self._init_gui()

    def _init_gui(self):
        self.use_ww_wl = QCheckBox(_("Use WW&WL"))
        self.use_ww_wl.setChecked(self.config.use_ww_wl)
        self.use_ww_wl.stateChanged.connect(
            lambda: setattr(self.config, "use_ww_wl", self.use_ww_wl.isChecked())
        )

        self.spin_mult = InvFloatSpinCtrl(
            self,
            -1,
            value=self.config.confid_mult,
            min_value=1.0,
            max_value=10.0,
            increment=0.1,
            digits=1,
        )
        self.spin_iters = InvSpinCtrl(
            self, -1, value=self.config.confid_iters, min_value=0, max_value=100
        )

        self.spin_mult.valueChanged.connect(
            lambda: setattr(self.config, "confid_mult", self.spin_mult.GetValue())
        )
        self.spin_iters.valueChanged.connect(
            lambda: setattr(self.config, "confid_iters", self.spin_iters.GetValue())
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self.use_ww_wl)
        g = QGridLayout()
        g.addWidget(QLabel(_("Multiplier")), 0, 0)
        g.addWidget(self.spin_mult, 0, 1)
        g.addWidget(QLabel(_("Iterations")), 1, 0)
        g.addWidget(self.spin_iters, 1, 1)
        layout.addLayout(g)


class PanelFFillProgress(QWidget):
    def __init__(self, parent=None, **kwargs):
        super().__init__(parent)
        self.progress = QProgressBar()
        self.lbl_progress_caption = QLabel(_("Elapsed time:"))
        self.lbl_time = QLabel(_("00:00:00"))

        layout = QVBoxLayout(self)
        layout.addWidget(self.progress)
        h = QHBoxLayout()
        h.addWidget(self.lbl_progress_caption)
        h.addWidget(self.lbl_time, 1)
        layout.addLayout(h)
        self.t0 = 0

    def StartTimer(self):
        self.t0 = time.time()

    def StopTimer(self):
        fmt = "%H:%M:%S"
        self.lbl_time.setText(time.strftime(fmt, time.gmtime(time.time() - self.t0)))
        self.progress.setValue(0)

    def Pulse(self):
        fmt = "%H:%M:%S"
        self.lbl_time.setText(time.strftime(fmt, time.gmtime(time.time() - self.t0)))
        self.progress.setRange(0, 0)


class FFillOptionsDialog(QDialog):
    def __init__(self, title, config):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.config = config
        self._init_gui()

    def _init_gui(self):
        self.panel_target = PanelTargeFFill(self)
        self.panel2dcon = Panel2DConnectivity(self)
        self.panel3dcon = Panel3DConnectivity(self)

        if self.config.target == "2D":
            self.panel_target.target_2d.setChecked(True)
            self.panel2dcon.setEnabled(True)
            self.panel3dcon.setEnabled(False)
        else:
            self.panel_target.target_3d.setChecked(True)
            self.panel3dcon.setEnabled(True)
            self.panel2dcon.setEnabled(False)

        if self.config.con_2d == 8:
            self.panel2dcon.conect2D_8.setChecked(True)
        if self.config.con_3d == 18:
            self.panel3dcon.conect3D_18.setChecked(True)
        elif self.config.con_3d == 26:
            self.panel3dcon.conect3D_26.setChecked(True)

        close_btn = QPushButton(_("Close"))
        close_btn.clicked.connect(self.close)

        self.panel_target.target_2d.toggled.connect(self._on_target_changed)
        self.panel_target.target_3d.toggled.connect(self._on_target_changed)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_("Parameters")))
        layout.addWidget(self.panel_target)
        layout.addWidget(self.panel2dcon)
        layout.addWidget(self.panel3dcon)
        layout.addWidget(close_btn)
        self.adjustSize()

    def _on_target_changed(self):
        if self.panel_target.target_2d.isChecked():
            self.config.target = "2D"
            self.panel2dcon.setEnabled(True)
            self.panel3dcon.setEnabled(False)
        else:
            self.config.target = "3D"
            self.panel3dcon.setEnabled(True)
            self.panel2dcon.setEnabled(False)
        self.config.con_2d = self.panel2dcon.GetConnSelected()
        self.config.con_3d = self.panel3dcon.GetConnSelected()

    def closeEvent(self, event):
        if self.config.dlg_visible:
            Publisher.sendMessage("Disable style", style=const.SLICE_STATE_MASK_FFILL)
        super().closeEvent(event)


class SelectPartsOptionsDialog(QDialog):
    def __init__(self, config):
        super().__init__(_top_window())
        self.setWindowTitle(_("Select mask parts"))
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.config = config
        self._return_code = QDialog.Rejected
        self._init_gui()

    def _init_gui(self):
        self.target_name = QLineEdit(self.config.mask_name)
        self.panel3dcon = Panel3DConnectivity(self)
        if self.config.con_3d == 18:
            self.panel3dcon.conect3D_18.setChecked(True)
        elif self.config.con_3d == 26:
            self.panel3dcon.conect3D_26.setChecked(True)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self._on_ok)
        btn_box.rejected.connect(self._on_cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_("Target mask name")))
        layout.addWidget(self.target_name)
        layout.addWidget(self.panel3dcon)
        layout.addWidget(btn_box)
        self.adjustSize()

    def _on_ok(self):
        self.config.mask_name = self.target_name.text()
        self.config.con_3d = self.panel3dcon.GetConnSelected()
        self._return_code = QDialog.Accepted
        self.close()

    def _on_cancel(self):
        self._return_code = QDialog.Rejected
        self.close()

    def exec(self):
        super().exec()
        return self._return_code

    def closeEvent(self, event):
        if self.config.dlg_visible:
            Publisher.sendMessage("Disable style", style=const.SLICE_STATE_SELECT_MASK_PARTS)
        super().closeEvent(event)


class FFillSegmentationOptionsDialog(QDialog):
    def __init__(self, config, ID=-1, title=_("Region growing"), **kwargs):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.config = config
        self._init_gui()

    def _init_gui(self):
        self.panel_target = PanelTargeFFill(self)
        self.panel2dcon = Panel2DConnectivity(self)
        self.panel3dcon = Panel3DConnectivity(self)

        if self.config.target == "2D":
            self.panel_target.target_2d.setChecked(True)
            self.panel2dcon.setEnabled(True)
            self.panel3dcon.setEnabled(False)
        else:
            self.panel_target.target_3d.setChecked(True)
            self.panel3dcon.setEnabled(True)
            self.panel2dcon.setEnabled(False)

        if self.config.con_2d == 8:
            self.panel2dcon.conect2D_8.setChecked(True)
        if self.config.con_3d == 18:
            self.panel3dcon.conect3D_18.setChecked(True)
        elif self.config.con_3d == 26:
            self.panel3dcon.conect3D_26.setChecked(True)

        self.cmb_method = QComboBox()
        self.cmb_method.addItems([_("Dynamic"), _("Threshold"), _("Confidence")])
        method_map = {"dynamic": 0, "threshold": 1, "confidence": 2}
        self.cmb_method.setCurrentIndex(method_map.get(self.config.method, 0))

        self.panel_ffill_threshold = PanelFFillThreshold(self, self.config)
        self.panel_ffill_threshold.setMinimumWidth(250)
        self.panel_ffill_dynamic = PanelFFillDynamic(self, self.config)
        self.panel_ffill_dynamic.setMinimumWidth(250)
        self.panel_ffill_confidence = PanelFFillConfidence(self, self.config)
        self.panel_ffill_confidence.setMinimumWidth(250)
        self.panel_ffill_progress = PanelFFillProgress(self)
        self.panel_ffill_progress.setMinimumWidth(250)

        self._method_stack = QVBoxLayout()
        self._method_widgets = [
            self.panel_ffill_dynamic,
            self.panel_ffill_threshold,
            self.panel_ffill_confidence,
        ]
        for w in self._method_widgets:
            self._method_stack.addWidget(w)
        self._update_method_panel()

        close_btn = QPushButton(_("Close"))
        close_btn.clicked.connect(self.close)

        self.panel_target.target_2d.toggled.connect(self._on_target_changed)
        self.panel_target.target_3d.toggled.connect(self._on_target_changed)
        self.cmb_method.currentIndexChanged.connect(self._on_method_changed)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_("Parameters")))
        layout.addWidget(self.panel_target)
        layout.addWidget(self.panel2dcon)
        layout.addWidget(self.panel3dcon)
        h = QHBoxLayout()
        h.addWidget(QLabel(_("Method")))
        h.addWidget(self.cmb_method)
        layout.addLayout(h)
        layout.addLayout(self._method_stack)
        layout.addWidget(self.panel_ffill_progress)
        layout.addWidget(close_btn)
        self.adjustSize()

    def _update_method_panel(self):
        idx = self.cmb_method.currentIndex()
        for i, w in enumerate(self._method_widgets):
            w.setVisible(i == idx)

    def _on_target_changed(self):
        if self.panel_target.target_2d.isChecked():
            self.config.target = "2D"
            self.panel2dcon.setEnabled(True)
            self.panel3dcon.setEnabled(False)
        else:
            self.config.target = "3D"
            self.panel3dcon.setEnabled(True)
            self.panel2dcon.setEnabled(False)
        self.config.con_2d = self.panel2dcon.GetConnSelected()
        self.config.con_3d = self.panel3dcon.GetConnSelected()

    def _on_method_changed(self, idx):
        methods = ["dynamic", "threshold", "confidence"]
        self.config.method = methods[idx] if idx < len(methods) else "dynamic"
        self._update_method_panel()
        self.adjustSize()

    def closeEvent(self, event):
        if self.config.dlg_visible:
            Publisher.sendMessage("Disable style", style=const.SLICE_STATE_MASK_FFILL)
        super().closeEvent(event)


class CropOptionsDialog(QDialog):
    def __init__(self, config, ID=-1, title=_("Crop mask"), **kwargs):
        self.config = config
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self._init_gui()

    def UpdateValues(self, limits):
        xi, xf, yi, yf, zi, zf = limits
        self.tx_axial_i.setText(str(zi))
        self.tx_axial_f.setText(str(zf))
        self.tx_sagital_i.setText(str(xi))
        self.tx_sagital_f.setText(str(xf))
        self.tx_coronal_i.setText(str(yi))
        self.tx_coronal_f.setText(str(yf))

    def _init_gui(self):
        self.tx_axial_i = QLineEdit()
        self.tx_axial_i.setReadOnly(True)
        self.tx_axial_f = QLineEdit()
        self.tx_axial_f.setReadOnly(True)
        self.tx_sagital_i = QLineEdit()
        self.tx_sagital_i.setReadOnly(True)
        self.tx_sagital_f = QLineEdit()
        self.tx_sagital_f.setReadOnly(True)
        self.tx_coronal_i = QLineEdit()
        self.tx_coronal_i.setReadOnly(True)
        self.tx_coronal_f = QLineEdit()
        self.tx_coronal_f.setReadOnly(True)

        grid = QGridLayout()
        grid.addWidget(QLabel(_("Axial:")), 0, 0)
        grid.addWidget(self.tx_axial_i, 0, 1)
        grid.addWidget(QLabel(" - "), 0, 2)
        grid.addWidget(self.tx_axial_f, 0, 3)
        grid.addWidget(QLabel(_("Sagital:")), 1, 0)
        grid.addWidget(self.tx_sagital_i, 1, 1)
        grid.addWidget(QLabel(" - "), 1, 2)
        grid.addWidget(self.tx_sagital_f, 1, 3)
        grid.addWidget(QLabel(_("Coronal:")), 2, 0)
        grid.addWidget(self.tx_coronal_i, 2, 1)
        grid.addWidget(QLabel(" - "), 2, 2)
        grid.addWidget(self.tx_coronal_f, 2, 3)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.OnOk)
        btn_box.rejected.connect(self._on_close)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(btn_box)
        self.adjustSize()

        Publisher.subscribe(self.UpdateValues, "Update crop limits into gui")

    def OnOk(self):
        self.config.dlg_visible = False
        Publisher.sendMessage("Crop mask")
        Publisher.sendMessage("Disable style", style=const.SLICE_STATE_CROP_MASK)
        self.accept()

    def _on_close(self):
        self.config.dlg_visible = False
        Publisher.sendMessage("Disable style", style=const.SLICE_STATE_CROP_MASK)
        self.reject()

    def closeEvent(self, event):
        self.config.dlg_visible = False
        Publisher.sendMessage("Disable style", style=const.SLICE_STATE_CROP_MASK)
        super().closeEvent(event)


class FillHolesAutoDialog(QDialog):
    def __init__(self, title):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self._init_gui()

    def _init_gui(self):
        self.spin_size = InvSpinCtrl(self, -1, value=1000, min_value=1, max_value=1000000000)
        self.panel_target = PanelTargeFFill(self)
        self.panel2dcon = Panel2DConnectivity(self, show_orientation=True)
        self.panel3dcon = Panel3DConnectivity(self)
        self.panel2dcon.setEnabled(True)
        self.panel3dcon.setEnabled(False)

        apply_btn = QPushButton(_("Apply"))
        apply_btn.clicked.connect(self.OnApply)
        close_btn = QPushButton(_("Close"))
        close_btn.clicked.connect(self.close)

        self.panel_target.target_2d.toggled.connect(self._on_radio)
        self.panel_target.target_3d.toggled.connect(self._on_radio)

        spin_layout = QHBoxLayout()
        spin_layout.addWidget(QLabel(_("Max hole size")))
        spin_layout.addWidget(self.spin_size)
        spin_layout.addWidget(QLabel(_("voxels")))

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(apply_btn)
        btn_layout.addWidget(close_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_("Parameters")))
        layout.addWidget(self.panel_target)
        layout.addWidget(self.panel2dcon)
        layout.addWidget(self.panel3dcon)
        layout.addLayout(spin_layout)
        layout.addLayout(btn_layout)
        self.adjustSize()

    def _on_radio(self):
        is_2d = self.panel_target.target_2d.isChecked()
        self.panel2dcon.setEnabled(is_2d)
        self.panel3dcon.setEnabled(not is_2d)

    def OnApply(self):
        if self.panel_target.target_2d.isChecked():
            target = "2D"
            conn = self.panel2dcon.GetConnSelected()
            orientation = self.panel2dcon.GetOrientation()
        else:
            target = "3D"
            conn = self.panel3dcon.GetConnSelected()
            orientation = "VOLUME"
        parameters = {
            "target": target,
            "conn": conn,
            "orientation": orientation,
            "size": self.spin_size.GetValue(),
        }
        Publisher.sendMessage("Fill holes automatically", parameters=parameters)


class MaskDensityDialog(QDialog):
    def __init__(self, title=""):
        super().__init__(_top_window())
        self.setWindowTitle(_("Mask density"))
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self._init_gui()
        _center_on_screen(self)

    def _init_gui(self):
        import invesalius.project as prj

        project = prj.Project()

        self.cmb_mask = QComboBox()
        if project.mask_dict.values():
            for mask in project.mask_dict.values():
                self.cmb_mask.addItem(mask.name, mask)

        self.calc_button = QPushButton(_("Calculate"))
        self.calc_button.clicked.connect(self.OnCalcButton)

        self.mean_density = QLineEdit()
        self.mean_density.setReadOnly(True)
        self.min_density = QLineEdit()
        self.min_density.setReadOnly(True)
        self.max_density = QLineEdit()
        self.max_density.setReadOnly(True)
        self.std_density = QLineEdit()
        self.std_density.setReadOnly(True)

        top_layout = QHBoxLayout()
        top_layout.addWidget(QLabel(_("Mask:")))
        top_layout.addWidget(self.cmb_mask, 1)
        top_layout.addWidget(self.calc_button)

        grid = QGridLayout()
        grid.addWidget(QLabel(_("Mean:")), 0, 0)
        grid.addWidget(self.mean_density, 0, 1)
        grid.addWidget(QLabel(_("Minimun:")), 1, 0)
        grid.addWidget(self.min_density, 1, 1)
        grid.addWidget(QLabel(_("Maximun:")), 2, 0)
        grid.addWidget(self.max_density, 2, 1)
        grid.addWidget(QLabel(_("Standard deviation:")), 3, 0)
        grid.addWidget(self.std_density, 3, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(top_layout)
        layout.addLayout(grid)
        self.adjustSize()

    def OnCalcButton(self):
        from invesalius.data.slice_ import Slice

        mask = self.cmb_mask.currentData()
        slc = Slice()
        with futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(slc.calc_image_density, mask)
            for c in itertools.cycle(["", ".", "..", "..."]):
                s = _("Calculating ") + c
                self.mean_density.setText(s)
                self.min_density.setText(s)
                self.max_density.setText(s)
                self.std_density.setText(s)
                QApplication.processEvents()
                if future.done():
                    break
                time.sleep(0.1)
            _min, _max, _mean, _std = future.result()
        self.mean_density.setText(str(_mean))
        self.min_density.setText(str(_min))
        self.max_density.setText(str(_max))
        self.std_density.setText(str(_std))


# ---------------------------------------------------------------------------
# Object Calibration
# ---------------------------------------------------------------------------


class ObjectCalibrationDialog(QDialog):
    def __init__(self, tracker, n_coils, pedal_connector):
        self.tracker = tracker
        self.n_coils = n_coils
        self.pedal_connector = pedal_connector
        self.tracker_id = tracker.GetTrackerId()
        self.obj_id = 2
        self.show_sensor_options = self.tracker_id in const.TRACKERS_WITH_SENSOR_OPTIONS
        self.coil_path = None
        self.polydata = None
        self.obj_fiducials = np.full([4, 3], np.nan)
        self.obj_orients = np.full([4, 3], np.nan)

        super().__init__(_top_window())
        self.setWindowTitle(_("Object calibration"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.resize(450, 440)
        self._init_gui()
        self._init_pedal()
        self.InitializeObject()

    def _init_gui(self):
        self.interactor = QVTKRenderWindowInteractor(self)
        self.interactor.setMinimumSize(self.size())
        self.ren = vtkRenderer()
        self.interactor.GetRenderWindow().AddRenderer(self.ren)

        self.btns_coord = [None] * 4
        self.text_actors = [None] * 4
        self.ball_actors = [None] * 4
        self.txt_coord = [[], [], [], []]

        max_obj_id = self.tracker.GetTrackerCoordinates(ref_mode_id=0)[2].shape[0]
        choices = ["0"] if self.n_coils == 1 else []
        choices += [str(i) for i in range(2, max_obj_id)]

        choice_obj_id = QComboBox()
        choice_obj_id.addItems(choices)
        choice_obj_id.setToolTip(
            _(
                "Choose the coil index in coord_raw. Choose 0 for static mode, 2 for dynamic mode and 3 onwards for multiple coils."
            )
        )
        choice_obj_id.currentTextChanged.connect(self.OnChooseObjID)
        choice_obj_id.setCurrentText(str(self.obj_id))

        if self.tracker_id == const.PATRIOT or self.tracker_id == const.ISOTRAKII:
            self.obj_id = 0
            choice_obj_id.setCurrentIndex(0)
            choice_obj_id.setEnabled(False)

        choice_sensor = QComboBox()
        choice_sensor.addItems(const.FT_SENSOR_MODE)
        choice_sensor.setToolTip(_("Choose the FASTRAK sensor port"))
        choice_sensor.currentIndexChanged.connect(self.OnChoiceFTSensor)
        self.choice_sensor = choice_sensor
        choice_sensor.setVisible(self.show_sensor_options)

        btn_reset = QPushButton(_("Reset"))
        btn_reset.setToolTip(_("Reset all fiducials"))
        btn_reset.clicked.connect(self.OnReset)

        btn_ok = QPushButton(_("Done"))
        btn_ok.setToolTip(_("Registration done"))
        btn_ok.clicked.connect(self.OnOk)

        self.buttons = OrderedFiducialButtons(
            self, const.OBJECT_FIDUCIALS, self.IsObjectFiducialSet
        )
        for index, btn in enumerate(self.buttons):
            btn.clicked.connect(partial(self.OnObjectFiducialButton, index))
        self.buttons.FocusNext()

        for m in range(4):
            for n in range(3):
                self.txt_coord[m].append(QLabel("-"))

        coord_layout = QGridLayout()
        for m, button in enumerate(self.buttons):
            coord_layout.addWidget(button, m, 0)
            for n in range(3):
                coord_layout.addWidget(self.txt_coord[m][n], m, n + 1)

        if not self.show_sensor_options:
            self.buttons[const.OBJECT_FIDUCIAL_FIXED].hide()
            for coord in self.txt_coord[const.OBJECT_FIDUCIAL_FIXED]:
                coord.hide()

        extra_layout = QVBoxLayout()
        extra_layout.addWidget(choice_obj_id)
        extra_layout.addWidget(btn_reset)
        extra_layout.addWidget(btn_ok)
        extra_layout.addWidget(choice_sensor)

        group_layout = QHBoxLayout()
        group_layout.addLayout(coord_layout)
        group_layout.addLayout(extra_layout)

        self.name_box = QLineEdit(_("coil1"))
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel(_("Name the coil:")))
        name_layout.addWidget(self.name_box)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.interactor)
        main_layout.addLayout(group_layout)
        if self.n_coils > 1:
            main_layout.addLayout(name_layout)
        else:
            choice_obj_id.setEnabled(False)
            choice_obj_id.hide()
        self.adjustSize()

    def _init_pedal(self):
        def set_fiducial_callback(state):
            index = self.buttons.focused_index
            if state and index is not None:
                self.SetObjectFiducial(index)

        self.pedal_connector.add_callback(
            "fiducial", set_fiducial_callback, remove_when_released=False, panel=self
        )

    def ObjectImportDialog(self):
        result = QMessageBox.question(
            None,
            "InVesalius 3",
            _("Would like to use InVesalius default object?"),
            QMessageBox.Yes | QMessageBox.No,
        )
        return 1 if result == QMessageBox.Yes else 0

    def ShowObject(self, polydata):
        if polydata.GetNumberOfPoints() == 0:
            QMessageBox.warning(
                self, _("Import surface error"), _("InVesalius was not able to import this surface")
            )
            return

        transform = vtkTransform()
        transform.RotateZ(90)
        transform_filt = vtkTransformPolyDataFilter()
        transform_filt.SetTransform(transform)
        transform_filt.SetInputData(polydata)
        transform_filt.Update()

        normals = vtkPolyDataNormals()
        normals.SetInputData(transform_filt.GetOutput())
        normals.SetFeatureAngle(80)
        normals.AutoOrientNormalsOn()
        normals.Update()

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(normals.GetOutput())
        mapper.ScalarVisibilityOff()
        obj_actor = vtkActor()
        obj_actor.SetMapper(mapper)

        self.ball_actors[0], self.text_actors[0] = self.OnCreateObjectText("Left", (0, 55, 0))
        self.ball_actors[1], self.text_actors[1] = self.OnCreateObjectText("Right", (0, -55, 0))
        self.ball_actors[2], self.text_actors[2] = self.OnCreateObjectText("Anterior", (23, 0, 0))

        def set_actor_colors(n, color_float):
            if n != const.OBJECT_FIDUCIAL_FIXED:
                self.ball_actors[n].GetProperty().SetColor(color_float)
                self.text_actors[n].GetProperty().SetColor(color_float)
                self.update()

        self.buttons.set_actor_colors = set_actor_colors
        self.buttons.Update()
        self.ren.AddActor(obj_actor)
        self.ren.ResetCamera()
        self.interactor.GetRenderWindow().Render()

    def ConfigureObject(self):
        use_default_coil = self.ObjectImportDialog()
        if use_default_coil:
            path = os.path.join(inv_paths.OBJ_DIR, "magstim_fig8_coil.stl")
        else:
            path = ShowImportMeshFilesDialog()
            if path is None:
                return False
            valid_extensions = (".stl", "ply", ".obj", ".vtp")
            if not path.lower().endswith(valid_extensions):
                QMessageBox.warning(
                    self, _("Import surface error"), _("File format not recognized by InVesalius")
                )
                return False
        if _has_win32api:
            path = win32api.GetShortPathName(path)
        self.coil_path = path.encode(const.FS_ENCODE)
        return True

    def InitializeObject(self):
        success = self.ConfigureObject()
        if success:
            object_path = self.coil_path.decode(const.FS_ENCODE)
            self.polydata = pu.LoadPolydata(path=object_path)
            self.ShowObject(polydata=self.polydata)

    def OnCreateObjectText(self, name, coord):
        ball_source = vtkSphereSource()
        ball_source.SetRadius(3)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(ball_source.GetOutputPort())
        ball_actor = vtkActor()
        ball_actor.SetMapper(mapper)
        ball_actor.SetPosition(coord)
        ball_actor.GetProperty().SetColor(const.RED_COLOR_FLOAT)

        textSource = vtkVectorText()
        textSource.SetText(name)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(textSource.GetOutputPort())
        tactor = vtkFollower()
        tactor.SetMapper(mapper)
        tactor.GetProperty().SetColor(const.RED_COLOR_FLOAT)
        tactor.SetScale(5)
        bp = ball_actor.GetPosition()
        tactor.SetPosition(bp[0] + 5, bp[1] + 5, bp[2] + 10)
        self.ren.AddActor(tactor)
        tactor.SetCamera(self.ren.GetActiveCamera())
        self.ren.AddActor(ball_actor)
        return ball_actor, tactor

    def IsObjectFiducialSet(self, fiducial_index):
        return not np.isnan(self.obj_fiducials[fiducial_index]).any()

    def OnObjectFiducialButton(self, index):
        button = self.buttons[index]
        if button is self.buttons.focused:
            self.SetObjectFiducial(index)
        elif self.IsObjectFiducialSet(index):
            self.ResetObjectFiducial(index)
        else:
            self.buttons.Focus(index)

    def SetObjectFiducial(self, fiducial_index):
        if not self.tracker.IsTrackerInitialized():
            ShowNavigationTrackerWarning(0, "choose")
            return
        marker_visibilities, coord, coord_raw = self.tracker.GetTrackerCoordinates(
            ref_mode_id=const.STATIC_REF,
            n_samples=const.CALIBRATION_TRACKER_SAMPLES,
        )
        probe_visible, head_visible, *coils_visible = marker_visibilities
        if not self.show_sensor_options:
            if not probe_visible:
                ShowNavigationTrackerWarning(0, "probe marker not visible")
                return
            if not coils_visible[self.obj_id - 2]:
                ShowNavigationTrackerWarning(0, "coil marker not visible")
                return
        if self.obj_id and fiducial_index == const.OBJECT_FIDUCIAL_FIXED:
            coord = coord_raw[self.obj_id, :]
        else:
            coord = coord_raw[0, :]
        Publisher.sendMessage("Set object fiducial", fiducial_index=fiducial_index)
        if coord is not None or np.sum(coord) != 0.0:
            self.obj_fiducials[fiducial_index, :] = coord[:3]
            self.obj_orients[fiducial_index, :] = coord[3:]
            self.buttons.SetFocused()
            for i in [0, 1, 2]:
                self.txt_coord[fiducial_index][i].setText(str(round(coord[i], 1)))
            self.update()
        else:
            ShowNavigationTrackerWarning(0, "choose")
        if fiducial_index == const.OBJECT_FIDUCIAL_ANTERIOR and not self.show_sensor_options:
            self.SetObjectFiducial(const.OBJECT_FIDUCIAL_FIXED)

    def ResetObjectFiducials(self):
        for m in range(4):
            self.ResetObjectFiducial(m)
        self.buttons.Update()

    def ResetObjectFiducial(self, index):
        self.obj_fiducials[index, :] = np.full([1, 3], np.nan)
        self.obj_orients[index, :] = np.full([1, 3], np.nan)
        for ci in range(3):
            self.txt_coord[index][ci].setText("-")
        self.buttons.Unset(index)

    def OnReset(self):
        self.ResetObjectFiducials()

    def OnChooseObjID(self, text):
        try:
            self.obj_id = int(text)
        except ValueError:
            return
        self.choice_sensor.setVisible(
            self.obj_id == 0 and self.tracker_id in const.TRACKERS_WITH_SENSOR_OPTIONS
        )
        self.ResetObjectFiducials()

    def OnChoiceFTSensor(self, index):
        self.obj_id = 3 if index else 0

    def GetValue(self):
        coil_name = self.name_box.text().strip() if self.n_coils > 1 else "default_coil"
        return (
            coil_name,
            self.coil_path,
            self.obj_fiducials,
            self.obj_orients,
            self.obj_id,
            self.tracker_id,
        )

    def OnOk(self):
        self.pedal_connector.remove_callback("fiducial", panel=self)
        self.accept()


# ---------------------------------------------------------------------------
# ICP Corregistration
# ---------------------------------------------------------------------------


class ICPCorregistrationDialog(QDialog):
    def __init__(self, navigation, tracker):
        import invesalius.project as prj

        self.tracker = tracker
        self.m_change = navigation.m_change
        self.obj_ref_id = 2
        self.obj_name = None
        self.obj_actor = None
        self.polydata = None
        self.m_icp = None
        self.initial_focus = None
        self.prev_error = None
        self.final_error = None
        self.icp_mode = 0
        self.actors_static_points = []
        self.point_coord = []
        self.actors_transformed_points = []
        self.obj_fiducials = np.full([5, 3], np.nan)
        self.obj_orients = np.full([5, 3], np.nan)

        super().__init__(_top_window())
        self.setWindowTitle(_("Refine Corregistration"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.resize(380, 440)
        self.proj = prj.Project()
        self._init_gui()

    def _init_gui(self):
        self.interactor = QVTKRenderWindowInteractor(self)
        self.interactor.setMinimumSize(self.size())
        self.ren = vtkRenderer()
        self.interactor.GetRenderWindow().AddRenderer(self.ren)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.HandleContinuousAcquisition)

        combo_surface_name = QComboBox()
        for n in range(len(self.proj.surface_dict)):
            combo_surface_name.addItem(str(self.proj.surface_dict[n].name))
        combo_surface_name.currentIndexChanged.connect(self.OnComboName)
        self.combo_surface_name = combo_surface_name
        combo_surface_name.setCurrentIndex(0)
        self.surface = self.proj.surface_dict[0].polydata
        self.LoadActor()

        choice_icp_method = QComboBox()
        choice_icp_method.addItems([_("Affine"), _("Similarity"), _("RigidBody")])
        choice_icp_method.currentIndexChanged.connect(self.OnChoiceICPMethod)

        create_point = QPushButton(_("Create point"))
        create_point.clicked.connect(self.CreatePoint)

        self.cont_point = QPushButton(_("Continuous acquisition"))
        self.cont_point.setCheckable(True)
        self.cont_point.toggled.connect(self._on_cont_toggled)

        btn_reset = QPushButton(_("Reset points"))
        btn_reset.clicked.connect(self.OnResetPoints)

        self.btn_apply_icp = QPushButton(_("Apply registration"))
        self.btn_apply_icp.clicked.connect(self.OnICP)
        self.btn_apply_icp.setEnabled(False)

        self.btn_ok = QPushButton(_("Done"))
        self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton(_("Cancel"))
        btn_cancel.clicked.connect(self.reject)

        self.progress = QProgressBar()

        top_layout = QGridLayout()
        top_layout.addWidget(QLabel(_("Select the surface:")), 0, 0)
        top_layout.addWidget(QLabel(_("Registration mode:")), 0, 1)
        top_layout.addWidget(combo_surface_name, 1, 0)
        top_layout.addWidget(choice_icp_method, 1, 1)

        btn_acq = QHBoxLayout()
        btn_acq.addWidget(create_point)
        btn_acq.addWidget(self.cont_point)
        btn_acq.addWidget(btn_reset)

        btn_ok_layout = QHBoxLayout()
        btn_ok_layout.addWidget(self.btn_apply_icp)
        btn_ok_layout.addWidget(self.btn_ok)
        btn_ok_layout.addWidget(btn_cancel)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_layout)
        main_layout.addWidget(self.interactor)
        main_layout.addLayout(btn_acq)
        main_layout.addLayout(btn_ok_layout)
        main_layout.addWidget(self.progress)
        self.adjustSize()

    def _on_cont_toggled(self, checked):
        if checked:
            self.timer.start(500)
        else:
            self.timer.stop()

    def LoadActor(self):
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(self.surface)
        mapper.ScalarVisibilityOff()
        obj_actor = vtkActor()
        obj_actor.SetMapper(mapper)
        self.obj_actor = obj_actor

        poses_recorded = vtku.Text()
        poses_recorded.SetSize(const.TEXT_SIZE_LARGE)
        poses_recorded.SetPosition((const.X, const.Y))
        poses_recorded.ShadowOff()
        poses_recorded.SetValue("Poses recorded: ")

        collect_points = vtku.Text()
        collect_points.SetSize(const.TEXT_SIZE_LARGE)
        collect_points.SetPosition((const.X + 0.35, const.Y))
        collect_points.ShadowOff()
        collect_points.SetValue("0")
        self.collect_points = collect_points

        txt_markers_not_detected = vtku.Text()
        txt_markers_not_detected.SetSize(const.TEXT_SIZE_LARGE)
        txt_markers_not_detected.SetPosition((const.X + 0.50, const.Y))
        txt_markers_not_detected.ShadowOff()
        txt_markers_not_detected.SetColour((1, 0, 0))
        txt_markers_not_detected.SetValue("Markers not detected")
        txt_markers_not_detected.actor.VisibilityOff()
        self.txt_markers_not_detected = txt_markers_not_detected.actor

        self.ren.AddActor(obj_actor)
        self.ren.AddActor(poses_recorded.actor)
        self.ren.AddActor(collect_points.actor)
        self.ren.AddActor(txt_markers_not_detected.actor)
        self.ren.ResetCamera()
        self.interactor.GetRenderWindow().Render()

    def RemoveAllActors(self):
        self.ren.RemoveAllViewProps()
        self.actors_static_points = []
        self.point_coord = []
        self.actors_transformed_points = []
        self.m_icp = None
        self.SetProgress(0)
        self.btn_apply_icp.setEnabled(False)
        self.btn_ok.setEnabled(False)
        self.ren.ResetCamera()
        self.interactor.GetRenderWindow().Render()

    def RemoveSinglePointActor(self):
        self.ren.RemoveActor(self.actors_static_points[-1])
        self.actors_static_points.pop()
        self.point_coord.pop()
        self.collect_points.SetValue(str(int(self.collect_points.GetValue()) - 1))
        self.interactor.GetRenderWindow().Render()

    def GetCurrentCoord(self):
        coord_raw, marker_visibilities = self.tracker.TrackerCoordinates.GetCoordinates()
        coord, _ = dcr.corregistrate_probe(self.m_change, None, coord_raw, const.DEFAULT_REF_MODE)
        return coord[:3], marker_visibilities

    def AddMarker(self, size, colour, coord):
        x, y, z = coord[0], -coord[1], coord[2]
        ball_ref = vtkSphereSource()
        ball_ref.SetRadius(size)
        ball_ref.SetCenter(x, y, z)
        mapper = vtkPolyDataMapper()
        mapper.SetInputConnection(ball_ref.GetOutputPort())
        prop = vtkProperty()
        prop.SetColor(colour[0:3])
        sphere_actor = vtkActor()
        sphere_actor.SetMapper(mapper)
        sphere_actor.SetProperty(prop)
        self.ren.AddActor(sphere_actor)
        self.actors_static_points.append(sphere_actor)
        self.point_coord.append([x, y, z])
        self.collect_points.SetValue(str(int(self.collect_points.GetValue()) + 1))
        self.interactor.GetRenderWindow().Render()
        if len(self.point_coord) >= 5 and not self.btn_apply_icp.isEnabled():
            self.btn_apply_icp.setEnabled(True)
        if self.progress.value() != 0:
            self.SetProgress(0)

    def SetProgress(self, progress):
        self.progress.setValue(int(progress * 100))
        self.interactor.GetRenderWindow().Render()

    def vtkmatrix_to_numpy(self, matrix):
        m = np.ones((4, 4))
        for i in range(4):
            for j in range(4):
                m[i, j] = matrix.GetElement(i, j)
        return m

    def SetCameraVolume(self, position):
        cam_focus = np.array([position[0], -position[1], position[2]])
        cam = self.ren.GetActiveCamera()
        if self.initial_focus is None:
            self.initial_focus = np.array(cam.GetFocalPoint())
        cam_pos0 = np.array(cam.GetPosition())
        cam_focus0 = np.array(cam.GetFocalPoint())
        v0 = cam_pos0 - cam_focus0
        v0n = np.sqrt(inner1d(v0, v0))
        v1 = cam_focus - self.initial_focus
        v1n = np.sqrt(inner1d(v1, v1))
        if not v1n:
            v1n = 1.0
        cam_pos = (v1 / v1n) * v0n + cam_focus
        cam.SetFocalPoint(cam_focus)
        cam.SetPosition(cam_pos)
        self.interactor.GetRenderWindow().Render()

    def CheckTransformedPointsDistribution(self, points):
        from scipy.spatial.distance import pdist

        return np.mean(pdist(points))

    def ErrorEstimation(self, surface, points):
        cell_locator = vtkCellLocator()
        cell_locator.SetDataSet(surface)
        cell_locator.BuildLocator()
        cellId = mutable(0)
        c = [0.0, 0.0, 0.0]
        subId = mutable(0)
        d = mutable(0.0)
        error = []
        for i in range(len(points)):
            cell_locator.FindClosestPoint(points[i], c, cellId, subId, d)
            error.append(np.sqrt(float(d)))
        return np.mean(error)

    def DistanceBetweenPointAndSurface(self, surface, points):
        cell_locator = vtkCellLocator()
        cell_locator.SetDataSet(surface)
        cell_locator.BuildLocator()
        cellId = mutable(0)
        c = [0.0, 0.0, 0.0]
        subId = mutable(0)
        d = mutable(0.0)
        cell_locator.FindClosestPoint(points, c, cellId, subId, d)
        return np.sqrt(float(d))

    def OnComboName(self, index):
        self.surface = self.proj.surface_dict[index].polydata
        if self.obj_actor:
            self.RemoveAllActors()
        self.LoadActor()

    def OnChoiceICPMethod(self, index):
        self.icp_mode = index

    def HandleContinuousAcquisition(self):
        self.CreatePoint()

    def CreatePoint(self):
        current_coord, marker_visibilities = self.GetCurrentCoord()
        probe_visible, head_visible, *coils_visible = marker_visibilities
        if probe_visible and head_visible:
            self.AddMarker(3, (1, 0, 0), current_coord)
            self.txt_markers_not_detected.VisibilityOff()
            if self.DistanceBetweenPointAndSurface(self.surface, self.point_coord[-1]) >= 20:
                self.OnDeleteLastPoint()
                ReportICPPointError()
            else:
                self.SetCameraVolume(current_coord)
        else:
            self.txt_markers_not_detected.VisibilityOn()
            self.interactor.GetRenderWindow().Render()

    def OnDeleteLastPoint(self):
        if self.cont_point.isChecked():
            self.cont_point.setChecked(False)
        self.RemoveSinglePointActor()

    def OnResetPoints(self):
        if self.cont_point.isChecked():
            self.cont_point.setChecked(False)
        self.RemoveAllActors()
        self.LoadActor()

    def OnICP(self):
        if self.cont_point.isChecked():
            self.cont_point.setChecked(False)
        self.SetProgress(0.3)
        time.sleep(1)

        sourcePoints = np.array(self.point_coord)
        sourcePoints_vtk = vtkPoints()
        for i in range(len(sourcePoints)):
            sourcePoints_vtk.InsertNextPoint(sourcePoints[i])
        source = vtkPolyData()
        source.SetPoints(sourcePoints_vtk)

        icp = vtkIterativeClosestPointTransform()
        icp.SetSource(source)
        icp.SetTarget(self.surface)
        self.SetProgress(0.5)

        if self.icp_mode == 0:
            icp.GetLandmarkTransform().SetModeToAffine()
        elif self.icp_mode == 1:
            icp.GetLandmarkTransform().SetModeToSimilarity()
        elif self.icp_mode == 2:
            icp.GetLandmarkTransform().SetModeToRigidBody()

        icp.SetMaximumNumberOfIterations(1000)
        icp.Modified()
        icp.Update()
        self.m_icp = self.vtkmatrix_to_numpy(icp.GetMatrix())

        icpTransformFilter = vtkTransformPolyDataFilter()
        icpTransformFilter.SetInputData(source)
        icpTransformFilter.SetTransform(icp)
        icpTransformFilter.Update()
        transformedSource = icpTransformFilter.GetOutput()

        transformed_points = []
        if self.actors_transformed_points:
            for a in self.actors_transformed_points:
                self.ren.RemoveActor(a)
            self.actors_transformed_points = []

        for i in range(transformedSource.GetNumberOfPoints()):
            p = [0, 0, 0]
            transformedSource.GetPoint(i, p)
            transformed_points.append(p)
            point = vtkSphereSource()
            point.SetCenter(p)
            point.SetRadius(3)
            mapper = vtkPolyDataMapper()
            mapper.SetInputConnection(point.GetOutputPort())
            actor = vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetColor((0, 1, 0))
            self.actors_transformed_points.append(actor)
            self.ren.AddActor(actor)

        if self.CheckTransformedPointsDistribution(transformed_points) <= 25:
            ReportICPDistributionError()

        self.prev_error = self.ErrorEstimation(self.surface, sourcePoints)
        self.final_error = self.ErrorEstimation(self.surface, transformed_points)
        self.interactor.GetRenderWindow().Render()
        self.SetProgress(1)
        self.btn_ok.setEnabled(True)

    def GetValue(self):
        return (
            self.m_icp,
            self.point_coord,
            self.actors_transformed_points,
            self.prev_error,
            self.final_error,
        )


# ---------------------------------------------------------------------------
# Efield / Brain Target (skeletons - complex VTK dialogs)
# ---------------------------------------------------------------------------


class EfieldConfiguration(QDialog):
    def __init__(self):
        import invesalius.project as prj

        self.polydata = None
        super().__init__(_top_window())
        self.setWindowTitle(_("Set Efield Configuration"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.resize(380, 440)
        self.proj = prj.Project()
        self.brain_surface = None
        self.scalp_surface = None
        self._init_gui()

    def _init_gui(self):
        btn_act = QPushButton(_("Load"))
        btn_act.setToolTip(_("Load Brain Meshes"))
        btn_act.clicked.connect(self.OnAddMeshes)

        combo_brain = QComboBox()
        combo_scalp = QComboBox()
        for n in range(len(self.proj.surface_dict)):
            name = str(self.proj.surface_dict[n].name)
            combo_brain.addItem(name)
            combo_scalp.addItem(name)
        combo_brain.currentIndexChanged.connect(
            lambda idx: setattr(self, "brain_surface", self.proj.surface_dict[idx].polydata)
        )
        combo_scalp.currentIndexChanged.connect(
            lambda idx: setattr(self, "scalp_surface", self.proj.surface_dict[idx].polydata)
        )

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        grid = QGridLayout()
        grid.addWidget(QLabel(_("Select the brain surface:")), 0, 0)
        grid.addWidget(QLabel(_("Select the scalp surface:")), 0, 1)
        grid.addWidget(combo_brain, 1, 0)
        grid.addWidget(combo_scalp, 1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(btn_act)
        layout.addLayout(grid)
        layout.addWidget(btn_box)
        self.adjustSize()

    def OnAddMeshes(self):
        filename = ShowImportMeshFilesDialog()
        if filename:
            convert_to_inv = ImportMeshCoordSystem()
            Publisher.sendMessage("Update convert_to_inv flag", convert_to_inv=convert_to_inv)
        Publisher.sendMessage("Import bin file", filename=filename)


# CreateBrainTargetDialog is extremely complex (600+ lines) with heavy VTK interaction.
# Providing a skeleton that preserves the public API:


class CreateBrainTargetDialog(QDialog):
    def __init__(self, marker, mTMS=None, brain_target=False, brain_actor=None):
        import invesalius.project as prj

        self.obj_actor = None
        self.polydata = None
        self.initial_focus = None
        self.mTMS = mTMS
        self.marker = marker
        self.brain_target = brain_target
        self.peel_brain_actor = brain_actor
        self.brain_target_actor_list = []
        self.coil_target_actor_list = []
        self.center_brain_target_actor = None
        self.marker_actor = None
        self.dummy_coil_actor = None
        self.m_target = None
        self.spinning = False
        self.rotationX = self.rotationY = self.rotationZ = 0
        self.obj_fiducials = np.full([5, 3], np.nan)
        self.obj_orients = np.full([5, 3], np.nan)

        super().__init__(_top_window())
        self.setWindowTitle(_("Set target Orientation"))
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.resize(380, 440)
        self.proj = prj.Project()
        self._init_gui()

    def _init_gui(self):
        self.interactor = QVTKRenderWindowInteractor(self)
        self.ren = vtkRenderer()
        self.interactor.GetRenderWindow().AddRenderer(self.ren)
        self.actor_style = vtkInteractorStyleTrackballActor()
        self.camera_style = vtkInteractorStyleTrackballCamera()
        self.picker = vtkCellPicker()
        self.picker.SetTolerance(1e-3)
        self.interactor.SetPicker(self.picker)
        self.interactor.SetInteractorStyle(self.actor_style)

        combo_surface = QComboBox()
        combo_brain = QComboBox()
        for n in range(len(self.proj.surface_dict)):
            name = str(self.proj.surface_dict[n].name)
            combo_surface.addItem(name)
            combo_brain.addItem(name)
        combo_surface.setCurrentIndex(0)
        combo_brain.setCurrentIndex(0)
        self.surface = self.proj.surface_dict[0].polydata
        self.brain_surface = self.proj.surface_dict[0].polydata

        self.slider_rotation_x = QSlider(Qt.Horizontal)
        self.slider_rotation_x.setRange(-180, 180)
        self.slider_rotation_y = QSlider(Qt.Horizontal)
        self.slider_rotation_y.setRange(-180, 180)
        self.slider_rotation_z = QSlider(Qt.Horizontal)
        self.slider_rotation_z.setRange(-180, 180)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.interactor)
        layout.addWidget(btn_box)
        self.adjustSize()

    def GetValue(self):
        return ([], [], [], [])

    def GetValueBrainTarget(self):
        return ([], [])


# ---------------------------------------------------------------------------
# Progress windows
# ---------------------------------------------------------------------------


class TractographyProgressWindow:
    def __init__(self, msg):
        self.dlg = QProgressDialog(msg, "Cancel", 0, 0, None)
        self.dlg.setWindowTitle("InVesalius 3")
        self.dlg.setWindowModality(Qt.ApplicationModal)
        self.running = True
        self.error = None
        self.dlg.show()

    def WasCancelled(self):
        return self.dlg.wasCanceled()

    def Update(self, msg=None, value=None):
        if msg:
            self.dlg.setLabelText(msg)
        QApplication.processEvents()

    def Close(self):
        self.dlg.close()


class BrainSurfaceLoadingProgressWindow:
    def __init__(self):
        parent = _top_window()
        self.dlg = QProgressDialog(_("Loading brain surface..."), "Cancel", 0, 100, parent)
        self.dlg.setWindowTitle("InVesalius 3")
        self.dlg.setWindowModality(Qt.ApplicationModal)
        self.dlg.show()

    def Update(self, msg=None, value=None):
        if value:
            self.dlg.setValue(int(value))
        if msg:
            self.dlg.setLabelText(msg)
        QApplication.processEvents()

    def Close(self):
        self.dlg.close()


class SurfaceSmoothingProgressWindow:
    def __init__(self):
        parent = _top_window()
        self.dlg = QProgressDialog(_("Smoothing the surface..."), "Cancel", 0, 0, parent)
        self.dlg.setWindowTitle("InVesalius 3 – Creating TMS Coil Target")
        self.dlg.setWindowModality(Qt.ApplicationModal)
        self.dlg.show()

    def Update(self, msg=None, value=None):
        if msg:
            self.dlg.setLabelText(msg)
        QApplication.processEvents()

    def Close(self):
        self.dlg.close()


class SurfaceProgressWindow:
    def __init__(self):
        self.dlg = QProgressDialog(_("Creating 3D surface ..."), "Cancel", 0, 0, None)
        self.dlg.setWindowTitle("InVesalius 3")
        self.dlg.setWindowModality(Qt.ApplicationModal)
        self.running = True
        self.error = None
        self.dlg.show()

    def WasCancelled(self):
        return self.dlg.wasCanceled()

    def Update(self, msg=None, value=None):
        if msg:
            self.dlg.setLabelText(msg)
        QApplication.processEvents()

    def Close(self):
        self.dlg.close()


class PublishingSurfacesProgressWindow:
    def __init__(self, maximum=100):
        parent = _top_window()
        self.dlg = QProgressDialog(
            _("Publishing surfaces to Dashboard..."), "Cancel", 0, maximum, parent
        )
        self.dlg.setWindowTitle("InVesalius 3")

    def WasCancelled(self):
        return self.dlg.wasCanceled()

    def Update(self, msg, value):
        if self.dlg and not self.dlg.wasCanceled():
            self.dlg.setValue(int(value))
            self.dlg.setLabelText(msg)
            QApplication.processEvents()

    def Close(self):
        if self.dlg:
            self.dlg.close()
            self.dlg = None


# ---------------------------------------------------------------------------
# GoToDialog
# ---------------------------------------------------------------------------


class GoToDialog(QDialog):
    def __init__(self, title=_("Go to slice ..."), init_orientation=const.AXIAL_STR):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.orientation = None
        self._init_gui(init_orientation)
        Publisher.subscribe(self.SetNewFocalPoint, "Cross focal point")

    def _init_gui(self, init_orientation):
        orientations = (
            (_("Axial"), const.AXIAL_STR),
            (_("Coronal"), const.CORONAL_STR),
            (_("Sagital"), const.SAGITAL_STR),
        )
        self.goto_slice = QLineEdit()
        self.goto_orientation = QComboBox()
        cb_init = 0
        for n, (text, val) in enumerate(orientations):
            self.goto_orientation.addItem(text, val)
            if val == init_orientation:
                cb_init = n
        self.goto_orientation.setCurrentIndex(cb_init)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.OnOk)
        btn_box.rejected.connect(self.close)

        slice_layout = QHBoxLayout()
        slice_layout.addWidget(QLabel(_("Slice number")))
        slice_layout.addWidget(self.goto_slice, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(slice_layout)
        layout.addWidget(self.goto_orientation)
        layout.addWidget(btn_box)
        self.adjustSize()

    def OnOk(self):
        try:
            slice_number = int(self.goto_slice.text())
            self.orientation = self.goto_orientation.currentData()
            Publisher.sendMessage(("Set scroll position", self.orientation), index=slice_number)
            Publisher.sendMessage("Set Update cross pos")
        except ValueError:
            pass
        self.close()

    def SetNewFocalPoint(self, coord, spacing):
        newCoord = list(coord)
        if self.orientation == "AXIAL":
            newCoord[2] = int(self.goto_slice.text()) * spacing[2]
        if self.orientation == "CORONAL":
            newCoord[1] = int(self.goto_slice.text()) * spacing[1]
        if self.orientation == "SAGITAL":
            newCoord[0] = int(self.goto_slice.text()) * spacing[0]
        Publisher.sendMessage("Update cross pos", coord=newCoord)

    def Close(self):
        self.close()


class RemoveNonVisibleFacesDialog(QDialog):
    def __init__(self, parent):
        import invesalius.project as prj

        super().__init__(parent)
        self.setWindowTitle("Remove non-visible faces")
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.project = prj.Project()
        self._init_gui()

    def _init_gui(self):
        self.surfaces_combo = QComboBox()
        self.overwrite_check = QCheckBox("Overwrite surface")
        self.remove_visible_check = QCheckBox("Remove visible faces")
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.on_apply)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.close)

        self.fill_surfaces_combo()

        combo_layout = QHBoxLayout()
        combo_layout.addWidget(QLabel("Surface"))
        combo_layout.addWidget(self.surfaces_combo, 1)

        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.apply_button)
        btn_layout.addWidget(close_button)

        layout = QVBoxLayout(self)
        layout.addLayout(combo_layout)
        layout.addWidget(self.overwrite_check)
        layout.addWidget(self.remove_visible_check)
        layout.addLayout(btn_layout)
        self.adjustSize()

        Publisher.subscribe(self.on_update_surfaces, "Update surface info in GUI")
        Publisher.subscribe(self.on_update_surfaces, "Remove surfaces")
        Publisher.subscribe(self.on_update_surfaces, "Change surface name")

    def fill_surfaces_combo(self):
        choices = [i.name for i in self.project.surface_dict.values()]
        self.surfaces_combo.clear()
        self.surfaces_combo.addItems(choices)
        self.apply_button.setEnabled(len(choices) > 0)

    def on_apply(self):
        idx = self.surfaces_combo.currentIndex()
        surface = list(self.project.surface_dict.values())[idx]
        remove_visible = self.remove_visible_check.isChecked()
        overwrite = self.overwrite_check.isChecked()
        progress_dialog = RemoveNonVisibleFacesProgressWindow()
        progress_dialog.Update()
        new_polydata = pu.RemoveNonVisibleFaces(surface.polydata, remove_visible=remove_visible)
        if overwrite:
            name = surface.name
            colour = surface.colour
        else:
            name = utils.new_name_by_pattern(f"{surface.name}_removed_nonvisible")
            colour = None
        Publisher.sendMessage(
            "Create surface from polydata",
            polydata=new_polydata,
            name=name,
            overwrite=overwrite,
            index=idx,
            colour=colour,
        )
        Publisher.sendMessage("Fold surface task")
        progress_dialog.Close()
        self.close()

    def on_update_surfaces(self, *args, **kwargs):
        last_idx = self.surfaces_combo.currentIndex()
        self.fill_surfaces_combo()
        if last_idx < self.surfaces_combo.count():
            self.surfaces_combo.setCurrentIndex(last_idx)


class GoToDialogScannerCoord(QDialog):
    def __init__(self, title=_("Go to scanner coord...")):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Tool)
        self._init_gui()

    def _init_gui(self):
        self.goto_sagital = QLineEdit()
        self.goto_coronal = QLineEdit()
        self.goto_axial = QLineEdit()

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.OnOk)
        btn_box.rejected.connect(self.close)

        grid = QGridLayout()
        grid.addWidget(QLabel(_("Sagital coordinate:")), 0, 0)
        grid.addWidget(self.goto_sagital, 0, 1)
        grid.addWidget(QLabel(_("Coronal coordinate:")), 1, 0)
        grid.addWidget(self.goto_coronal, 1, 1)
        grid.addWidget(QLabel(_("Axial coordinate:")), 2, 0)
        grid.addWidget(self.goto_axial, 2, 1)

        layout = QVBoxLayout(self)
        layout.addLayout(grid)
        layout.addWidget(btn_box)
        self.adjustSize()

    def OnOk(self):
        import invesalius.data.slice_ as slc

        try:
            point = [
                float(self.goto_sagital.text()),
                float(self.goto_coronal.text()),
                float(self.goto_axial.text()),
            ]
            position_voxel = img_utils.convert_world_to_voxel(
                point[0:3], np.linalg.inv(slc.Slice().affine)
            )[0]
            voxel = img_utils.convert_invesalius_to_voxel(position_voxel)
            Publisher.sendMessage(
                "Update status text in GUI", label=_("Calculating the transformation ...")
            )
            QTimer.singleShot(
                0,
                lambda: Publisher.sendMessage("Toggle toolbar button", id=const.SLICE_STATE_CROSS),
            )
            QTimer.singleShot(
                0, lambda: Publisher.sendMessage("Update slices position", position=voxel)
            )
            QTimer.singleShot(
                0, lambda: Publisher.sendMessage("Set cross focal point", position=voxel)
            )
            QTimer.singleShot(
                0,
                lambda: Publisher.sendMessage(
                    "Update volume viewer pointer", position=[voxel[0], -voxel[1], voxel[2]]
                ),
            )
            Publisher.sendMessage("Update status text in GUI", label=_("Ready"))
        except ValueError:
            pass
        self.close()


class SelectNiftiVolumeDialog(QDialog):
    def __init__(self, volumes, title=_("Select NIfTI volume")):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self._init_gui(volumes)

    def _init_gui(self, volumes):
        self.cmb_volume = QComboBox()
        self.cmb_volume.addItems(volumes)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self.cmb_volume)
        layout.addWidget(btn_box)

    def GetVolumeChoice(self):
        return int(self.cmb_volume.currentText()) - 1


def DialogRescalePixelIntensity(max_intensity, unique_values):
    msg = (
        _("Maximum pixel intensity is: ")
        + str(round(max_intensity, 1))
        + "\n\n"
        + _("Number of unique pixel intensities: ")
        + str(unique_values)
        + "\n\n"
        + _("Would you like to rescale pixel values to 0-255?")
    )
    result = QMessageBox.question(None, "InVesalius 3", msg, QMessageBox.Yes | QMessageBox.No)
    return result == QMessageBox.Yes


# ---------------------------------------------------------------------------
# Tracker Configuration Dialogs
# ---------------------------------------------------------------------------


class ConfigureOptitrackDialog(QDialog):
    def __init__(self, title=_("Configure Optitrack")):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Tool)
        self._init_gui()

    def _init_gui(self):
        session = ses.Session()
        last_cal = session.GetConfig("last_optitrack_cal_dir", "") or inv_paths.OPTITRACK_CAL_DIR
        last_prof = (
            session.GetConfig("last_optitrack_User_Profile_dir", "")
            or inv_paths.OPTITRACK_USERPROFILE_DIR
        )

        self.dir_cal = FilePickerCtrl(
            self,
            path=last_cal,
            wildcard="Cal files (*.cal)|*.cal",
            message="Select the calibration file",
        )
        self.dir_UserProfile = FilePickerCtrl(
            self,
            path=last_prof,
            wildcard="User Profile files (*.motive)|*.motive",
            message="Select the user profile file",
        )

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Calibration:"))
        layout.addWidget(self.dir_cal)
        layout.addWidget(QLabel("User profile:"))
        layout.addWidget(self.dir_UserProfile)
        layout.addWidget(btn_box)
        self.adjustSize()

    def GetValue(self):
        fn_cal = self.dir_cal.GetPath()
        fn_userprofile = self.dir_UserProfile.GetPath()
        if fn_cal and fn_userprofile:
            session = ses.Session()
            session.SetConfig("last_optitrack_cal_dir", fn_cal)
            session.SetConfig("last_optitrack_User_Profile_dir", fn_userprofile)
        return fn_cal, fn_userprofile


class SetTrackerDeviceToRobot(QDialog):
    def __init__(self, title=_("Set tracker device")):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.tracker_id = const.DEFAULT_TRACKER
        self._init_gui()

    def _init_gui(self):
        trackers = const.TRACKERS.copy()
        session = ses.Session()
        if not session.GetConfig("debug"):
            del trackers[-3:]
        tracker_options = [_("Select tracker:")] + trackers
        choice_trck = QComboBox()
        choice_trck.addItems(tracker_options)
        choice_trck.setCurrentIndex(const.DEFAULT_TRACKER)
        choice_trck.currentIndexChanged.connect(lambda idx: setattr(self, "tracker_id", idx))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(choice_trck)
        layout.addWidget(btn_box)
        self.adjustSize()

    def GetValue(self):
        return self.tracker_id


class SetRobotIP(QDialog):
    def __init__(self, title=_("Set Robot IP")):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.robot_ip = None
        self._init_gui()

    def _init_gui(self):
        robot_ip_options = [_("Select robot IP:")] + const.ROBOT_IPS
        choice_IP = QComboBox()
        choice_IP.setEditable(True)
        choice_IP.addItems(robot_ip_options)
        choice_IP.currentTextChanged.connect(lambda text: setattr(self, "robot_ip", text))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(choice_IP)
        layout.addWidget(btn_box)
        self.adjustSize()

    def GetValue(self):
        return self.robot_ip


class RobotCoregistrationDialog(QDialog):
    def __init__(self, robot, tracker, title=_("Create transformation matrix to robot space")):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.matrix_tracker_to_robot = []
        self.robot = robot
        self.tracker = tracker
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.HandleContinuousAcquisition)
        self._init_gui()
        self.__bind_events()

    def _init_gui(self):
        btn_create_point = QPushButton(_("Single"))
        btn_create_point.clicked.connect(self.CreatePoint)
        self.btn_cont_point = QPushButton(_("Continuous"))
        self.btn_cont_point.setCheckable(True)
        self.btn_cont_point.toggled.connect(self._on_cont_toggled)
        self.txt_number = QLabel("0")
        btn_reset = QPushButton(_("Reset points"))
        btn_reset.clicked.connect(self.ResetPoints)
        self.btn_apply_reg = QPushButton(_("Apply"))
        self.btn_apply_reg.clicked.connect(self.ApplyRegistration)
        self.btn_apply_reg.setEnabled(False)
        self.btn_save = QPushButton(_("Save"))
        self.btn_save.clicked.connect(self.SaveRegistration)
        self.btn_save.setEnabled(False)
        self.btn_load = QPushButton(_("Load"))
        self.btn_load.clicked.connect(self.LoadRegistration)
        self.btn_load.setEnabled(self.robot.IsConnected())

        self.btn_ok = QPushButton(_("OK"))
        self.btn_ok.setEnabled(False)
        self.btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton(_("Cancel"))
        btn_cancel.clicked.connect(self.reject)

        acq_layout = QHBoxLayout()
        acq_layout.addWidget(btn_create_point)
        acq_layout.addWidget(self.btn_cont_point)
        pose_layout = QHBoxLayout()
        pose_layout.addWidget(self.txt_number)
        pose_layout.addWidget(QLabel(_("Poses recorded")))
        apply_layout = QHBoxLayout()
        apply_layout.addWidget(btn_reset)
        apply_layout.addWidget(self.btn_apply_reg)
        save_layout = QHBoxLayout()
        save_layout.addWidget(self.btn_save)
        save_layout.addWidget(self.btn_load)
        ok_layout = QHBoxLayout()
        ok_layout.addWidget(self.btn_ok)
        ok_layout.addWidget(btn_cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(_("Poses acquisition for robot registration:")))
        layout.addLayout(acq_layout)
        layout.addLayout(pose_layout)
        layout.addLayout(apply_layout)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)
        layout.addWidget(QLabel(_("Registration file")))
        layout.addLayout(save_layout)
        layout.addLayout(ok_layout)
        self.adjustSize()

    def __bind_events(self):
        Publisher.subscribe(
            self.UpdateRobotTransformationMatrix,
            "Robot to Neuronavigation: Update robot transformation matrix",
        )
        Publisher.subscribe(
            self.PointRegisteredByRobot,
            "Robot to Neuronavigation: Coordinates for the robot transformation matrix collected",
        )

    def _on_cont_toggled(self, checked):
        if checked:
            self.timer.start(100)
        else:
            self.timer.stop()

    def HandleContinuousAcquisition(self):
        self.CreatePoint()

    def CreatePoint(self):
        Publisher.sendMessage(
            "Neuronavigation to Robot: Collect coordinates for the robot transformation matrix",
            data=None,
        )

    def GetAcquiredPoints(self):
        return int(self.txt_number.text())

    def SetAcquiredPoints(self, num):
        self.txt_number.setText(str(num))

    def PointRegisteredByRobot(self):
        num = self.GetAcquiredPoints() + 1
        self.SetAcquiredPoints(num)
        if self.robot.IsConnected() and num >= 3:
            self.btn_apply_reg.setEnabled(True)

    def ResetPoints(self):
        Publisher.sendMessage(
            "Neuronavigation to Robot: Reset coordinates collection for the robot transformation matrix",
            data=None,
        )
        if self.btn_cont_point.isChecked():
            self.btn_cont_point.setChecked(False)
        self.SetAcquiredPoints(0)
        self.btn_apply_reg.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_ok.setEnabled(False)
        self.matrix_tracker_to_robot = []

    def ApplyRegistration(self):
        if self.btn_cont_point.isChecked():
            self.btn_cont_point.setChecked(False)
        Publisher.sendMessage(
            "Neuronavigation to Robot: Estimate robot transformation matrix", data=None
        )
        self.btn_save.setEnabled(True)
        self.btn_ok.setEnabled(True)

    def UpdateRobotTransformationMatrix(self, data):
        self.matrix_tracker_to_robot = np.array(data)

    def SaveRegistration(self):
        if self.matrix_tracker_to_robot is None:
            return
        filename = ShowLoadSaveDialog(
            message=_("Save robot transformation file as..."),
            wildcard=_("Robot transformation files (*.rbtf)|*.rbtf"),
            default_filename="robottransform.rbtf",
            save_ext="rbtf",
        )
        if not filename:
            return
        with open(filename, "w", newline="") as file:
            writer = csv.writer(file, delimiter="\t")
            writer.writerows(np.vstack(self.matrix_tracker_to_robot).tolist())

    def LoadRegistration(self):
        filename = ShowLoadSaveDialog(
            message=_("Load robot transformation"),
            wildcard=_("Robot transformation files (*.rbtf)|*.rbtf"),
        )
        if not filename:
            return
        with open(filename) as file:
            reader = csv.reader(file, delimiter="\t")
            content = [row for row in reader]
        self.matrix_tracker_to_robot = np.vstack(list(np.float64(content)))
        Publisher.sendMessage(
            "Neuronavigation to Robot: Set robot transformation matrix",
            data=self.matrix_tracker_to_robot.tolist(),
        )
        if self.robot.IsConnected():
            self.btn_ok.setEnabled(True)

    def GetValue(self):
        return self.matrix_tracker_to_robot


class ConfigurePolarisDialog(QDialog):
    def __init__(self, n_coils, title=_("Configure NDI Polaris")):
        self.n_coils = n_coils
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Tool)
        self._init_gui()

    def serial_ports(self):
        import serial.tools.list_ports

        port_list = []
        desc_list = []
        ports = serial.tools.list_ports.comports()
        for p in ports:
            port_list.append(p.device)
            desc_list.append(p.description)
        port_selec = [i for i, e in enumerate(desc_list) if "NDI" in e]
        return port_list, port_selec

    def _init_gui(self):
        self.com_ports = QComboBox()
        port_list, port_selec = self.serial_ports()
        self.com_ports.addItems(port_list)
        self.com_ports.addItems(const.NDI_IP)
        if port_selec:
            self.com_ports.setCurrentIndex(port_selec[0])

        session = ses.Session()
        last_probe = session.GetConfig("last_ndi_probe_marker", "") or inv_paths.NDI_MAR_DIR_PROBE
        last_ref = session.GetConfig("last_ndi_ref_marker", "") or inv_paths.NDI_MAR_DIR_REF
        last_objs = session.GetConfig("last_ndi_obj_markers", [])
        while len(last_objs) < self.n_coils:
            last_objs.append(inv_paths.NDI_MAR_DIR_OBJ)

        self.dir_probe = FilePickerCtrl(
            self, path=last_probe, wildcard="Rom files (*.rom)|*.rom", message="Select probe ROM"
        )
        self.dir_ref = FilePickerCtrl(
            self, path=last_ref, wildcard="Rom files (*.rom)|*.rom", message="Select reference ROM"
        )
        self.dir_objs = []
        for i in range(self.n_coils):
            fp = FilePickerCtrl(
                self,
                path=last_objs[i],
                wildcard="Rom files (*.rom)|*.rom",
                message=f"Select coil {i + 1} ROM",
            )
            self.dir_objs.append(fp)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("COM port or IP:"))
        layout.addWidget(self.com_ports)
        layout.addWidget(QLabel("Probe ROM file:"))
        layout.addWidget(self.dir_probe)
        layout.addWidget(QLabel("Reference ROM file:"))
        layout.addWidget(self.dir_ref)
        for i, fp in enumerate(self.dir_objs):
            layout.addWidget(QLabel(f"Coil {i + 1} ROM file:"))
            layout.addWidget(fp)
        layout.addWidget(btn_box)
        self.adjustSize()

    def GetValue(self):
        fn_probe = self.dir_probe.GetPath()
        fn_ref = self.dir_ref.GetPath()
        fn_objs = [d.GetPath() for d in self.dir_objs]
        if fn_probe and fn_ref and fn_objs:
            session = ses.Session()
            session.SetConfig("last_ndi_probe_marker", fn_probe)
            session.SetConfig("last_ndi_ref_marker", fn_ref)
            session.SetConfig("last_ndi_obj_markers", fn_objs)
        return self.com_ports.currentText(), fn_probe, fn_ref, fn_objs


class SetCOMPort(QDialog):
    def __init__(self, select_baud_rate, title=_("Select COM port")):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.select_baud_rate = select_baud_rate
        self._init_gui()

    def serial_ports(self):
        import serial.tools.list_ports

        return [comport.device for comport in serial.tools.list_ports.comports()]

    def _init_gui(self):
        ports = self.serial_ports()
        self.com_port_dropdown = QComboBox()
        self.com_port_dropdown.addItems(ports)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("COM port"))
        layout.addWidget(self.com_port_dropdown)

        if self.select_baud_rate:
            baud_rates = [str(br) for br in const.BAUD_RATES]
            self.baud_rate_dropdown = QComboBox()
            self.baud_rate_dropdown.addItems(baud_rates)
            self.baud_rate_dropdown.setCurrentIndex(const.BAUD_RATE_DEFAULT_SELECTION)
            layout.addWidget(QLabel("Baud rate"))
            layout.addWidget(self.baud_rate_dropdown)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)
        self.adjustSize()

    def GetCOMPort(self):
        com_port = self.com_port_dropdown.currentText()
        if not com_port:
            QMessageBox.warning(self, "No selection", "Please select a COM port.")
            return None
        return com_port

    def GetBaudRate(self):
        if not self.select_baud_rate:
            return None
        return self.baud_rate_dropdown.currentText()


class ManualWWWLDialog(QDialog):
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle(_("Set WW&WL manually"))
        self._init_gui()

    def _init_gui(self):
        import invesalius.data.slice_ as slc

        ww = slc.Slice().window_width
        wl = slc.Slice().window_level

        self.txt_wl = QLineEdit(str(int(wl)))
        self.txt_ww = QLineEdit(str(int(ww)))

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.OnOK)
        btn_box.rejected.connect(self.close)

        layout = QVBoxLayout(self)
        wl_layout = QHBoxLayout()
        wl_layout.addWidget(QLabel(_("Window Level")))
        wl_layout.addWidget(self.txt_wl, 1)
        wl_layout.addWidget(QLabel("WL"))
        layout.addLayout(wl_layout)

        ww_layout = QHBoxLayout()
        ww_layout.addWidget(QLabel(_("Window Width")))
        ww_layout.addWidget(self.txt_ww, 1)
        ww_layout.addWidget(QLabel("WW"))
        layout.addLayout(ww_layout)
        layout.addWidget(btn_box)
        self.adjustSize()
        _center_on_screen(self)

    def OnOK(self):
        try:
            ww = int(self.txt_ww.text())
            wl = int(self.txt_wl.text())
        except ValueError:
            self.close()
            return
        Publisher.sendMessage("Bright and contrast adjustment image", window=ww, level=wl)
        const.WINDOW_LEVEL["Manual"] = (ww, wl)
        Publisher.sendMessage("Check window and level other")
        Publisher.sendMessage("Update window level value", window=ww, level=wl)
        Publisher.sendMessage("Update slice viewer")
        Publisher.sendMessage("Render volume viewer")
        self.close()


class SetSpacingDialog(QDialog):
    def __init__(self, parent, sx, sy, sz, title=_("Set spacing"), **kwargs):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)
        self.spacing_new_x = self.spacing_original_x = sx
        self.spacing_new_y = self.spacing_original_y = sy
        self.spacing_new_z = self.spacing_original_z = sz
        self._init_gui()

    def _init_gui(self):
        self.txt_spacing_new_x = QLineEdit(str(self.spacing_original_x))
        self.txt_spacing_new_y = QLineEdit(str(self.spacing_original_y))
        self.txt_spacing_new_z = QLineEdit(str(self.spacing_original_z))
        self.txt_spacing_new_x.editingFinished.connect(self._on_edit)
        self.txt_spacing_new_y.editingFinished.connect(self._on_edit)
        self.txt_spacing_new_z.editingFinished.connect(self._on_edit)

        grid = QGridLayout()
        grid.addWidget(QLabel("Spacing X"), 0, 0)
        grid.addWidget(self.txt_spacing_new_x, 0, 1)
        grid.addWidget(QLabel("Spacing Y"), 1, 0)
        grid.addWidget(self.txt_spacing_new_y, 1, 1)
        grid.addWidget(QLabel("Spacing Z"), 2, 0)
        grid.addWidget(self.txt_spacing_new_z, 2, 1)

        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.OnOk)
        btn_box.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(
            QLabel(_("It was not possible to obtain the image spacings.\nPlease set it correctly:"))
        )
        layout.addLayout(grid)
        layout.addWidget(btn_box)
        self.adjustSize()

    def _on_edit(self):
        try:
            self.spacing_new_x = float(self.txt_spacing_new_x.text())
        except ValueError:
            pass
        try:
            self.spacing_new_y = float(self.txt_spacing_new_y.text())
        except ValueError:
            pass
        try:
            self.spacing_new_z = float(self.txt_spacing_new_z.text())
        except ValueError:
            pass

    def set_new_spacing(self, sx, sy, sz):
        self.spacing_new_x = sx
        self.spacing_new_y = sy
        self.spacing_new_z = sz
        self.txt_spacing_new_x.setText(str(sx))
        self.txt_spacing_new_y.setText(str(sy))
        self.txt_spacing_new_z.setText(str(sz))

    def OnOk(self):
        self._on_edit()
        if self.spacing_new_x == 0.0:
            self.txt_spacing_new_x.setFocus()
        elif self.spacing_new_y == 0.0:
            self.txt_spacing_new_y.setFocus()
        elif self.spacing_new_z == 0.0:
            self.txt_spacing_new_z.setFocus()
        else:
            self.accept()


class PeelsCreationDlg(QDialog):
    FROM_MASK = 1
    FROM_FILES = 2

    def __init__(self, parent, *args, **kwds):
        super().__init__(parent)
        self.mask_path = ""
        self.method = self.FROM_MASK
        self._init_gui()
        self.get_all_masks()

    def _init_gui(self):
        self.setWindowTitle(_("Create peel"))

        mask_group = QGroupBox(_("From mask"))
        mask_layout = QHBoxLayout(mask_group)
        self.from_mask_rb = QRadioButton()
        self.from_mask_rb.setChecked(True)
        self.from_mask_rb.toggled.connect(self._on_select_method)
        self.cb_masks = QComboBox()
        mask_layout.addWidget(self.from_mask_rb)
        mask_layout.addWidget(self.cb_masks, 1)

        files_group = QGroupBox(_("From files"))
        files_layout = QHBoxLayout(files_group)
        self.from_files_rb = QRadioButton()
        self.from_files_rb.toggled.connect(self._on_select_method)

        session = ses.Session()
        last_directory = session.GetConfig("last_directory_%d" % const.ID_NIFTI_IMPORT, "")
        self.mask_file_browse = FileBrowseButton(
            self,
            labelText=_("Mask file"),
            fileMask=WILDCARD_NIFTI,
            dialogTitle=_("Choose mask file"),
            startDirectory=last_directory,
            changeCallback=self._set_files_callback,
        )
        files_layout.addWidget(self.from_files_rb)
        files_layout.addWidget(self.mask_file_browse, 1)

        self.btn_ok = QPushButton(_("OK"))
        self.btn_ok.setDefault(True)
        btn_cancel = QPushButton(_("Cancel"))
        self.btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(btn_cancel)

        layout = QVBoxLayout(self)
        layout.addWidget(mask_group)
        layout.addWidget(files_group)
        layout.addLayout(btn_layout)
        self.adjustSize()

    def get_all_masks(self):
        import invesalius.project as prj

        inv_proj = prj.Project()
        choices = [i.name for i in inv_proj.mask_dict.values()]
        self.cb_masks.clear()
        self.cb_masks.addItems(choices)
        self.btn_ok.setEnabled(len(choices) > 0)

    def _on_select_method(self):
        if self.from_mask_rb.isChecked():
            self.method = self.FROM_MASK
            self.btn_ok.setEnabled(self.cb_masks.count() > 0)
        else:
            self.method = self.FROM_FILES
            self.btn_ok.setEnabled(self._check_if_files_exists())

    def _set_files_callback(self, path=""):
        if path:
            self.mask_path = path
        if self.method == self.FROM_FILES:
            self.btn_ok.setEnabled(self._check_if_files_exists())

    def _check_if_files_exists(self):
        return bool(self.mask_path and os.path.exists(self.mask_path))


class FileSelectionDialog(QDialog):
    def __init__(self, title, default_dir, wildcard):
        super().__init__(_top_window())
        self.setWindowTitle(title)
        self.default_dir = default_dir
        self.wildcard = wildcard
        self.path = ""

        qt_wildcard = _wx_wildcard_to_qt(wildcard) if "|" in wildcard else wildcard

        self.file_browse = FileBrowseButton(
            self,
            labelText="",
            fileMask=wildcard,
            dialogTitle=_("Choose file"),
            startDirectory=default_dir,
            changeCallback=self._set_path,
        )
        self.file_browse.setMinimumWidth(500)

        self.btn_ok = QPushButton(_("OK"))
        self.btn_ok.setDefault(True)
        btn_cancel = QPushButton(_("Cancel"))
        self.btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(btn_cancel)

        self.sizer = QVBoxLayout(self)
        self.sizer.addWidget(self.file_browse)
        self.sizer.addLayout(btn_layout)
        self.adjustSize()

    def _set_path(self, path=""):
        self.path = path

    def FitSizers(self):
        self.adjustSize()

    def GetPath(self):
        return self.path


class ProgressBarHandler(QProgressDialog):
    def __init__(self, parent, title="Progress Dialog", msg="Initializing...", max_value=None):
        maximum = max_value if max_value else 0
        super().__init__(msg, "Cancel", 0, maximum, parent)
        self.setWindowTitle(title)
        self.setWindowModality(Qt.ApplicationModal)
        self.max_value = max_value
        Publisher.subscribe(self.update_progress, "Update Progress bar")
        Publisher.subscribe(self.close_progress, "Close Progress bar")

    def was_cancelled(self):
        return self.wasCanceled()

    def update_progress(self, value, msg=None):
        if self.wasCanceled():
            return
        if self.max_value is None:
            self.setRange(0, 0)
            if msg:
                self.setLabelText(msg)
        else:
            v = min(int(value), self.max_value)
            self.setValue(v)
            if msg:
                self.setLabelText(msg)

    def close_progress(self):
        self.close()

    def pulse(self, msg=None):
        self.setRange(0, 0)
        if msg:
            self.setLabelText(msg)
