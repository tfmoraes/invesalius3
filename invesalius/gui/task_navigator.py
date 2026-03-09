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
import itertools
import os
import time
from functools import partial
from typing import Optional

import numpy as np

try:
    # TODO: the try-except could be done inside the mTMS() method call
    from invesalius.navigation.mtms import mTMS

    mTMS()
    has_mTMS = True
except Exception:
    has_mTMS = False

import sys
import uuid

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QToolBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.data.coordinates as dco
import invesalius.data.slice_ as slice_
import invesalius.gui.dialogs as dlg
import invesalius.gui.widgets.gradient as grad
import invesalius.project as prj
import invesalius.session as ses
from invesalius import inv_paths, utils
from invesalius.data.markers.marker import Marker, MarkerType
from invesalius.gui import deep_learning_seg_dialog
from invesalius.gui.default_viewers import ColourSelectButton
from invesalius.gui.widgets.fiducial_buttons import OrderedFiducialButtons
from invesalius.i18n import tr as _
from invesalius.navigation.navigation import NavigationHub
from invesalius.navigation.robot import RobotObjective
from invesalius.pubsub import pub as Publisher

BTN_NEW = 0
BTN_IMPORT_LOCAL = 0


def GetBitMapForBackground():
    image_file = os.path.join("head.png")
    pixmap = QPixmap(str(inv_paths.ICON_DIR.joinpath(image_file)))
    return pixmap


class TaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        inner_panel = InnerTaskPanel(self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(7, 0, 7, 7)
        layout.addWidget(inner_panel, 1)


class InnerTaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet("background-color: white;")

        fold_panel = FoldPanel(self)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 0, 5, 0)
        main_layout.addWidget(fold_panel, 1)
        main_layout.addSpacing(5)


class FoldPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        inner_panel = InnerFoldPanel(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(inner_panel)


class InnerFoldPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.__bind_events()

        nav_hub = NavigationHub(window=self)

        self.nav_hub = nav_hub
        self.tracker = nav_hub.tracker
        self.image = nav_hub.image
        self.navigation = nav_hub.navigation
        self.mep_visualizer = nav_hub.mep_visualizer

        self.toolbox = QToolBox(self)

        coreg_panel = CoregistrationPanel(parent=self.toolbox, nav_hub=nav_hub)
        self.toolbox.addItem(coreg_panel, _("Coregistration"))

        nav_panel = NavigationPanel(parent=self.toolbox, nav_hub=nav_hub)
        self.toolbox.addItem(nav_panel, _("Navigation"))
        self.nav_panel_index = 1

        self.toolbox.setCurrentIndex(0)
        self.toolbox.currentChanged.connect(self.OnFoldPressCaption)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbox)

        self.__bind_events()

    def __bind_events(self):
        Publisher.subscribe(self.OpenNavigation, "Open navigation menu")
        Publisher.subscribe(self.OnEnableState, "Enable state project")
        Publisher.subscribe(self.CollapseNavigation, "Coil selection done")

    def OnEnableState(self, state):
        if not state:
            self.toolbox.setCurrentIndex(0)

    def OnShowDbs(self):
        pass

    def OnHideDbs(self):
        pass

    def OnCheckStatus(self, nav_status, vis_status):
        if nav_status:
            self.checkbox_serial_port.setEnabled(False)
        else:
            self.checkbox_serial_port.setEnabled(True)

    def OnEnableSerialPort(self, evt, ctrl):
        if ctrl.isChecked():
            dlg_port = dlg.SetCOMPort(select_baud_rate=False)

            if dlg_port.exec() != QDialog.Accepted:
                ctrl.setChecked(False)
                return

            com_port = dlg_port.GetCOMPort()
            if not com_port:
                ctrl.setChecked(False)
                return

            baud_rate = 115200

            Publisher.sendMessage(
                "Update serial port",
                serial_port_in_use=True,
                com_port=com_port,
                baud_rate=baud_rate,
            )
        else:
            Publisher.sendMessage("Update serial port", serial_port_in_use=False)

    def PressShowCoilButton(self, pressed=False):
        self.show_coil_button.setChecked(pressed)
        self.OnShowCoil()

    def EnableShowCoilButton(self, enabled=False):
        self.show_coil_button.setEnabled(enabled)

    def OnShowCoil(self, evt=None):
        pressed = self.show_coil_button.isChecked()
        Publisher.sendMessage("Show coil in viewer volume", state=pressed, coil_name=None)

    def CollapseNavigation(self, done):
        if not done:
            self.toolbox.setCurrentIndex(0)

    def OnFoldPressCaption(self, index):
        pass

    def ResizeFPB(self):
        pass

    def CheckRegistration(self):
        return (
            self.tracker.AreTrackerFiducialsSet()
            and self.image.AreImageFiducialsSet()
            and self.navigation.CoilSelectionDone()
        )

    def OpenNavigation(self):
        self.toolbox.setCurrentIndex(self.nav_panel_index)


class CoregistrationPanel(QWidget):
    def __init__(self, parent, nav_hub):
        super().__init__(parent)
        self.setStyleSheet("background-color: white;")

        book = QTabWidget(self)
        book.currentChanged.connect(self.OnPageChanged)

        self.nav_hub = nav_hub
        self.tracker = nav_hub.tracker
        self.image = nav_hub.image

        self._current_page = 0

        book.addTab(ImportsPage(book, nav_hub), _("Imports"))
        book.addTab(HeadPage(book, nav_hub), _("Head"))
        book.addTab(ImagePage(book, nav_hub), _("Image"))
        book.addTab(TrackerPage(book, nav_hub), _("Patient"))
        book.addTab(RefinePage(book, nav_hub), _("Refine"))
        book.addTab(StylusPage(book, nav_hub), _("Stylus"))
        book.addTab(StimulatorPage(book, nav_hub), _("TMS Coil"))

        session = ses.Session()
        project_status = session.GetConfig("project_status")

        if project_status == const.PROJECT_STATUS_OPENED:
            book.setCurrentIndex(const.HEAD_PAGE)
            self._current_page = const.HEAD_PAGE
        else:
            book.setCurrentIndex(const.IMPORTS_PAGE)
            self._current_page = const.IMPORTS_PAGE

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(book)

        self.book = book
        self.__bind_events()

    def __bind_events(self):
        Publisher.subscribe(self._FoldImports, "Move to imports page")
        Publisher.subscribe(self._FoldHead, "Move to head model page")
        Publisher.subscribe(self._FoldTracker, "Move to tracker page")
        Publisher.subscribe(self._FoldRefine, "Move to refine page")
        Publisher.subscribe(self._FoldStylus, "Move to stylus page")
        Publisher.subscribe(self._FoldStimulator, "Move to stimulator page")
        Publisher.subscribe(self._FoldImage, "Move to image page")
        Publisher.subscribe(self.OnCloseProject, "Close project data")

    def OnCloseProject(self):
        self.book.setCurrentIndex(const.IMPORTS_PAGE)

    def OnPageChanged(self, new_page):
        old_page = self._current_page
        self._current_page = new_page

        session = ses.Session()
        project_status = session.GetConfig("project_status")
        if (
            old_page == const.IMPORTS_PAGE
            and project_status == const.PROJECT_STATUS_CLOSED
            and new_page != const.IMPORTS_PAGE
        ):
            self.book.setCurrentIndex(const.IMPORTS_PAGE)
            from invesalius.error_handling import show_warning

            show_warning(_("InVesalius 3"), _("Please import image first."))
            return

        if old_page <= const.IMAGE_PAGE and new_page > const.IMAGE_PAGE:
            if not self.image.AreImageFiducialsSet():
                self.book.setCurrentIndex(const.IMAGE_PAGE)
                from invesalius.error_handling import show_warning

                show_warning(_("InVesalius 3"), _("Please do the image registration first."))
        if old_page != const.REFINE_PAGE:
            Publisher.sendMessage("Update UI for refine tab")

        if (old_page == const.TRACKER_PAGE) and (new_page > const.TRACKER_PAGE):
            if self.image.AreImageFiducialsSet() and not self.tracker.AreTrackerFiducialsSet():
                self.book.setCurrentIndex(const.TRACKER_PAGE)
                from invesalius.error_handling import show_warning

                show_warning(_("InVesalius 3"), _("Please do the tracker registration first."))

    def _FoldImports(self):
        self.book.setCurrentIndex(const.IMPORTS_PAGE)

    def _FoldHead(self):
        self.book.setCurrentIndex(const.HEAD_PAGE)

    def _FoldImage(self):
        self.book.setCurrentIndex(const.IMAGE_PAGE)

    def _FoldTracker(self):
        Publisher.sendMessage("Disable style", style=const.SLICE_STATE_CROSS)
        self.book.setCurrentIndex(const.TRACKER_PAGE)

    def _FoldRefine(self):
        self.book.setCurrentIndex(const.REFINE_PAGE)

    def _FoldStylus(self):
        self.book.setCurrentIndex(const.STYLUS_PAGE)

    def _FoldStimulator(self):
        self.book.setCurrentIndex(const.STIMULATOR_PAGE)


class ImportsPage(QWidget):
    def __init__(self, parent, nav_hub):
        super().__init__(parent)
        self.setStyleSheet("background-color: white;")

        self.navigation = nav_hub
        self.proj_count = 0

        self.top_layout = QVBoxLayout()
        self.bottom_layout = QVBoxLayout()
        self.main_layout = QVBoxLayout(self)

        tooltip = _("Select DICOM files to be reconstructed")
        link_import_local = QPushButton(_("Import DICOM images..."), self)
        link_import_local.setFlat(True)
        link_import_local.setCursor(QCursor(Qt.PointingHandCursor))
        link_import_local.setStyleSheet("text-align: left; font-weight: bold; color: black;")
        link_import_local.setToolTip(tooltip)
        link_import_local.clicked.connect(self.OnLinkImport)

        tooltip = _("Select NIFTI files to be reconstructed")
        link_import_nifti = QPushButton(_("Import NIFTI images..."), self)
        link_import_nifti.setFlat(True)
        link_import_nifti.setCursor(QCursor(Qt.PointingHandCursor))
        link_import_nifti.setStyleSheet("text-align: left; font-weight: bold; color: black;")
        link_import_nifti.setToolTip(tooltip)
        link_import_nifti.clicked.connect(self.OnLinkImportNifti)

        tooltip = _("Open an existing InVesalius project...")
        link_open_proj = QPushButton(_("Open an existing project..."), self)
        link_open_proj.setFlat(True)
        link_open_proj.setCursor(QCursor(Qt.PointingHandCursor))
        link_open_proj.setStyleSheet("text-align: left; font-weight: bold; color: black;")
        link_open_proj.setToolTip(tooltip)
        link_open_proj.clicked.connect(self.OnLinkOpenProject)

        BMP_IMPORT = QPixmap(str(inv_paths.ICON_DIR.joinpath("file_import_original.png")))
        BMP_OPEN_PROJECT = QPixmap(str(inv_paths.ICON_DIR.joinpath("file_open_original.png")))

        button_import_local = QPushButton(self)
        button_import_local.setIcon(QIcon(BMP_IMPORT))
        button_import_local.setFlat(True)
        button_import_local.clicked.connect(self.ImportDicom)

        button_import_nifti = QPushButton(self)
        button_import_nifti.setIcon(QIcon(BMP_IMPORT))
        button_import_nifti.setFlat(True)
        button_import_nifti.clicked.connect(self.ImportNifti)

        button_open_proj = QPushButton(self)
        button_open_proj.setIcon(QIcon(BMP_OPEN_PROJECT))
        button_open_proj.setFlat(True)
        button_open_proj.clicked.connect(lambda: self.OpenProject())

        next_button = QPushButton("Next", self)
        next_button.clicked.connect(lambda: Publisher.sendMessage("Move to head model page"))
        self.bottom_layout.addWidget(next_button, 0, Qt.AlignRight)

        fixed_layout = QGridLayout()
        fixed_layout.addWidget(link_import_local, 0, 0)
        fixed_layout.addWidget(button_import_local, 0, 1)
        fixed_layout.addWidget(link_import_nifti, 1, 0)
        fixed_layout.addWidget(button_import_nifti, 1, 1)
        fixed_layout.addWidget(link_open_proj, 2, 0)
        fixed_layout.addWidget(button_open_proj, 2, 1)
        fixed_layout.setColumnStretch(0, 1)

        self.top_layout.addLayout(fixed_layout)
        self.main_layout.addLayout(self.top_layout)
        self.main_layout.addStretch(1)
        self.main_layout.addLayout(self.bottom_layout)

        self.LoadRecentProjects()

    def OnLinkOpenProject(self):
        self.OpenProject()

    def OpenProject(self, path=None):
        if path:
            Publisher.sendMessage("Open recent project", filepath=path)
        else:
            Publisher.sendMessage("Show open project dialog")
        self.OnMoveToHeadModelPage()

    def OnLinkImport(self):
        self.ImportDicom()

    def ImportDicom(self):
        Publisher.sendMessage("Show import directory dialog")
        self.OnMoveToHeadModelPage()

    def OnLinkImportNifti(self):
        self.ImportNifti()

    def ImportNifti(self):
        Publisher.sendMessage("Show import other files dialog", id_type=const.ID_NIFTI_IMPORT)
        self.OnMoveToHeadModelPage()

    def OnMoveToHeadModelPage(self):
        session = ses.Session()
        project_status = session.GetConfig("project_status")
        if project_status != const.PROJECT_STATUS_CLOSED:
            Publisher.sendMessage("Move to head model page")

    def LoadRecentProjects(self):
        import invesalius.session as ses

        session = ses.Session()
        recent_projects = session.GetConfig("recent_projects")

        for path, filename in recent_projects:
            self.LoadProject(filename, path)

    def LoadProject(self, proj_name="Unnamed", proj_dir=""):
        proj_path = os.path.join(proj_dir, proj_name)

        if self.proj_count < 3:
            self.proj_count += 1

            label = "     " + str(self.proj_count) + ". " + proj_name

            proj_link = QPushButton(label, self)
            proj_link.setFlat(True)
            proj_link.setCursor(QCursor(Qt.PointingHandCursor))
            proj_link.setStyleSheet("text-align: left; color: black;")
            proj_link.clicked.connect(lambda checked, p=proj_path: self.OpenProject(p))

            self.top_layout.addWidget(proj_link)


class HeadPage(QWidget):
    def __init__(self, parent, nav_hub):
        super().__init__(parent)

        top_layout = QVBoxLayout()
        bottom_layout = QHBoxLayout()
        main_layout = QVBoxLayout(self)

        label_combo = QLabel("Mask selection", self)
        label_combo.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(label_combo)
        main_layout.addSpacing(10)

        self.combo_mask = QComboBox(self)
        top_layout.addWidget(self.combo_mask)

        label_thresh = QLabel("Threshold", self)
        label_thresh.setAlignment(Qt.AlignCenter)
        top_layout.addSpacing(10)
        top_layout.addWidget(label_thresh)

        gradient = grad.GradientCtrl(self, -1, -5000, 5000, 0, 5000, (0, 255, 0, 100))
        self.gradient = gradient
        top_layout.addWidget(self.gradient)

        self.select_largest_surface_checkbox = QCheckBox("Select largest surface", self)
        top_layout.addStretch(1)
        top_layout.addWidget(self.select_largest_surface_checkbox)
        top_layout.addSpacing(5)
        self.select_largest_surface_checkbox.setChecked(True)

        self.remove_non_visible_checkbox = QCheckBox("Remove non-visible faces", self)
        top_layout.addWidget(self.remove_non_visible_checkbox)
        top_layout.addSpacing(5)
        self.remove_non_visible_checkbox.setChecked(True)

        self.smooth_surface_checkbox = QCheckBox("Smooth scalp surface", self)
        top_layout.addWidget(self.smooth_surface_checkbox)
        top_layout.addSpacing(5)
        self.smooth_surface_checkbox.setChecked(True)

        self.brain_segmentation_checkbox = QCheckBox("Brain segmentation (~ a few minutes)", self)
        top_layout.addWidget(self.brain_segmentation_checkbox)

        create_head_button = QPushButton("Create head surface", self)
        create_head_button.clicked.connect(self.OnCreateHeadSurface)
        top_layout.addStretch(1)
        top_layout.addWidget(create_head_button, 0, Qt.AlignCenter)

        back_button = QPushButton("Back", self)
        back_button.clicked.connect(self.OnBack)
        bottom_layout.addWidget(back_button)
        bottom_layout.addStretch(1)
        next_button = QPushButton("Next", self)
        next_button.clicked.connect(self.OnNext)
        bottom_layout.addWidget(next_button)

        main_layout.addLayout(top_layout)
        main_layout.addStretch(1)
        main_layout.addLayout(bottom_layout)

        self.__bind_events()
        self.__bind_events_qt()

    def OnNext(self):
        Publisher.sendMessage("Move to image page")

    def OnBack(self):
        Publisher.sendMessage("Move to imports page")

    def __bind_events(self):
        Publisher.subscribe(self.OnSuccessfulBrainSegmentation, "Brain segmentation completed")
        Publisher.subscribe(self.SetThresholdBounds, "Update threshold limits")
        Publisher.subscribe(self.SetThresholdValues, "Set threshold values in gradient")
        Publisher.subscribe(self.SetThresholdValues2, "Set threshold values")
        Publisher.subscribe(self.SelectMaskName, "Select mask name in combo")
        Publisher.subscribe(self.SetItemsColour, "Set GUI items colour")
        Publisher.subscribe(self.OnRemoveMasks, "Remove masks")
        Publisher.subscribe(self.AddMask, "Add mask")
        Publisher.subscribe(self.OnCloseProject, "Close project data")

    def OnCloseProject(self):
        self.OnRemoveMasks(list(reversed(range(self.combo_mask.count()))))

    def __bind_events_qt(self):
        self.combo_mask.currentIndexChanged.connect(self.OnComboName)
        self.gradient.threshold_changed.connect(self.OnSlideChanged)
        self.gradient.threshold_changing.connect(self.OnSlideChanging)

    def OnComboName(self, mask_index):
        Publisher.sendMessage("Change mask selected", index=mask_index)
        Publisher.sendMessage("Show mask", index=mask_index, value=True)

    def AddMask(self, mask):
        self.combo_mask.addItem(mask.name)

    def SelectMaskName(self, index):
        if index >= 0:
            self.combo_mask.setCurrentIndex(index)
        else:
            self.combo_mask.setCurrentText("")

    def OnRemoveMasks(self, mask_indexes):
        self.combo_mask.setUpdatesEnabled(False)
        try:
            count = self.combo_mask.count()
            idxs = sorted(set(mask_indexes), reverse=True)

            if len(idxs) >= count:
                self.combo_mask.clear()
                return

            for i in idxs:
                if 0 <= i < self.combo_mask.count():
                    self.combo_mask.removeItem(i)
        finally:
            self.combo_mask.setUpdatesEnabled(True)

    def SetThresholdBounds(self, threshold_range):
        thresh_min = threshold_range[0]
        thresh_max = threshold_range[1]
        self.gradient.SetMinRange(thresh_min)
        self.gradient.SetMaxRange(thresh_max)

    def SetThresholdValues(self, threshold_range):
        thresh_min, thresh_max = threshold_range
        self.gradient.SetMinValue(thresh_min)
        self.gradient.SetMaxValue(thresh_max)

    def SetThresholdValues2(self, threshold_range):
        thresh_min, thresh_max = threshold_range
        self.gradient.SetMinValue(thresh_min)
        self.gradient.SetMaxValue(thresh_max)

    def OnSlideChanged(self, evt):
        thresh_min = self.gradient.GetMinValue()
        thresh_max = self.gradient.GetMaxValue()
        Publisher.sendMessage("Set threshold values", threshold_range=(thresh_min, thresh_max))
        session = ses.Session()
        session.ChangeProject()

    def OnSlideChanging(self, evt):
        thresh_min = self.gradient.GetMinValue()
        thresh_max = self.gradient.GetMaxValue()
        Publisher.sendMessage("Changing threshold values", threshold_range=(thresh_min, thresh_max))
        session = ses.Session()
        session.ChangeProject()

    def SetItemsColour(self, colour):
        self.gradient.SetColour(colour)

    def OnCreateHeadSurface(self):
        if not self.CreateSurface():
            return

        if self.select_largest_surface_checkbox.isChecked():
            self.SelectLargestSurface()

        if self.remove_non_visible_checkbox.isChecked():
            self.RemoveNonVisibleFaces()

        if self.smooth_surface_checkbox.isChecked():
            self.SmoothSurface()

        self.VisualizeScalpSurface()

        if self.brain_segmentation_checkbox.isChecked():
            self.SegmentBrain()

        Publisher.sendMessage("Move to image page")

    def CreateBrainSurface(self):
        options = {"angle": 0.7, "max distance": 3.0, "min weight": 0.5, "steps": 10}
        algorithm = "ca_smoothing"
        proj = prj.Project()
        mask_index = len(proj.mask_dict) - 1
        brain_colour = [235, 245, 255]

        if self.combo_mask.currentIndex() != -1:
            sl = slice_.Slice()
            for idx in proj.mask_dict:
                if proj.mask_dict[idx] is sl.current_mask:
                    mask_index = idx
                    break

            method = {"algorithm": algorithm, "options": options}
            srf_options = {
                "index": mask_index,
                "name": "Brain",
                "quality": _("Optimal *"),
                "fill": False,
                "keep_largest": True,
                "overwrite": False,
            }
            Publisher.sendMessage(
                "Create surface from index",
                surface_parameters={"method": method, "options": srf_options},
            )
            Publisher.sendMessage("Fold surface task")

            surface_idx = len(proj.surface_dict) - 1
            brain_vtk_colour = [c / 255.0 for c in brain_colour]

            Publisher.sendMessage(
                "Set surface colour", surface_index=surface_idx, colour=brain_vtk_colour
            )
            Publisher.sendMessage("Change surface selected", surface_index=surface_idx)

            last_two = list(range(len(proj.surface_dict) - 2, len(proj.surface_dict)))
            Publisher.sendMessage("Show multiple surfaces", index_list=last_two, visibility=True)

        else:
            dlg.InexistentMask()

    def CreateSurface(self):
        algorithm = "Default"
        options = {}
        to_generate = True
        if self.combo_mask.currentIndex() != -1:
            sl = slice_.Slice()
            if sl.current_mask.was_edited:
                surface_dlg = dlg.SurfaceDialog()
                if surface_dlg.exec() == QDialog.Accepted:
                    algorithm = surface_dlg.GetAlgorithmSelected()
                    options = surface_dlg.GetOptions()
                else:
                    to_generate = False
                surface_dlg.close()
            if to_generate:
                proj = prj.Project()
                for idx in proj.mask_dict:
                    if proj.mask_dict[idx] is sl.current_mask:
                        mask_index = idx
                        break
                else:
                    return False
                method = {"algorithm": algorithm, "options": options}
                srf_options = {
                    "index": mask_index,
                    "name": "Scalp",
                    "quality": _("Optimal *"),
                    "fill": True,
                    "keep_largest": False,
                    "overwrite": False,
                }
                Publisher.sendMessage(
                    "Create surface from index",
                    surface_parameters={"method": method, "options": srf_options},
                )
                Publisher.sendMessage("Fold surface task")
            return True
        else:
            dlg.InexistentMask()
            return False

    def SelectLargestSurface(self):
        Publisher.sendMessage("Create surface from largest region", overwrite=True, name="Scalp")

    def RemoveNonVisibleFaces(self):
        Publisher.sendMessage("Remove non-visible faces")

    def SmoothSurface(self):
        Publisher.sendMessage("Create smooth surface", overwrite=True, name="Scalp")

    def VisualizeScalpSurface(self):
        proj = prj.Project()
        surface_idx = len(proj.surface_dict) - 1
        scalp_colour = [255, 235, 255]
        transparency = 0.25
        scalp_vtk_colour = [c / 255.0 for c in scalp_colour]

        Publisher.sendMessage(
            "Set surface colour", surface_index=surface_idx, colour=scalp_vtk_colour
        )
        Publisher.sendMessage(
            "Set surface transparency", surface_index=surface_idx, transparency=transparency
        )
        Publisher.sendMessage("Change surface selected", surface_index=surface_idx)
        Publisher.sendMessage("Show single surface", index=surface_idx, visibility=True)

    def OnSuccessfulBrainSegmentation(self):
        self.CreateBrainSurface()

    def SegmentBrain(self):
        if deep_learning_seg_dialog.HAS_TORCH:
            segmentation_dlg = deep_learning_seg_dialog.BrainSegmenterDialog(
                self, auto_segment=True
            )
            segmentation_dlg.show()
        else:
            QMessageBox.information(
                self,
                "InVesalius 3 - Brain segmenter",
                _(
                    "It's not possible to run brain segmenter because your system doesn't have the following modules installed:"
                )
                + " Torch",
            )


class ImagePage(QWidget):
    def __init__(self, parent, nav_hub):
        super().__init__(parent)

        self.image = nav_hub.image
        self.btns_set_fiducial = [None, None, None]
        self.numctrls_fiducial = [[], [], []]
        self.current_coord = 0, 0, 0, None, None, None

        self.bg_bmp = GetBitMapForBackground()
        background = QLabel(self)
        background.setPixmap(self.bg_bmp)

        for n, fiducial in enumerate(const.IMAGE_FIDUCIALS):
            label = fiducial["label"]
            tip = fiducial["tip"]

            ctrl = QPushButton(label, self)
            ctrl.setCheckable(True)
            ctrl.setToolTip(tip)
            ctrl.clicked.connect(partial(self.OnImageFiducials, n))
            ctrl.setEnabled(False)

            self.btns_set_fiducial[n] = ctrl

        for m in range(len(self.btns_set_fiducial)):
            for n in range(3):
                spinbox = QDoubleSpinBox(self)
                spinbox.setRange(-9999.9, 9999.9)
                spinbox.setDecimals(1)
                spinbox.setButtonSymbols(QDoubleSpinBox.NoButtons)
                spinbox.hide()
                self.numctrls_fiducial[m].append(spinbox)

        start_button = QPushButton("Start Registration", self)
        start_button.setCheckable(True)
        start_button.clicked.connect(partial(self.OnStartRegistration, ctrl=start_button))
        self.start_button = start_button

        reset_button = QPushButton("Reset", self)
        reset_button.clicked.connect(partial(self.OnReset, ctrl=reset_button))
        self.reset_button = reset_button

        next_button = QPushButton("Next", self)
        next_button.clicked.connect(self.OnNext)
        next_button.setEnabled(False)
        self.next_button = next_button

        back_button = QPushButton("Back", self)
        back_button.clicked.connect(self.OnBack)
        self.back_button = back_button

        top_layout = QHBoxLayout()
        top_layout.addWidget(start_button)
        top_layout.addWidget(reset_button)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(back_button)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(next_button)

        grid = QGridLayout()
        grid.addWidget(self.btns_set_fiducial[0], 1, 0, 1, 2, Qt.AlignVCenter)
        grid.addWidget(self.btns_set_fiducial[2], 0, 2, 1, 2, Qt.AlignHCenter)
        grid.addWidget(self.btns_set_fiducial[1], 1, 3, 1, 2, Qt.AlignVCenter)
        grid.addWidget(background, 1, 2)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_layout)
        main_layout.addLayout(grid)
        main_layout.addStretch(1)
        main_layout.addLayout(bottom_layout)

        self.__bind_events()

    def __bind_events(self):
        Publisher.subscribe(self.LoadImageFiducials, "Load image fiducials")
        Publisher.subscribe(self.SetImageFiducial, "Set image fiducial")
        Publisher.subscribe(self.UpdateImageCoordinates, "Set cross focal point")
        Publisher.subscribe(self.OnResetImageFiducials, "Reset image fiducials")
        Publisher.subscribe(self._OnStateProject, "Enable state project")
        Publisher.subscribe(self.StopRegistration, "Stop image registration")

    def _OnStateProject(self, state):
        self.UpdateData()

    def UpdateData(self):
        for m, btn in enumerate(self.btns_set_fiducial):
            btn.setChecked(self.image.IsImageFiducialSet(m))

            for n in range(3):
                value = self.image.GetImageFiducialForUI(m, n)
                self.numctrls_fiducial[m][n].setValue(value)

        self.UpdateNextButton()

    def LoadImageFiducials(self, label, position):
        fiducial = self.GetFiducialByAttribute(const.IMAGE_FIDUCIALS, "fiducial_name", label[:2])

        fiducial_index = fiducial["fiducial_index"]
        fiducial_name = fiducial["fiducial_name"]

        Publisher.sendMessage("Set image fiducial", fiducial_name=fiducial_name, position=position)

        self.btns_set_fiducial[fiducial_index].setChecked(True)
        for m in [0, 1, 2]:
            self.numctrls_fiducial[fiducial_index][m].setValue(position[m])

        self.UpdateNextButton()

    def GetFiducialByAttribute(self, fiducials, attribute_name, attribute_value):
        found = [fiducial for fiducial in fiducials if fiducial[attribute_name] == attribute_value]

        assert len(found) != 0, f"No fiducial found for which {attribute_name} = {attribute_value}"
        return found[0]

    def SetImageFiducial(self, fiducial_name, position):
        fiducial = self.GetFiducialByAttribute(
            const.IMAGE_FIDUCIALS, "fiducial_name", fiducial_name
        )
        fiducial_index = fiducial["fiducial_index"]

        self.image.SetImageFiducial(fiducial_index, position)

        if self.image.AreImageFiducialsSet():
            self.StopRegistration()
        self.UpdateNextButton()

    def UpdateImageCoordinates(self, position):
        self.current_coord = position

        for m in [0, 1, 2]:
            if not self.btns_set_fiducial[m].isChecked():
                for n in [0, 1, 2]:
                    self.numctrls_fiducial[m][n].setValue(float(position[n]))

    def OnImageFiducials(self, n):
        fiducial_name = const.IMAGE_FIDUCIALS[n]["fiducial_name"]

        if self.btns_set_fiducial[n].isChecked():
            position = (
                self.numctrls_fiducial[n][0].value(),
                self.numctrls_fiducial[n][1].value(),
                self.numctrls_fiducial[n][2].value(),
            )
        else:
            for m in [0, 1, 2]:
                self.numctrls_fiducial[n][m].setValue(float(self.current_coord[m]))
            position = np.nan

        Publisher.sendMessage("Set image fiducial", fiducial_name=fiducial_name, position=position)

    def OnNext(self):
        Publisher.sendMessage("Move to tracker page")

    def UpdateNextButton(self):
        self.next_button.setEnabled(self.image.AreImageFiducialsSet())

    def OnReset(self, ctrl=None):
        self.image.ResetImageFiducials()
        self.OnResetImageFiducials()

    def OnBack(self):
        Publisher.sendMessage("Move to head model page")

    def OnResetImageFiducials(self):
        self.next_button.setEnabled(False)
        for ctrl in self.btns_set_fiducial:
            ctrl.setChecked(False)
        self.start_button.setChecked(False)
        self.OnStartRegistration(ctrl=self.start_button)

    def StartRegistration(self):
        Publisher.sendMessage("Enable style", style=const.STATE_REGISTRATION)
        for button in self.btns_set_fiducial:
            button.setEnabled(True)
        self.start_button.setText("Stop Registration")
        self.start_button.setChecked(True)

    def StopRegistration(self):
        self.start_button.setText("Start Registration")
        self.start_button.setChecked(False)
        for button in self.btns_set_fiducial:
            button.setEnabled(False)
        Publisher.sendMessage("Disable style", style=const.STATE_REGISTRATION)

    def OnStartRegistration(self, ctrl=None):
        if ctrl is None:
            ctrl = self.start_button
        value = ctrl.isChecked()
        if value:
            self.StartRegistration()
        else:
            self.StopRegistration()


class TrackerPage(QWidget):
    def __init__(self, parent, nav_hub):
        super().__init__(parent)

        self.icp = nav_hub.icp
        self.tracker = nav_hub.tracker
        self.navigation = nav_hub.navigation
        self.pedal_connector = nav_hub.pedal_connector

        self.START_REGISTRATION_LABEL = _("Start Patient Registration")
        self.STOP_REGISTRATION_LABEL = _("Stop Patient Registration")
        self.registration_on = False

        self.bg_bmp = GetBitMapForBackground()

        self.fiducial_buttons = OrderedFiducialButtons(
            self,
            const.TRACKER_FIDUCIALS,
            self.tracker.IsTrackerFiducialSet,
            order=const.FIDUCIAL_REGISTRATION_ORDER,
        )
        background = QLabel(self)
        background.setPixmap(self.bg_bmp)

        for index, btn in enumerate(self.fiducial_buttons):
            btn.clicked.connect(partial(self.OnFiducialButton, index))
            btn.setEnabled(False)

        self.fiducial_buttons.Update()

        register_button = QPushButton("Record Fiducial", self)
        register_button.clicked.connect(self.OnRegister)
        register_button.setEnabled(False)
        self.register_button = register_button

        start_button = QPushButton("Start Patient Registration", self)
        start_button.setCheckable(True)
        start_button.clicked.connect(partial(self.OnStartRegistration, ctrl=start_button))
        self.start_button = start_button

        reset_button = QPushButton("Reset", self)
        reset_button.clicked.connect(self.OnReset)
        self.reset_button = reset_button

        back_button = QPushButton("Back", self)
        back_button.clicked.connect(self.OnBack)
        self.back_button = back_button

        preferences_button = QPushButton("Change tracker", self)
        preferences_button.clicked.connect(self.OnPreferences)
        self.preferences_button = preferences_button

        next_button = QPushButton("Next", self)
        next_button.clicked.connect(self.OnNext)
        if not self.tracker.AreTrackerFiducialsSet():
            next_button.setEnabled(False)
        self.next_button = next_button

        tracker_status = self.tracker.IsTrackerInitialized()
        current_label = QLabel(_("Current tracker: "), self)
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        current_label.setFont(font)
        main_label = QLabel(_("No tracker selected"), self)

        if tracker_status:
            main_label.setText(self.tracker.get_trackers()[self.tracker.GetTrackerId() - 1])

        self.main_label = main_label

        top_layout = QHBoxLayout()
        top_layout.addWidget(start_button)
        top_layout.addWidget(reset_button)

        middle_layout = QHBoxLayout()
        middle_layout.addWidget(current_label)
        middle_layout.addWidget(main_label)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(back_button)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(preferences_button)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(next_button)

        grid = QGridLayout()
        grid.addWidget(self.fiducial_buttons[0], 1, 0, 1, 2, Qt.AlignVCenter)
        grid.addWidget(self.fiducial_buttons[2], 0, 2, 1, 2, Qt.AlignHCenter)
        grid.addWidget(self.fiducial_buttons[1], 1, 3, 1, 2, Qt.AlignVCenter)
        grid.addWidget(background, 1, 2)
        grid.addWidget(register_button, 2, 2, 1, 2)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_layout)
        main_layout.addLayout(grid)
        main_layout.addStretch(1)
        main_layout.addLayout(middle_layout)
        main_layout.addSpacing(5)
        main_layout.addLayout(bottom_layout)

        self.__bind_events()

    def __bind_events(self):
        Publisher.subscribe(self.SetTrackerFiducial, "Set tracker fiducial")
        Publisher.subscribe(self.OnTrackerChanged, "Tracker changed")
        Publisher.subscribe(self.OnResetTrackerFiducials, "Reset tracker fiducials")

    def UpdateElements(self):
        if self.tracker.AreTrackerFiducialsSet():
            self.next_button.setEnabled(True)
        else:
            self.next_button.setEnabled(False)
        self.fiducial_buttons.Update()

    def StartRegistration(self):
        if not self.tracker.IsTrackerInitialized():
            self.start_button.setChecked(False)
            Publisher.sendMessage("Open preferences menu", page=2)
            self.StartRegistration()
            return

        self.registration_on = True
        for button in self.fiducial_buttons:
            button.setEnabled(True)
        self.fiducial_buttons.FocusNext()
        self.register_button.setEnabled(True)
        self.start_button.setText(self.STOP_REGISTRATION_LABEL)

        def set_fiducial_callback(state):
            index = self.fiducial_buttons.focused_index
            if state and index is not None:
                self.SetTrackerFiducial(index)

        self.pedal_connector.add_callback(
            "fiducial", set_fiducial_callback, remove_when_released=False
        )

    def StopRegistration(self):
        self.registration_on = False
        for button in self.fiducial_buttons:
            button.setEnabled(False)

        self.fiducial_buttons.ClearFocus()
        self.register_button.setEnabled(False)
        self.start_button.setChecked(False)
        self.start_button.setText(self.START_REGISTRATION_LABEL)

        self.pedal_connector.remove_callback("fiducial")

    def GetFiducialByAttribute(self, fiducials, attribute_name, attribute_value):
        found = [fiducial for fiducial in fiducials if fiducial[attribute_name] == attribute_value]

        assert len(found) != 0, f"No fiducial found for which {attribute_name} = {attribute_value}"
        return found[0]

    def OnSetTrackerFiducial(self, fiducial_name):
        fiducial = self.GetFiducialByAttribute(
            const.TRACKER_FIDUCIALS,
            "fiducial_name",
            fiducial_name,
        )
        fiducial_index = fiducial["fiducial_index"]
        self.SetTrackerFiducial(fiducial_index)

    def SetTrackerFiducial(self, fiducial_index):
        ref_mode_id = self.navigation.GetReferenceMode()
        success = self.tracker.SetTrackerFiducial(ref_mode_id, fiducial_index)

        if not success:
            return

        self.ResetICP()
        self.fiducial_buttons.Set(fiducial_index)

        if self.tracker.AreTrackerFiducialsSet():
            Publisher.sendMessage("Tracker fiducials set")

            self.next_button.setEnabled(True)
            self.StopRegistration()

        self.update()

    def OnFiducialButton(self, index):
        button = self.fiducial_buttons[index]

        if button is self.fiducial_buttons.focused:
            self.SetTrackerFiducial(index)
        elif not self.tracker.IsTrackerFiducialSet(index):
            self.fiducial_buttons.Focus(index)

    def OnRegister(self):
        index = self.fiducial_buttons.focused_index
        if index is not None:
            self.SetTrackerFiducial(index)

    def ResetICP(self):
        self.icp.ResetICP()

    def OnReset(self):
        self.tracker.ResetTrackerFiducials()
        self.update()

    def OnResetTrackerFiducials(self):
        self.UpdateElements()

        if self.registration_on:
            self.fiducial_buttons.FocusNext()

    def OnNext(self):
        Publisher.sendMessage("Move to refine page")

    def OnBack(self):
        Publisher.sendMessage("Move to image page")

    def OnPreferences(self):
        Publisher.sendMessage("Open preferences menu", page=2)

    def OnStartRegistration(self, ctrl=None):
        if ctrl is None:
            ctrl = self.start_button
        started = ctrl.isChecked()
        if started:
            self.tracker.ResetTrackerFiducials()
            self.StartRegistration()
        else:
            self.StopRegistration()

    def OnTrackerChanged(self):
        if self.tracker.GetTrackerId() != const.DEFAULT_TRACKER:
            self.main_label.setText(self.tracker.get_trackers()[self.tracker.GetTrackerId() - 1])
        else:
            self.main_label.setText(_("No tracker selected"))


class RefinePage(QWidget):
    def __init__(self, parent, nav_hub):
        super().__init__(parent)
        self.icp = nav_hub.icp
        self.tracker = nav_hub.tracker
        self.image = nav_hub.image
        self.navigation = nav_hub.navigation

        self.numctrls_fiducial = [[], [], [], [], [], []]
        const_labels = [label for label in const.FIDUCIAL_LABELS]
        labels = const_labels + const_labels
        self.labels = [QLabel(_(label), self) for label in labels]

        for m in range(6):
            for n in range(3):
                if m <= 2:
                    value = self.image.GetImageFiducialForUI(m, n)
                else:
                    value = self.tracker.GetTrackerFiducialForUI(m - 3, n)

                spinbox = QDoubleSpinBox(self)
                spinbox.setRange(-9999.9, 9999.9)
                spinbox.setDecimals(1)
                spinbox.setValue(value)
                spinbox.setReadOnly(True)
                spinbox.setButtonSymbols(QDoubleSpinBox.NoButtons)
                self.numctrls_fiducial[m].append(spinbox)

        txt_label_image = QLabel(_("Image Fiducials:"), self)
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        txt_label_image.setFont(font)

        coord_layout = QGridLayout()
        for m in range(3):
            coord_layout.addWidget(self.labels[m], m, 0)
            for n in range(3):
                coord_layout.addWidget(self.numctrls_fiducial[m][n], m, n + 1)

        txt_label_track = QLabel(_("Tracker Fiducials:"), self)
        txt_label_track.setFont(font)

        coord_layout_track = QGridLayout()
        for m in range(3, 6):
            coord_layout_track.addWidget(self.labels[m], m - 3, 0)
            for n in range(3):
                coord_layout_track.addWidget(self.numctrls_fiducial[m][n], m - 3, n + 1)

        txt_fre = QLabel(_("FRE:"), self)
        tooltip = _("Fiducial registration error")
        txt_fre.setToolTip(tooltip)

        value = self.icp.GetFreForUI()
        txtctrl_fre = QLineEdit(value, self)
        txtctrl_fre.setFixedWidth(60)
        txtctrl_fre.setAlignment(Qt.AlignCenter)
        txtctrl_fre.setFont(font)
        txtctrl_fre.setStyleSheet("background-color: white;")
        txtctrl_fre.setReadOnly(True)
        txtctrl_fre.setToolTip(tooltip)
        self.txtctrl_fre = txtctrl_fre

        self.OnUpdateUI()

        fre_layout = QHBoxLayout()
        fre_layout.addWidget(txt_fre)
        fre_layout.addWidget(txtctrl_fre)

        back_button = QPushButton("Back", self)
        back_button.clicked.connect(self.OnBack)
        self.back_button = back_button

        refine_button = QPushButton("Refine", self)
        refine_button.clicked.connect(self.OnRefine)
        self.refine_button = refine_button

        next_button = QPushButton("Next", self)
        next_button.clicked.connect(self.OnNext)
        self.next_button = next_button

        button_layout = QHBoxLayout()
        button_layout.addWidget(back_button)
        button_layout.addStretch(1)
        button_layout.addWidget(refine_button)
        button_layout.addStretch(1)
        button_layout.addWidget(next_button)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(txt_label_image)
        main_layout.addLayout(coord_layout)
        main_layout.addWidget(txt_label_track)
        main_layout.addLayout(coord_layout_track)
        main_layout.addSpacing(10)
        main_layout.addLayout(fre_layout)
        main_layout.addStretch(1)
        main_layout.addLayout(button_layout)

        self.__bind_events()

    def __bind_events(self):
        Publisher.subscribe(self.OnUpdateUI, "Update UI for refine tab")
        Publisher.subscribe(self.OnResetTrackerFiducials, "Reset tracker fiducials")

    def OnUpdateUI(self):
        for m in range(6):
            for n in range(3):
                if m <= 2:
                    value = self.image.GetImageFiducialForUI(m, n)
                else:
                    value = self.tracker.GetTrackerFiducialForUI(m - 3, n)
                self.numctrls_fiducial[m][n].setValue(value)

        if self.tracker.AreTrackerFiducialsSet() and self.image.AreImageFiducialsSet():
            self.navigation.EstimateTrackerToInVTransformationMatrix(self.tracker, self.image)
            self.navigation.UpdateFiducialRegistrationError(self.tracker, self.image)
            fre, fre_ok = self.navigation.GetFiducialRegistrationError(self.icp)

            self.txtctrl_fre.setText(str(round(fre, 2)))
            if fre_ok:
                r, g, b = const.GREEN_COLOR_RGB
                self.txtctrl_fre.setStyleSheet(f"background-color: rgb({r},{g},{b});")
            else:
                r, g, b = const.RED_COLOR_RGB
                self.txtctrl_fre.setStyleSheet(f"background-color: rgb({r},{g},{b});")

    def OnResetTrackerFiducials(self):
        for m in range(3):
            for n in range(3):
                value = self.tracker.GetTrackerFiducialForUI(m, n)
                self.numctrls_fiducial[m + 3][n].setValue(value)

    def OnBack(self):
        Publisher.sendMessage("Move to tracker page")

    def OnNext(self):
        Publisher.sendMessage("Move to stylus page")

    def OnRefine(self):
        self.icp.RegisterICP(self.navigation, self.tracker)
        if self.icp.use_icp:
            self.OnUpdateUI()


class StylusPage(QWidget):
    def __init__(self, parent, nav_hub):
        super().__init__(parent)
        self.navigation = nav_hub.navigation
        self.tracker = nav_hub.tracker

        self.done = False

        lbl = QLabel(_("Calibrate stylus with head"), self)
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        lbl.setFont(font)
        self.lbl = lbl

        self.help_img = QImage(os.path.join(inv_paths.ICON_DIR, "align.png"))

        grey_img = self.help_img.convertToFormat(QImage.Format_Grayscale8)
        self.help_label = QLabel(self)
        self.help_label.setPixmap(QPixmap.fromImage(grey_img))

        lbl_rec = QLabel(_("Point stylus up relative to head, like so:"), self)
        btn_rec = QPushButton(_("Record"), self)
        btn_rec.setToolTip("Record stylus orientation relative to head")
        btn_rec.clicked.connect(self.onRecord)
        self.btn_rec = btn_rec

        content_layout = QVBoxLayout()
        content_layout.addWidget(lbl)
        content_layout.addWidget(lbl_rec)
        content_layout.addWidget(self.help_label)
        content_layout.addWidget(btn_rec)
        self.content_layout = content_layout

        back_button = QPushButton("Back", self)
        back_button.clicked.connect(self.OnBack)
        next_button = QPushButton("Next", self)
        next_button.clicked.connect(self.OnNext)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(back_button)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(next_button)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(content_layout)
        main_layout.addStretch(1)
        main_layout.addLayout(bottom_layout)

    def onRecord(self):
        marker_visibilities, __, coord_raw = self.tracker.GetTrackerCoordinates(
            ref_mode_id=0, n_samples=1
        )

        if marker_visibilities[0] and marker_visibilities[1]:
            if self.navigation.OnRecordStylusOrientation(coord_raw) and not self.done:
                self.done = True
                self.help_label.setPixmap(QPixmap.fromImage(self.help_img))
        else:
            QMessageBox.information(
                self, _("InVesalius 3"), _("Probe or head not visible to tracker!")
            )

    def OnBack(self):
        Publisher.sendMessage("Move to refine page")

    def OnNext(self):
        Publisher.sendMessage("Move to stimulator page")


class StimulatorPage(QWidget):
    def __init__(self, parent, nav_hub):
        super().__init__(parent)
        self.navigation = nav_hub.navigation

        self.coil_registrations = []

        lbl = QLabel(
            _(
                f"Ready for navigation with {self.navigation.n_coils} coil{'' if self.navigation.n_coils == 1 else 's'}!"
            ),
            self,
        )
        self.lbl = lbl

        btn_edit = QPushButton(_("Edit coil registration in Preferences"), self)
        btn_edit.setToolTip("Open preferences menu")
        btn_edit.clicked.connect(self.OnEditPreferences)

        back_button = QPushButton("Back", self)
        back_button.clicked.connect(self.OnBack)

        next_button = QPushButton("Navigation", self)
        next_button.clicked.connect(self.OnNext)
        if not self.navigation.CoilSelectionDone():
            self.lbl.setText("Please select a coil registration")
            next_button.setEnabled(False)
        self.next_button = next_button

        top_layout = QVBoxLayout()
        top_layout.addWidget(lbl)
        top_layout.addWidget(btn_edit)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(back_button)
        bottom_layout.addStretch(1)
        bottom_layout.addWidget(next_button)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_layout)
        main_layout.addStretch(1)
        main_layout.addLayout(bottom_layout)

        self.__bind_events()

    def __bind_events(self):
        Publisher.subscribe(self.CoilSelectionDone, "Coil selection done")
        Publisher.subscribe(self.OnCloseProject, "Close project data")
        Publisher.subscribe(self.OnCloseProject, "Remove object data")

    def OnCloseProject(self):
        Publisher.sendMessage("Enable start navigation button", enabled=False)

    def CoilSelectionDone(self, done):
        if done:
            self.lbl.setText(
                f"Ready for navigation with {self.navigation.n_coils} coil{'' if self.navigation.n_coils == 1 else 's'}!"
            )
        else:
            self.lbl.setText("Please select which coil(s) to track")

        self.next_button.setEnabled(done)
        self.lbl.setVisible(True)

    def OnEditPreferences(self):
        Publisher.sendMessage("Open preferences menu", page=3)

    def OnBack(self):
        Publisher.sendMessage("Move to stylus page")

    def OnNext(self):
        Publisher.sendMessage("Open navigation menu")


class NavigationPanel(QWidget):
    def __init__(self, parent, nav_hub):
        super().__init__(parent)

        self.nav_hub = nav_hub
        self.navigation = nav_hub.navigation
        self.tracker = nav_hub.tracker
        self.icp = nav_hub.icp
        self.image = nav_hub.image
        self.pedal_connector = nav_hub.pedal_connector
        self.neuronavigation_api = nav_hub.neuronavigation_api
        self.mep_visualizer = nav_hub.mep_visualizer

        self.__bind_events()

        self.control_panel = ControlPanel(self, nav_hub)
        self.marker_panel = MarkersPanel(self, nav_hub)

        top_layout = QHBoxLayout()
        top_layout.addWidget(self.marker_panel, 1)

        bottom_layout = QHBoxLayout()
        bottom_layout.addWidget(self.control_panel)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(top_layout, 1)
        main_layout.addLayout(bottom_layout)

        self.main_layout = main_layout

    def __bind_events(self):
        Publisher.subscribe(self.OnCloseProject, "Close project data")
        Publisher.subscribe(self.OnUpdateNavigationPanel, "Update navigation panel")

    def OnUpdateNavigationPanel(self):
        self.updateGeometry()

    def OnCloseProject(self):
        self.tracker.ResetTrackerFiducials()
        self.image.ResetImageFiducials()

        Publisher.sendMessage("Disconnect tracker")
        Publisher.sendMessage("Delete all markers")
        Publisher.sendMessage("Update marker offset state", create=False)
        Publisher.sendMessage("Remove tracts")
        Publisher.sendMessage("Disable style", style=const.SLICE_STATE_CROSS)
        Publisher.sendMessage("Reset cam clipping range")
        self.navigation.StopNavigation()
        self.navigation.__init__(
            pedal_connector=self.pedal_connector, neuronavigation_api=self.neuronavigation_api
        )
        self.tracker.__init__()
        self.icp.__init__()


class ControlPanel(QWidget):
    def __init__(self, parent, nav_hub):
        super().__init__(parent)

        self.navigation = nav_hub.navigation
        self.tracker = nav_hub.tracker
        self.robot = nav_hub.robot
        self.icp = nav_hub.icp
        self.image = nav_hub.image
        self.mep_visualizer = nav_hub.mep_visualizer

        self.nav_status = False
        self.target_mode = False

        self.navigation_status = False

        self.target_selected = False

        ICON_SIZE = QSize(48, 48)
        RED_COLOR = const.RED_COLOR_RGB
        self.RED_COLOR = RED_COLOR
        GREEN_COLOR = const.GREEN_COLOR_RGB
        self.GREEN_COLOR = GREEN_COLOR
        GREY_COLOR = (217, 217, 217)
        self.GREY_COLOR = GREY_COLOR

        tooltip = _("Start navigation")
        btn_nav = QPushButton(_("Start navigation"), self)
        btn_nav.setCheckable(True)
        font = QFont()
        font.setPointSize(9)
        font.setBold(True)
        btn_nav.setFont(font)
        btn_nav.setToolTip(tooltip)
        self.btn_nav = btn_nav
        self.btn_nav.clicked.connect(partial(self.OnStartNavigationButton, btn_nav=self.btn_nav))

        def _make_toggle_button(icon_path, tooltip_text, size, bg_color):
            bmp = QPixmap(str(inv_paths.ICON_DIR.joinpath(icon_path)))
            btn = QPushButton(self)
            btn.setCheckable(True)
            btn.setFixedSize(size)
            btn.setIcon(QIcon(bmp))
            btn.setIconSize(QSize(size.width() - 8, size.height() - 8))
            r, g, b = bg_color
            btn.setStyleSheet(f"background-color: rgb({r},{g},{b});")
            btn.setToolTip(tooltip_text)
            return btn

        tractography_checkbox = _make_toggle_button(
            "tract.png", _("Control Tractography"), ICON_SIZE, GREY_COLOR
        )
        tractography_checkbox.setChecked(False)
        tractography_checkbox.setEnabled(False)
        tractography_checkbox.clicked.connect(
            partial(self.OnTractographyCheckbox, ctrl=tractography_checkbox)
        )
        self.tractography_checkbox = tractography_checkbox

        track_object_button = _make_toggle_button(
            "coil.png", _("Track coil"), ICON_SIZE, GREY_COLOR
        )
        track_object_button.setEnabled(True)
        track_object_button.setChecked(False)
        track_object_button.clicked.connect(
            partial(self.OnTrackObjectButton, ctrl=track_object_button)
        )
        self.track_object_button = track_object_button

        lock_to_target_button = _make_toggle_button(
            "lock_to_target.png",
            _("Allow triggering only if the coil is at the target"),
            ICON_SIZE,
            GREY_COLOR,
        )
        lock_to_target_button.setChecked(False)
        lock_to_target_button.setEnabled(False)
        lock_to_target_button.clicked.connect(
            partial(self.OnLockToTargetButton, ctrl=lock_to_target_button)
        )
        self.lock_to_target_button = lock_to_target_button

        show_coil_button = _make_toggle_button(
            "coil_eye.png", _("Show coil"), ICON_SIZE, GREY_COLOR
        )
        show_coil_button.setChecked(False)
        show_coil_button.setEnabled(True)
        show_coil_button.clicked.connect(self.OnShowCoil)
        show_coil_button.setContextMenuPolicy(Qt.CustomContextMenu)
        show_coil_button.customContextMenuRequested.connect(self.ShowCoilChoice)
        self.show_coil_button = show_coil_button

        show_probe_button = _make_toggle_button(
            "stylus_eye.png", _("Show probe"), ICON_SIZE, GREY_COLOR
        )
        show_probe_button.setEnabled(True)
        self.UpdateToggleButton(show_probe_button, False)
        show_probe_button.clicked.connect(self.OnShowProbe)
        self.show_probe_button = show_probe_button

        checkbox_serial_port = _make_toggle_button(
            "wave.png",
            _("Enable serial port communication to trigger pulse and create markers"),
            ICON_SIZE,
            RED_COLOR,
        )
        checkbox_serial_port.setChecked(False)
        checkbox_serial_port.clicked.connect(
            partial(self.OnEnableSerialPort, ctrl=checkbox_serial_port)
        )
        self.checkbox_serial_port = checkbox_serial_port

        efield_checkbox = _make_toggle_button(
            "field.png", _("Control E-Field"), ICON_SIZE, GREY_COLOR
        )
        efield_checkbox.setChecked(False)
        efield_checkbox.setEnabled(False)
        efield_checkbox.clicked.connect(partial(self.OnEfieldCheckbox, ctrl=efield_checkbox))
        self.efield_checkbox = efield_checkbox

        target_mode_button = _make_toggle_button(
            "target.png", _("Target mode"), ICON_SIZE, GREY_COLOR
        )
        target_mode_button.setChecked(False)
        target_mode_button.setEnabled(False)
        target_mode_button.clicked.connect(self.OnTargetButton)
        self.target_mode_button = target_mode_button
        self.UpdateTargetButton()

        robot_track_target_button = _make_toggle_button(
            "robot_track_target.png", _("Track target with robot"), ICON_SIZE, GREY_COLOR
        )
        robot_track_target_button.setChecked(False)
        robot_track_target_button.setEnabled(False)
        robot_track_target_button.clicked.connect(
            partial(self.OnRobotTrackTargetButton, ctrl=robot_track_target_button)
        )
        self.robot_track_target_button = robot_track_target_button

        robot_move_away_button = _make_toggle_button(
            "robot_move_away.png", _("Move robot away from head"), ICON_SIZE, GREY_COLOR
        )
        robot_move_away_button.setChecked(False)
        robot_move_away_button.setEnabled(False)
        robot_move_away_button.clicked.connect(
            partial(self.OnRobotMoveAwayButton, ctrl=robot_move_away_button)
        )
        self.robot_move_away_button = robot_move_away_button

        robot_free_drive_button = _make_toggle_button(
            "robot_free_drive.png", _("Free drive robot"), ICON_SIZE, GREY_COLOR
        )
        robot_free_drive_button.setChecked(False)
        robot_free_drive_button.setEnabled(False)
        robot_free_drive_button.clicked.connect(
            partial(self.OnRobotFreeDriveButton, ctrl=robot_free_drive_button)
        )
        self.robot_free_drive_button = robot_free_drive_button

        show_motor_map_button = _make_toggle_button(
            "brain_eye.png", _("Show TMS motor mapping on brain"), ICON_SIZE, GREY_COLOR
        )
        show_motor_map_button.setChecked(False)
        show_motor_map_button.setEnabled(True)
        show_motor_map_button.clicked.connect(
            partial(self.OnShowMotorMapButton, ctrl=show_motor_map_button)
        )
        self.show_motor_map_button = show_motor_map_button

        start_navigation_layout = QVBoxLayout()
        start_navigation_layout.addWidget(btn_nav)

        navigation_buttons_layout = QGridLayout()
        navigation_buttons_layout.setSpacing(5)
        navigation_buttons_layout.addWidget(tractography_checkbox, 0, 0)
        navigation_buttons_layout.addWidget(target_mode_button, 0, 1)
        navigation_buttons_layout.addWidget(track_object_button, 0, 2)
        navigation_buttons_layout.addWidget(checkbox_serial_port, 0, 3)
        navigation_buttons_layout.addWidget(efield_checkbox, 1, 0)
        navigation_buttons_layout.addWidget(lock_to_target_button, 1, 1)
        navigation_buttons_layout.addWidget(show_coil_button, 1, 2)
        navigation_buttons_layout.addWidget(show_probe_button, 1, 3)
        navigation_buttons_layout.addWidget(show_motor_map_button, 2, 0)

        robot_buttons_layout = QGridLayout()
        robot_buttons_layout.setSpacing(5)
        robot_buttons_layout.addWidget(robot_track_target_button, 0, 0)
        robot_buttons_layout.addWidget(robot_move_away_button, 0, 1)
        robot_buttons_layout.addWidget(robot_free_drive_button, 0, 2)

        main_layout = QVBoxLayout(self)
        main_layout.addLayout(start_navigation_layout)
        main_layout.addLayout(navigation_buttons_layout)
        main_layout.addLayout(robot_buttons_layout)

        self.__bind_events()
        self.LoadConfig()

    def __bind_events(self):
        Publisher.subscribe(self.OnStartNavigation, "Start navigation")
        Publisher.subscribe(self.OnStopNavigation, "Stop navigation")
        Publisher.subscribe(self.OnCheckStatus, "Navigation status")
        Publisher.subscribe(self.SetTarget, "Set target")
        Publisher.subscribe(self.UnsetTarget, "Unset target")
        Publisher.subscribe(self.UpdateNavigationStatus, "Navigation status")

        Publisher.subscribe(self.OnRobotStatus, "Robot to Neuronavigation: Robot connection status")
        Publisher.subscribe(self.SetTargetMode, "Set target mode")

        Publisher.subscribe(self.UpdateTractsVisualization, "Update tracts visualization")

        Publisher.subscribe(self.PressShowProbeButton, "Press show-probe button")

        Publisher.subscribe(self.OnCoilSelectionDone, "Coil selection done")

        Publisher.subscribe(self.PressShowCoilButton, "Press show-coil button")
        Publisher.subscribe(self.EnableShowCoilButton, "Enable show-coil button")

        Publisher.subscribe(self.PressTrackObjectButton, "Press track object button")
        Publisher.subscribe(self.EnableTrackObjectButton, "Enable track object button")

        Publisher.subscribe(self.PressRobotTrackTargetButton, "Press robot button")
        Publisher.subscribe(self.EnableRobotTrackTargetButton, "Enable robot button")

        Publisher.subscribe(self.PressRobotMoveAwayButton, "Press move away button")
        Publisher.subscribe(self.EnableRobotMoveAwayButton, "Enable move away button")

        Publisher.subscribe(self.EnableRobotFreeDriveButton, "Enable free drive button")

        Publisher.subscribe(self.ShowTargetButton, "Show target button")
        Publisher.subscribe(self.HideTargetButton, "Hide target button")
        Publisher.subscribe(self.PressTargetModeButton, "Press target mode button")

        Publisher.subscribe(self.PressMotorMapButton, "Press motor map button")
        Publisher.subscribe(self.EnableMotorMapButton, "Enable motor map button")

        Publisher.subscribe(self.TrackObject, "Track object")

        Publisher.subscribe(self.UpdateTrekkerObject, "Update Trekker object")
        Publisher.subscribe(self.UpdateNumTracts, "Update number of tracts")
        Publisher.subscribe(self.UpdateSeedOffset, "Update seed offset")
        Publisher.subscribe(self.UpdateSeedRadius, "Update seed radius")
        Publisher.subscribe(self.UpdateNumberThreads, "Update number of threads")
        Publisher.subscribe(self.UpdateTractsVisualization, "Update tracts visualization")
        Publisher.subscribe(self.UpdatePeelVisualization, "Update peel visualization")
        Publisher.subscribe(self.UpdateEfieldVisualization, "Update e-field visualization")
        Publisher.subscribe(self.EnableACT, "Enable ACT")
        Publisher.subscribe(self.UpdateACTData, "Update ACT data")

    def LoadConfig(self):
        session = ses.Session()
        state = session.GetConfig("navigation", {})
        track_coil = state.get("track_coil", False)
        self.PressTrackObjectButton(track_coil)

    def UpdateToggleButton(self, ctrl, state=None):
        if state is None:
            state = ctrl.isChecked()

        ctrl.setChecked(state)

        if state:
            r, g, b = self.GREEN_COLOR
        else:
            r, g, b = self.RED_COLOR
        ctrl.setStyleSheet(f"background-color: rgb({r},{g},{b});")

    def EnableToggleButton(self, ctrl, state):
        if ctrl.isEnabled() == state:
            return

        ctrl.setEnabled(state)
        r, g, b = self.GREY_COLOR
        ctrl.setStyleSheet(f"background-color: rgb({r},{g},{b});")

    def OnStartNavigation(self):
        if not self.tracker.AreTrackerFiducialsSet() or not self.image.AreImageFiducialsSet():
            QMessageBox.information(
                self, _("InVesalius 3"), _("Invalid fiducials, select all coordinates.")
            )

        elif not self.tracker.IsTrackerInitialized():
            dlg.ShowNavigationTrackerWarning(0, "choose")

        else:
            Publisher.sendMessage("Enable style", style=const.STATE_NAVIGATION)
            Publisher.sendMessage("Hide current mask")

            self.navigation.EstimateTrackerToInVTransformationMatrix(self.tracker, self.image)
            self.navigation.StartNavigation(self.tracker, self.icp)

            self.robot.SendTargetToRobot()

    def OnStartNavigationButton(self, btn_nav=None):
        if btn_nav is None:
            btn_nav = self.btn_nav
        nav_id = btn_nav.isChecked()
        if not nav_id:
            QTimer.singleShot(0, lambda: Publisher.sendMessage("Stop navigation"))
            tooltip = _("Start navigation")
            btn_nav.setToolTip(tooltip)
            btn_nav.setText(_("Start navigation"))
        else:
            Publisher.sendMessage("Start navigation")
            if self.nav_status:
                tooltip = _("Stop navigation")
                btn_nav.setToolTip(tooltip)
                btn_nav.setText(_("Stop navigation"))
            else:
                btn_nav.setChecked(False)

    def OnStopNavigation(self):
        Publisher.sendMessage("Disable style", style=const.STATE_NAVIGATION)
        self.robot.SetObjective(RobotObjective.NONE)
        self.navigation.StopNavigation()

    def UnsetTarget(self, marker):
        self.navigation.target = None
        self.target_selected = False
        self.UpdateTargetButton()

    def SetTarget(self, marker):
        coord = marker.position + marker.orientation
        coord[1] = -coord[1]

        self.navigation.target = coord

        self.EnableToggleButton(self.lock_to_target_button, 1)
        self.UpdateToggleButton(self.lock_to_target_button, True)
        self.navigation.SetLockToTarget(True)

        self.target_selected = True
        self.UpdateTargetButton()
        self.UpdateRobotButtons()

    def UpdateNavigationStatus(self, nav_status, vis_status):
        if not nav_status:
            self.nav_status = False
            self.current_orientation = None, None, None
        else:
            self.nav_status = True

        self.UpdateRobotButtons()

    def OnCheckStatus(self, nav_status, vis_status):
        if nav_status:
            self.UpdateToggleButton(self.checkbox_serial_port)
            self.EnableToggleButton(self.checkbox_serial_port, 0)
        else:
            self.EnableToggleButton(self.checkbox_serial_port, 1)
            self.UpdateToggleButton(self.checkbox_serial_port)

    def OnCoilSelectionDone(self, done):
        self.PressTrackObjectButton(done)
        self.PressShowCoilButton(pressed=done)

    def OnRobotStatus(self, data):
        if data:
            self.updateGeometry()

    def UpdateRobotButtons(self):
        track_target_button_enabled = (
            self.nav_status
            and self.target_selected
            and self.target_mode
            and self.robot.IsConnected()
        )
        self.EnableRobotTrackTargetButton(enabled=track_target_button_enabled)

        move_away_button_enabled = self.robot.IsConnected()
        self.EnableRobotMoveAwayButton(enabled=move_away_button_enabled)

        free_drive_button_enabled = self.robot.IsConnected()
        self.EnableRobotFreeDriveButton(enabled=free_drive_button_enabled)

    def SetTargetMode(self, enabled=False):
        self.target_mode = enabled
        self.UpdateRobotButtons()

        if not enabled:
            self.robot.SetObjective(RobotObjective.NONE)

    def OnTractographyCheckbox(self, ctrl=None):
        if ctrl is None:
            ctrl = self.tractography_checkbox
        self.view_tracts = ctrl.isChecked()
        self.UpdateToggleButton(ctrl)
        Publisher.sendMessage("Update tracts visualization", data=self.view_tracts)
        if not self.view_tracts:
            Publisher.sendMessage("Remove tracts")
            Publisher.sendMessage("Update marker offset state", create=False)

    def UpdateTractsVisualization(self, data):
        self.navigation.view_tracts = data
        self.EnableToggleButton(self.tractography_checkbox, 1)
        self.UpdateToggleButton(self.tractography_checkbox, data)

    def UpdatePeelVisualization(self, data):
        self.navigation.peel_loaded = data

    def UpdateEfieldVisualization(self, data):
        self.navigation.e_field_loaded = data

    def UpdateTrekkerObject(self, data):
        self.navigation.trekker = data

    def UpdateNumTracts(self, data):
        self.navigation.n_tracts = data

    def UpdateSeedOffset(self, data):
        self.navigation.seed_offset = data

    def UpdateSeedRadius(self, data):
        self.navigation.seed_radius = data

    def UpdateNumberThreads(self, data):
        self.navigation.n_threads = data

    def UpdateACTData(self, data):
        self.navigation.act_data = data

    def EnableACT(self, data):
        self.navigation.enable_act = data

    def EnableTrackObjectButton(self, enabled):
        self.EnableToggleButton(self.track_object_button, enabled)
        self.UpdateToggleButton(self.track_object_button)

    def PressTrackObjectButton(self, pressed):
        self.UpdateToggleButton(self.track_object_button, pressed)
        self.OnTrackObjectButton()

    def OnTrackObjectButton(self, evt=None, ctrl=None):
        if ctrl is not None:
            self.UpdateToggleButton(ctrl)
        pressed = self.track_object_button.isChecked()
        Publisher.sendMessage("Track object", enabled=pressed)
        if not pressed and self.target_mode_button.isChecked():
            Publisher.sendMessage("Press target mode button", pressed=False)

        Publisher.sendMessage("Press show-coil button", pressed=pressed)
        Publisher.sendMessage("Press show-probe button", pressed=(not pressed))

    def OnLockToTargetButton(self, ctrl=None):
        if ctrl is None:
            ctrl = self.lock_to_target_button
        self.UpdateToggleButton(ctrl)
        value = ctrl.isChecked()
        self.navigation.SetLockToTarget(value)

    def PressShowCoilButton(self, pressed=False):
        self.UpdateToggleButton(self.show_coil_button, pressed)
        self.OnShowCoil()

    def EnableShowCoilButton(self, enabled=False):
        self.EnableToggleButton(self.show_coil_button, enabled)
        self.UpdateToggleButton(self.show_coil_button)

    def ShowCoilChoice(self, pos):
        coil_names = list(self.navigation.coil_registrations)

        show_coil_menu = QMenu(self)
        for coil_name in coil_names:
            action = show_coil_menu.addAction(coil_name)
            action.triggered.connect(
                lambda checked, name=coil_name: self.OnShowCoil(coil_name=name)
            )

        show_coil_menu.exec_(self.show_coil_button.mapToGlobal(pos))

    def OnShowCoil(self, evt=None, coil_name=None):
        self.UpdateToggleButton(self.show_coil_button)
        pressed = self.show_coil_button.isChecked()
        Publisher.sendMessage("Show coil in viewer volume", state=pressed, coil_name=coil_name)

    def PressShowProbeButton(self, pressed=False):
        self.UpdateToggleButton(self.show_probe_button, pressed)
        self.OnShowProbe()

    def OnShowProbe(self, evt=None):
        self.UpdateToggleButton(self.show_probe_button)
        pressed = self.show_probe_button.isChecked()
        Publisher.sendMessage("Show probe in viewer volume", state=pressed)

    def OnEnableSerialPort(self, ctrl=None):
        if ctrl is None:
            ctrl = self.checkbox_serial_port
        self.UpdateToggleButton(ctrl)
        if ctrl.isChecked():
            dlg_port = dlg.SetCOMPort(select_baud_rate=False)

            if dlg_port.exec() != QDialog.Accepted:
                self.UpdateToggleButton(ctrl, False)
                return

            com_port = dlg_port.GetCOMPort()
            if not com_port:
                self.UpdateToggleButton(ctrl, False)
                return
            baud_rate = 115200

            Publisher.sendMessage(
                "Update serial port",
                serial_port_in_use=True,
                com_port=com_port,
                baud_rate=baud_rate,
            )
        else:
            Publisher.sendMessage("Update serial port", serial_port_in_use=False)

    def OnEfieldCheckbox(self, ctrl=None):
        if ctrl is None:
            ctrl = self.efield_checkbox
        self.UpdateToggleButton(ctrl)

    def TrackObject(self, enabled):
        self.UpdateTargetButton()

    def ShowTargetButton(self):
        self.target_mode_button.setVisible(True)

    def HideTargetButton(self):
        self.target_mode_button.setVisible(False)

    def UpdateTargetButton(self):
        enabled = self.target_selected and self.navigation.track_coil
        self.EnableToggleButton(self.target_mode_button, enabled)

    def PressTargetModeButton(self, pressed):
        if pressed:
            self.EnableToggleButton(self.target_mode_button, True)

        self.UpdateToggleButton(self.target_mode_button, pressed)
        self.OnTargetButton()

    def OnTargetButton(self, evt=None):
        pressed = self.target_mode_button.isChecked()
        self.UpdateToggleButton(self.target_mode_button, pressed)

        Publisher.sendMessage("Set target mode", enabled=pressed)
        if pressed:
            self.robot.SetObjective(RobotObjective.NONE)

    def EnableRobotTrackTargetButton(self, enabled=False):
        self.EnableToggleButton(self.robot_track_target_button, enabled)
        self.UpdateToggleButton(self.robot_track_target_button)

    def PressRobotTrackTargetButton(self, pressed):
        if pressed:
            if not self.robot_track_target_button.isEnabled():
                return
        self.UpdateToggleButton(self.robot_track_target_button, pressed)
        self.OnRobotTrackTargetButton()

    def OnRobotTrackTargetButton(self, evt=None, ctrl=None):
        self.UpdateToggleButton(self.robot_track_target_button)
        pressed = self.robot_track_target_button.isChecked()
        Publisher.sendMessage("Robot tracking status", status=pressed)
        if pressed:
            self.robot.SetObjective(RobotObjective.TRACK_TARGET)
        else:
            if self.robot.objective == RobotObjective.TRACK_TARGET:
                self.robot.SetObjective(RobotObjective.NONE)
            Publisher.sendMessage(
                "Robot to Neuronavigation: Update robot warning", robot_warning=""
            )

    def EnableRobotMoveAwayButton(self, enabled=False):
        self.EnableToggleButton(self.robot_move_away_button, enabled)
        self.UpdateToggleButton(self.robot_move_away_button)

    def PressRobotMoveAwayButton(self, pressed):
        self.UpdateToggleButton(self.robot_move_away_button, pressed)
        self.OnRobotMoveAwayButton()

    def OnRobotMoveAwayButton(self, evt=None, ctrl=None):
        self.UpdateToggleButton(self.robot_move_away_button)
        pressed = self.robot_move_away_button.isChecked()
        if pressed:
            self.robot.SetObjective(RobotObjective.MOVE_AWAY_FROM_HEAD)
        else:
            if self.robot.objective == RobotObjective.MOVE_AWAY_FROM_HEAD:
                self.robot.SetObjective(RobotObjective.NONE)
            Publisher.sendMessage(
                "Robot to Neuronavigation: Update robot warning", robot_warning=""
            )

    def EnableRobotFreeDriveButton(self, enabled=False):
        self.EnableToggleButton(self.robot_free_drive_button, enabled)
        self.UpdateToggleButton(self.robot_free_drive_button)

    def OnRobotFreeDriveButton(self, evt=None, ctrl=None):
        self.UpdateToggleButton(self.robot_free_drive_button)
        pressed = self.robot_free_drive_button.isChecked()
        if pressed:
            Publisher.sendMessage("Neuronavigation to Robot: Set free drive", set=True)
        else:
            Publisher.sendMessage("Neuronavigation to Robot: Set free drive", set=False)

    def PressMotorMapButton(self, pressed=False):
        self.UpdateToggleButton(self.show_motor_map_button, pressed)
        self.OnShowMotorMapButton()

    def EnableMotorMapButton(self, enabled=False):
        self.EnableToggleButton(self.show_motor_map_button, enabled)
        self.UpdateToggleButton(self.show_motor_map_button)

    def OnShowMotorMapButton(self, evt=None, ctrl=None):
        pressed = self.show_motor_map_button.isChecked()
        if self.mep_visualizer.DisplayMotorMap(show=pressed):
            self.UpdateToggleButton(self.show_motor_map_button)


def _set_tree_item_bg(item, color, col_count):
    brush = QBrush(color)
    for c in range(col_count):
        item.setBackground(c, brush)


class MarkersPanel(QWidget):
    def __init__(self, parent, nav_hub):
        super().__init__(parent)

        self.navigation = nav_hub.navigation
        self.markers = nav_hub.markers

        if has_mTMS:
            self.mTMS = mTMS()
        else:
            self.mTMS = None

        self.__bind_events()

        self.session = ses.Session()

        self.currently_focused_marker = None
        self.current_position = [0, 0, 0]
        self.current_orientation = [None, None, None]
        self.current_seed = 0, 0, 0
        self.cortex_position_orientation = [None, None, None, None, None, None]
        self.nav_status = False
        self.efield_data_saved = False
        self.efield_target_idx = None

        self.marker_colour = const.MARKER_COLOUR
        self.marker_size = const.MARKER_SIZE
        self.arrow_marker_size = const.ARROW_MARKER_SIZE
        self.current_session = 1

        self.itemDataMap = {}

        self.brain_actor = None

        spin_session = QSpinBox(self)
        spin_session.setFixedSize(40, 23)
        spin_session.setRange(1, 99)
        spin_session.setValue(self.current_session)
        spin_session.setToolTip("Set session")
        spin_session.valueChanged.connect(lambda val: self.OnSessionChanged(val))

        select_colour = ColourSelectButton(
            self, colour=[int(255 * s) for s in self.marker_colour], size=(20, 23)
        )
        select_colour.setToolTip("Set colour")
        select_colour.colour_selected.connect(self.OnSelectColour)

        btn_create = QPushButton(_("Create marker"), self)
        btn_create.setFixedSize(135, 23)
        btn_create.clicked.connect(self.OnCreateMarker)

        sizer_create = QHBoxLayout()
        sizer_create.setSpacing(5)
        sizer_create.addWidget(spin_session)
        sizer_create.addWidget(select_colour)
        sizer_create.addWidget(btn_create)

        btn_save = QPushButton(_("Save"), self)
        btn_save.setFixedSize(65, 23)
        btn_save.clicked.connect(self.OnSaveMarkers)

        btn_load = QPushButton(_("Load"), self)
        btn_load.setFixedSize(65, 23)
        btn_load.clicked.connect(self.OnLoadMarkers)

        btn_show_hide_all = QPushButton(_("Hide all"), self)
        btn_show_hide_all.setCheckable(True)
        btn_show_hide_all.setFixedSize(65, 23)
        btn_show_hide_all.clicked.connect(
            partial(self.OnShowHideAllMarkers, ctrl=btn_show_hide_all)
        )

        sizer_btns = QHBoxLayout()
        sizer_btns.setSpacing(5)
        sizer_btns.addWidget(btn_save)
        sizer_btns.addWidget(btn_load)
        sizer_btns.addWidget(btn_show_hide_all)

        btn_delete_single = QPushButton(_("Delete"), self)
        btn_delete_single.setFixedSize(65, 23)
        btn_delete_single.clicked.connect(self.OnDeleteSelectedMarkers)

        btn_delete_all = QPushButton(_("Delete all"), self)
        btn_delete_all.setFixedSize(135, 23)
        btn_delete_all.clicked.connect(self.OnDeleteAllMarkers)

        sizer_delete = QHBoxLayout()
        sizer_delete.setSpacing(5)
        sizer_delete.addWidget(btn_delete_single)
        sizer_delete.addWidget(btn_delete_all)

        self.select_main_coil = select_main_coil = QComboBox(self)
        select_main_coil.setFixedWidth(145)
        select_main_coil.addItems(list(self.navigation.coil_registrations))
        maincoil_tooltip = "Select which coil to record markers with"
        select_main_coil.setToolTip(maincoil_tooltip)
        select_main_coil.currentIndexChanged.connect(self.OnChooseMainCoil)

        nav_state = self.session.GetConfig("navigation", {})
        if (main_coil := nav_state.get("main_coil", None)) is not None:
            main_coil_index = select_main_coil.findText(main_coil)
            select_main_coil.setCurrentIndex(main_coil_index)

        select_main_coil.setVisible(nav_state.get("n_coils", 1) != 1)

        sizer_main_coil = QHBoxLayout()
        sizer_main_coil.addWidget(select_main_coil)

        screen = QApplication.primaryScreen()
        if screen:
            screen_height = screen.size().height()
        else:
            screen_height = 1080

        marker_list_height = max(120, int(screen_height / 4))
        self.marker_list_height = marker_list_height

        marker_list_ctrl = QTreeWidget(self)
        marker_list_ctrl.setRootIsDecorated(False)
        marker_list_ctrl.setSelectionMode(QAbstractItemView.ExtendedSelection)
        marker_list_ctrl.setFixedHeight(marker_list_height)

        columns = []
        col_widths = []

        columns.append("#")
        col_widths.append(24)
        columns.append("Session")
        col_widths.append(51)
        columns.append("Type")
        col_widths.append(77)
        columns.append("Label")
        col_widths.append(95)
        columns.append("Target")
        col_widths.append(45)
        columns.append("Z-offset")
        col_widths.append(45)
        columns.append("Efield Target")
        col_widths.append(45)
        columns.append("MEP (uV)")
        col_widths.append(45)
        columns.append("UUID")
        col_widths.append(45)

        if self.session.GetConfig("debug"):
            columns.append("X")
            col_widths.append(45)
            columns.append("Y")
            col_widths.append(45)
            columns.append("Z")
            col_widths.append(45)

        marker_list_ctrl.setHeaderLabels(columns)
        for i, w in enumerate(col_widths):
            marker_list_ctrl.setColumnWidth(i, w)

        marker_list_ctrl.setSortingEnabled(True)
        marker_list_ctrl.setContextMenuPolicy(Qt.CustomContextMenu)
        marker_list_ctrl.customContextMenuRequested.connect(self.OnMouseRightDown)
        marker_list_ctrl.itemSelectionChanged.connect(self._onSelectionChanged)
        marker_list_ctrl.itemDoubleClicked.connect(self._onItemDoubleClicked)

        self.marker_list_ctrl = marker_list_ctrl

        brain_targets_list_ctrl = QTreeWidget(self)
        brain_targets_list_ctrl.setRootIsDecorated(False)
        brain_targets_list_ctrl.setFixedHeight(marker_list_height)

        brain_columns = [
            "#",
            "Session",
            "Type",
            "Label",
            "MEP (uV)",
            "X (mm)",
            "Y (mm)",
            "R (°)",
            "Int. (V/m)",
            "UUID",
        ]
        brain_col_widths = [26, 51, 77, 95, 45, 45, 45, 45, 45, 45]
        brain_targets_list_ctrl.setHeaderLabels(brain_columns)
        for i, w in enumerate(brain_col_widths):
            brain_targets_list_ctrl.setColumnWidth(i, w)
        brain_targets_list_ctrl.hide()

        brain_targets_list_ctrl.setContextMenuPolicy(Qt.CustomContextMenu)
        brain_targets_list_ctrl.customContextMenuRequested.connect(
            self.OnMouseRightDownBrainTargets
        )
        self.brain_targets_list_ctrl = brain_targets_list_ctrl

        try:
            self.markers.LoadState()
        except:
            self.session.DeleteStateFile()

        group_layout = QVBoxLayout(self)
        group_layout.addLayout(sizer_create)
        group_layout.addLayout(sizer_btns)
        group_layout.addLayout(sizer_delete)
        group_layout.addLayout(sizer_main_coil)
        group_layout.addWidget(marker_list_ctrl)
        group_layout.addWidget(brain_targets_list_ctrl)

    def GetListCtrl(self):
        return self.marker_list_ctrl

    def __bind_events(self):
        Publisher.subscribe(self.UpdateCurrentCoord, "Set cross focal point")

        Publisher.subscribe(self.OnSelectMarkerByActor, "Select marker by actor")

        Publisher.subscribe(self.OnDeleteFiducialMarker, "Delete fiducial marker")
        Publisher.subscribe(self.OnDeleteSelectedMarkers, "Delete selected markers")
        Publisher.subscribe(self.OnDeleteAllMarkers, "Delete all markers")
        Publisher.subscribe(self.OnCreateMarker, "Create marker")
        Publisher.subscribe(self.UpdateNavigationStatus, "Navigation status")
        Publisher.subscribe(self.UpdateSeedCoordinates, "Update tracts")
        Publisher.subscribe(self.OnChangeCurrentSession, "Current session changed")
        Publisher.subscribe(self.UpdateMarker, "Update marker")
        Publisher.subscribe(self.UpdateMarkerOrientation, "Open marker orientation dialog")
        Publisher.subscribe(self.AddPeeledSurface, "Update peel")
        Publisher.subscribe(self.GetEfieldDataStatus, "Get status of Efield saved data")
        Publisher.subscribe(self.GetIdList, "Get ID list")
        Publisher.subscribe(self.GetRotationPosition, "Send coil position and rotation")
        Publisher.subscribe(self.CreateMarkerEfield, "Create Marker from tangential")
        Publisher.subscribe(self.UpdateCortexMarker, "Update Cortex Marker")
        Publisher.subscribe(
            self.UpdateCoilTarget, "NeuroSimo to Neuronavigation: Update coil target"
        )

        Publisher.subscribe(self.CreateCoilTargetFromLandmark, "Create coil target from landmark")

        Publisher.subscribe(self.UpdateMainCoilCombobox, "Coil selection done")

        Publisher.subscribe(self._AddMarker, "Add marker")
        Publisher.subscribe(self._DeleteMarker, "Delete marker")
        Publisher.subscribe(self._DeleteMultiple, "Delete markers")
        Publisher.subscribe(self._DuplicateMarker, "Duplicate marker")
        Publisher.subscribe(self._SetPointOfInterest, "Set point of interest")
        Publisher.subscribe(self._SetTarget, "Set target")
        Publisher.subscribe(self._UnsetTarget, "Unset target")
        Publisher.subscribe(self._UnsetPointOfInterest, "Unset point of interest")
        Publisher.subscribe(self._UpdateMarkerLabel, "Update marker label")
        Publisher.subscribe(self._UpdateMEP, "Update marker mep")

        Publisher.subscribe(self.SetBrainTarget, "Set brain targets")

    def __get_selected_items(self):
        return [
            self.marker_list_ctrl.indexOfTopLevelItem(item)
            for item in self.marker_list_ctrl.selectedItems()
        ]

    def __delete_multiple_markers(self, indexes):
        marker_ids = [self.__get_marker_id(idx) for idx in indexes]
        self.markers.DeleteMultiple(marker_ids)

    def _DeleteMarker(self, marker):
        deleted_marker_id = marker.marker_id
        deleted_marker_uuid = marker.marker_uuid
        idx = self.__find_marker_index(deleted_marker_id)
        self.marker_list_ctrl.takeTopLevelItem(idx)
        print("_DeleteMarker:", deleted_marker_uuid)

        for key, data in list(self.itemDataMap.items()):
            current_uuid = data[-1]
            if current_uuid == deleted_marker_uuid:
                self.itemDataMap.pop(key)

        num_items = self.marker_list_ctrl.topLevelItemCount()
        for n in range(num_items):
            m_id = self.__get_marker_id(n)
            if m_id > deleted_marker_id:
                self.marker_list_ctrl.topLevelItem(n).setText(const.ID_COLUMN, str(m_id - 1))

    def _DeleteMultiple(self, markers):
        if len(markers) == self.marker_list_ctrl.topLevelItemCount():
            self.marker_list_ctrl.clear()
            self.itemDataMap.clear()
            return

        min_for_fast_deletion = 10
        if len(markers) > min_for_fast_deletion:
            self.marker_list_ctrl.hide()

        deleted_ids = []
        deleted_keys = []
        for marker in markers:
            idx = self.__find_marker_index(marker.marker_id)
            if idx is None:
                continue
            deleted_uuid = marker.marker_uuid
            for key, data in self.itemDataMap.items():
                current_uuid = data[-1]
                if current_uuid == deleted_uuid:
                    deleted_keys.append(key)

            self.marker_list_ctrl.takeTopLevelItem(idx)
            deleted_ids.append(marker.marker_id)

        for key in deleted_keys:
            try:
                self.itemDataMap.pop(key)
            except KeyError:
                print("Invalid itemDataMap key:", key)

        for idx in range(self.marker_list_ctrl.topLevelItemCount()):
            self.marker_list_ctrl.topLevelItem(idx).setText(const.ID_COLUMN, str(idx))

        self.marker_list_ctrl.show()

    def _SetPointOfInterest(self, marker):
        idx = self.__find_marker_index(marker.marker_id)
        item = self.marker_list_ctrl.topLevelItem(idx)
        _set_tree_item_bg(item, QColor("purple"), self.marker_list_ctrl.columnCount())
        item.setText(const.POINT_OF_INTEREST_TARGET_COLUMN, _("Yes"))
        uuid_val = marker.marker_uuid

        for key, data in self.itemDataMap.items():
            current_uuid = data[-1]
            if current_uuid == uuid_val:
                self.itemDataMap[key][const.POINT_OF_INTEREST_TARGET_COLUMN] = "Yes"

    def _UnsetPointOfInterest(self, marker):
        idx = self.__find_marker_index(marker.marker_id)
        item = self.marker_list_ctrl.topLevelItem(idx)

        _set_tree_item_bg(item, QColor("white"), self.marker_list_ctrl.columnCount())
        item.setText(const.POINT_OF_INTEREST_TARGET_COLUMN, "")
        uuid_val = marker.marker_uuid

        for key, data in self.itemDataMap.items():
            current_uuid = data[-1]
            if current_uuid == uuid_val:
                self.itemDataMap[key][const.POINT_OF_INTEREST_TARGET_COLUMN] = ""

    def _UpdateMarkerLabel(self, marker):
        idx = self.__find_marker_index(marker.marker_id)
        self.marker_list_ctrl.topLevelItem(idx).setText(const.LABEL_COLUMN, marker.label)

        uuid_val = marker.marker_uuid
        for key, data in self.itemDataMap.items():
            current_uuid = data[-1]
            if current_uuid == uuid_val:
                self.itemDataMap[key][const.LABEL_COLUMN] = marker.label

    def _UpdateMEP(self, marker):
        idx = self.__find_marker_index(marker.marker_id)
        self.marker_list_ctrl.topLevelItem(idx).setText(const.MEP_COLUMN, str(marker.mep_value))

        uuid_val = marker.marker_uuid
        for key, data in self.itemDataMap.items():
            current_uuid = data[-1]
            if current_uuid == uuid_val:
                self.itemDataMap[key][const.MEP_COLUMN] = marker.mep_value

        Publisher.sendMessage("Redraw MEP mapping")

    @staticmethod
    def __list_fiducial_labels():
        return list(
            itertools.chain(*(const.BTNS_IMG_MARKERS[i].values() for i in const.BTNS_IMG_MARKERS))
        )

    def UpdateCurrentCoord(self, position):
        self.current_position = list(position[:3])
        self.current_orientation = list(position[3:])
        if not self.navigation.track_coil:
            self.current_orientation = None, None, None

    def UpdateNavigationStatus(self, nav_status, vis_status):
        if not nav_status:
            self.nav_status = False
            self.current_orientation = None, None, None
        else:
            self.nav_status = True

    def UpdateSeedCoordinates(
        self, root=None, affine_vtk=None, coord_offset=(0, 0, 0), coord_offset_w=(0, 0, 0)
    ):
        self.current_seed = coord_offset_w

    def UpdateCortexMarker(self, CoGposition, CoGorientation):
        self.cortex_position_orientation = CoGposition + CoGorientation

    def UpdateCoilTarget(self, coil_target):
        markers = self.markers.FindLabel(coil_target)
        if markers:
            QTimer.singleShot(0, lambda: Publisher.sendMessage("Press robot button", pressed=False))
            for marker in markers:
                if marker.marker_type == MarkerType.COIL_TARGET:
                    self.markers.SetTarget(marker.marker_id)
                    QTimer.singleShot(
                        0, lambda: Publisher.sendMessage("Press robot button", pressed=True)
                    )
                    return

            self.markers.CreateCoilTargetFromLandmark(markers[0], markers[0].label)
            self.markers.SetTarget(-1)
            QTimer.singleShot(0, lambda: Publisher.sendMessage("Press robot button", pressed=True))
        return

    def SetBrainTarget(self, brain_targets):
        marker_target = self.markers.FindTarget()
        if not marker_target:
            return

        position = marker_target.position
        orientation = marker_target.orientation
        position[1] = -position[1]
        m_marker_target = dco.coordinates_to_transformation_matrix(
            position=position,
            orientation=orientation,
            axes="sxyz",
        )

        for target in brain_targets:
            m_offset_brain = dco.coordinates_to_transformation_matrix(
                position=target["position"],
                orientation=target["orientation"],
                axes="sxyz",
            )
            m_brain = m_marker_target @ m_offset_brain
            new_position, new_orientation = dco.transformation_matrix_to_coordinates(
                m_brain, "sxyz"
            )
            new_position[1] = -new_position[1]
            marker = self.CreateMarker(
                position=new_position.tolist(),
                orientation=new_orientation.tolist(),
                colour=target["color"],
                size=target["length"],
                label=str(marker_target.label),
                marker_type=MarkerType.BRAIN_TARGET,
            )
            marker.marker_uuid = str(uuid.uuid4())
            marker.x_mtms = target["mtms"][0]
            marker.y_mtms = target["mtms"][1]
            marker.r_mtms = target["mtms"][2]
            marker.intensity_mtms = target["mtms"][3]
            marker.mep_value = target["mep"] or None
            marker_target.brain_target_list.append(marker.to_brain_targets_dict())

        idx = self.__find_marker_index(marker_target.marker_id)
        item = self.marker_list_ctrl.topLevelItem(idx)
        _set_tree_item_bg(item, QColor(246, 226, 182), self.marker_list_ctrl.columnCount())
        Publisher.sendMessage(
            "Redraw MEP mapping from brain targets",
            marker_target=marker_target,
            brain_target_list=marker_target.brain_target_list,
        )
        self.markers.SaveState()

    def OnMouseRightDown(self, pos):
        current_item = self.marker_list_ctrl.currentItem()
        if current_item is None:
            return
        focused_marker_idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item)
        if focused_marker_idx == -1:
            return
        focused_marker = self.__get_marker(focused_marker_idx)
        marker_type = focused_marker.marker_type

        is_active_target = focused_marker.is_target
        is_coil_target = marker_type == MarkerType.COIL_TARGET
        is_coil_pose = marker_type == MarkerType.COIL_POSE
        is_landmark = marker_type == MarkerType.LANDMARK
        is_fiducial = marker_type == MarkerType.FIDUCIAL

        menu = QMenu(self)

        edit_action = menu.addAction(_("Change label"))
        edit_action.triggered.connect(self.ChangeLabel)

        color_action = menu.addAction(_("Change color"))
        color_action.triggered.connect(self.ChangeColor)

        delete_action = menu.addAction(_("Delete"))
        delete_action.triggered.connect(self.OnDeleteSelectedMarkers)

        if not is_fiducial:
            duplicate_action = menu.addAction(_("Duplicate"))
            duplicate_action.triggered.connect(self.OnMenuDuplicateMarker)

        menu.addSeparator()

        if is_coil_target:
            mep_action = menu.addAction(_("Change MEP value"))
            mep_action.triggered.connect(self.OnMenuChangeMEP)
            if is_active_target:
                target_action = menu.addAction(_("Unset target"))
                target_action.triggered.connect(self.OnMenuUnsetTarget)
                if has_mTMS:
                    brain_target_action = menu.addAction(_("Set brain target"))
                    brain_target_action.triggered.connect(self.OnSetBrainTarget)
            else:
                target_action = menu.addAction(_("Set as target"))
                target_action.triggered.connect(self.OnMenuSetTarget)

        if is_coil_pose:
            create_coil_target_action = menu.addAction(_("Create coil target"))
            create_coil_target_action.triggered.connect(self.OnCreateCoilTargetFromCoilPose)

        if is_landmark:
            create_brain_target_action = menu.addAction(_("Create brain target"))
            create_brain_target_action.triggered.connect(self.OnCreateBrainTargetFromLandmark)

            create_coil_target_action = menu.addAction(_("Create coil target"))
            create_coil_target_action.triggered.connect(self.OnCreateCoilTargetFromLandmark)

        is_brain_target = focused_marker.marker_type == MarkerType.BRAIN_TARGET
        if is_brain_target and has_mTMS:
            send_brain_target_action = menu.addAction(_("Send brain target to mTMS"))
            send_brain_target_action.triggered.connect(self.OnSendBrainTarget)

        if self.nav_status and self.navigation.e_field_loaded:
            if is_active_target:
                efield_action = menu.addAction(_("Save Efield target Data"))
                efield_action.triggered.connect(self.OnMenuSaveEfieldTargetData)

        if self.navigation.e_field_loaded:
            clear_efield_action = menu.addAction(_("Clear saved Efield data"))
            clear_efield_action.triggered.connect(self.OnClearEfieldSavedData)

            efield_target_action = menu.addAction(_("Set as Efield target 1 (origin)"))
            efield_target_action.triggered.connect(self.OnMenuSetEfieldTarget)

            efield_target2_action = menu.addAction(_("Set as Efield target 2"))
            efield_target2_action.triggered.connect(self.OnMenuSetEfieldTarget2)

        if self.navigation.e_field_loaded and not self.nav_status:
            if is_active_target:
                efield_vector_action = menu.addAction(_("Show vector field"))
                efield_vector_action.triggered.connect(self.OnMenuShowVectorField)

        if self.navigation.e_field_loaded:
            if focused_marker.is_point_of_interest:
                cortex_action = menu.addAction(_("Remove Efield Cortex target"))
                cortex_action.triggered.connect(self.OnMenuRemoveEfieldTargetatCortex)
            else:
                cortex_action = menu.addAction(_("Set as Efield Cortex target"))
                cortex_action.triggered.connect(self.OnSetEfieldBrainTarget)

        menu.addSeparator()

        menu.exec_(self.marker_list_ctrl.viewport().mapToGlobal(pos))

    def OnMouseRightDownBrainTargets(self, pos):
        current_item = self.brain_targets_list_ctrl.currentItem()
        if current_item is None:
            return
        focused_marker_idx = self.brain_targets_list_ctrl.indexOfTopLevelItem(current_item)
        if focused_marker_idx == -1:
            return
        focused_marker = self.currently_focused_marker.brain_target_list[focused_marker_idx]
        self.focused_brain_marker = focused_marker

        menu = QMenu(self)

        edit_action = menu.addAction(_("Change label"))
        edit_action.triggered.connect(self.ChangeLabelBrainTarget)

        delete_action = menu.addAction(_("Delete"))
        delete_action.triggered.connect(self.OnDeleteSelectedBrainTarget)

        menu.addSeparator()

        mep_action = menu.addAction(_("Change MEP value"))
        mep_action.triggered.connect(self.OnMenuChangeMEPBrainTarget)

        create_coil_target_action = menu.addAction(_("Create coil target"))
        create_coil_target_action.triggered.connect(self.OnCreateCoilTargetFromBrainTargets)

        if has_mTMS:
            send_brain_target_action = menu.addAction(_("Send brain target to mTMS"))
            send_brain_target_action.triggered.connect(self.OnSendBrainTarget)

        menu.addSeparator()

        menu.exec_(self.brain_targets_list_ctrl.viewport().mapToGlobal(pos))

    def FocusOnMarker(self, idx):
        if self.currently_focused_marker is not None:
            current_marker_idx = self.__find_marker_index(self.currently_focused_marker.marker_id)
            if current_marker_idx is not None:
                old_item = self.marker_list_ctrl.topLevelItem(current_marker_idx)
                if old_item:
                    old_item.setSelected(False)

        item = self.marker_list_ctrl.topLevelItem(idx)
        if item:
            self.marker_list_ctrl.setCurrentItem(item)
            item.setSelected(True)
            self.marker_list_ctrl.scrollToItem(item)
            self._handleMarkerFocused(idx)

    def populate_sub_list(self, sub_items_list):
        self.brain_targets_list_ctrl.clear()
        current_item = self.marker_list_ctrl.currentItem()
        if current_item is None:
            return
        focused_marker_idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item)
        marker = self.__get_marker(focused_marker_idx)
        num_items = focused_marker_idx
        brain_targets = []
        for i, sub_item in enumerate(sub_items_list):
            list_entry = ["" for _ in range(0, const.BRAIN_UUID + 1)]
            list_entry[const.BRAIN_ID_COLUMN] = str(num_items) + "." + str(i)
            list_entry[const.BRAIN_SESSION_COLUMN] = str(marker.brain_target_list[i]["session_id"])
            list_entry[const.BRAIN_MARKER_TYPE_COLUMN] = MarkerType.BRAIN_TARGET.human_readable
            list_entry[const.BRAIN_LABEL_COLUMN] = marker.brain_target_list[i]["label"]
            list_entry[const.BRAIN_MEP_COLUMN] = (
                str(marker.brain_target_list[i]["mep_value"])
                if marker.brain_target_list[i]["mep_value"]
                else ""
            )
            list_entry[const.BRAIN_X_MTMS] = str(marker.brain_target_list[i]["x_mtms"])
            list_entry[const.BRAIN_Y_MTMS] = str(marker.brain_target_list[i]["y_mtms"])
            list_entry[const.BRAIN_R_MTMS] = str(marker.brain_target_list[i]["r_mtms"])
            list_entry[const.BRAIN_INTENSITY_MTMS] = str(
                marker.brain_target_list[i]["intensity_mtms"]
            )
            list_entry[const.BRAIN_UUID] = (
                str(marker.brain_target_list[i]["marker_uuid"])
                if marker.brain_target_list[i]["marker_uuid"]
                else ""
            )
            tree_item = QTreeWidgetItem([str(e) for e in list_entry])
            self.brain_targets_list_ctrl.addTopLevelItem(tree_item)
            x, y, z = marker.brain_target_list[i]["position"]
            brain_targets.append(
                {
                    "position": [x, -y, z],
                    "orientation": marker.brain_target_list[i]["orientation"],
                    "color": marker.brain_target_list[i]["colour"],
                    "length": marker.brain_target_list[i]["size"],
                }
            )
        Publisher.sendMessage("Update brain targets", brain_targets=brain_targets)

    def ResizeListCtrl(self, width):
        self.brain_targets_list_ctrl.setFixedHeight(int(width))
        self.marker_list_ctrl.setFixedHeight(int(width))

    def _onSelectionChanged(self):
        current_item = self.marker_list_ctrl.currentItem()
        if current_item is None:
            self.markers.DeselectMarker()
            return
        idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item)
        if idx == -1:
            self.markers.DeselectMarker()
            return
        if current_item.isSelected():
            self._handleMarkerFocused(idx)
        else:
            self.markers.DeselectMarker()

    def _handleMarkerFocused(self, idx):
        if idx == -1:
            return

        marker_id = self.__get_marker_id(idx)
        marker = self.__get_marker(idx)

        if self.currently_focused_marker is not None:
            Publisher.sendMessage("Unhighlight marker")

        self.currently_focused_marker = marker
        self.markers.SelectMarker(marker_id)
        self.brain_targets_list_ctrl.clear()
        if marker.brain_target_list:
            Publisher.sendMessage("Set vector field assembly visibility", enabled=True)
            self.populate_sub_list(marker.brain_target_list)
            self.brain_targets_list_ctrl.show()
            width = self.marker_list_height / 2
            Publisher.sendMessage(
                "Redraw MEP mapping from brain targets",
                marker_target=marker,
                brain_target_list=marker.brain_target_list,
            )
        else:
            Publisher.sendMessage("Set vector field assembly visibility", enabled=False)
            self.brain_targets_list_ctrl.hide()
            width = self.marker_list_height
        self.ResizeListCtrl(width)
        Publisher.sendMessage("Update navigation panel")
        self.update()

    def _onItemDoubleClicked(self, item, column):
        idx = self.marker_list_ctrl.indexOfTopLevelItem(item)
        if idx == -1:
            return
        marker = self.__get_marker(idx)
        Publisher.sendMessage("Set camera to focus on marker", marker=marker)

    def OnMarkerFocused(self, evt=None):
        current_item = self.marker_list_ctrl.currentItem()
        if current_item is None:
            return
        idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item)
        self._handleMarkerFocused(idx)

    def OnMarkerUnfocused(self, evt=None):
        self.markers.DeselectMarker()

    def SetCameraToFocusOnMarker(self, evt=None):
        current_item = self.marker_list_ctrl.currentItem()
        if current_item is None:
            return
        idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item)
        marker = self.__get_marker(idx)
        Publisher.sendMessage("Set camera to focus on marker", marker=marker)

    def CreateCoilTargetFromLandmark(self, index=None, label=None):
        if index:
            self.FocusOnMarker(index)
        self.OnCreateCoilTargetFromLandmark(label=label)

    def OnCreateCoilTargetFromLandmark(self, evt=None, label=None):
        current_item = self.marker_list_ctrl.currentItem()
        list_index = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        if list_index == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return
        marker = self.__get_marker(list_index)

        proj = prj.Project()
        if not proj.surface_dict:
            QMessageBox.information(self, _("InVesalius 3"), _("No 3D surface was created."))
            return
        self.markers.CreateCoilTargetFromLandmark(marker, label)

    def OnCreateCoilTargetFromBrainTargets(self):
        self.markers.CreateCoilTargetFromBrainTarget(self.focused_brain_marker)

    def OnCreateCoilTargetFromCoilPose(self):
        current_item = self.marker_list_ctrl.currentItem()
        list_index = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        if list_index == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return
        marker = self.__get_marker(list_index)

        self.markers.CreateCoilTargetFromCoilPose(marker)

    def UpdateMainCoilCombobox(self, done):
        select_main_coil = self.select_main_coil
        if done:
            select_main_coil.clear()
            select_main_coil.addItems(list(self.navigation.coil_registrations))
            main_coil_index = select_main_coil.findText(self.navigation.main_coil)
            select_main_coil.setCurrentIndex(main_coil_index)
        else:
            select_main_coil.clear()

        if self.navigation.n_coils == 1:
            select_main_coil.hide()
        else:
            select_main_coil.show()
        self.updateGeometry()

    def OnChooseMainCoil(self, choice):
        main_coil = self.select_main_coil.itemText(choice)
        self.navigation.SetMainCoil(main_coil)
        self.select_main_coil.setCurrentIndex(choice)

    def ChangeLabel(self):
        current_item = self.marker_list_ctrl.currentItem()
        list_index = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        if list_index == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return
        marker = self.__get_marker(list_index)
        new_label = dlg.ShowEnterMarkerID(
            self.marker_list_ctrl.topLevelItem(list_index).text(const.LABEL_COLUMN)
        )
        self.markers.ChangeLabel(marker, new_label)

    def ChangeLabelBrainTarget(self):
        current_item = self.brain_targets_list_ctrl.currentItem()
        list_index = (
            self.brain_targets_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        )
        if list_index == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return
        marker = self.currently_focused_marker.brain_target_list[list_index]
        marker["label"] = dlg.ShowEnterMarkerID(marker["label"])
        self.brain_targets_list_ctrl.topLevelItem(list_index).setText(
            const.BRAIN_LABEL_COLUMN, marker["label"]
        )
        self.markers.SaveState()

    def OnMenuSetTarget(self):
        current_item = self.marker_list_ctrl.currentItem()
        idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        if idx == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return
        if not self.navigation.coil_registrations:
            QMessageBox.information(self, _("InVesalius 3"), _("TMS coil not registered."))
            return

        marker_id = self.__get_marker_id(idx)
        self.markers.SetTarget(marker_id)

    def _SetTarget(self, marker):
        idx = self.__find_marker_index(marker.marker_id)
        item = self.marker_list_ctrl.topLevelItem(idx)
        _set_tree_item_bg(item, QColor(255, 220, 209), self.marker_list_ctrl.columnCount())
        item.setText(const.TARGET_COLUMN, _("Yes"))

        target_uuid = marker.marker_uuid

        for key, data in self.itemDataMap.items():
            current_uuid = data[-1]
            if current_uuid == target_uuid:
                self.itemDataMap[key][const.TARGET_COLUMN] = "Yes"

    def _DuplicateMarker(
        self, marker_idx: Optional[int] = None, duplicate_brain_target_list: bool = True
    ) -> None:
        set_target = False
        if marker_idx is None:
            target = self.markers.FindTarget()
            if target:
                marker_idx = self.__find_marker_index(target.marker_id)
                set_target = marker_idx is not None

        item_count = self.marker_list_ctrl.topLevelItemCount()
        if marker_idx is None or not (0 <= marker_idx < item_count):
            QMessageBox.information(self, _("InVesalius 3"), _("Marker index not valid."))
            return

        marker = self.__get_marker(marker_idx)
        new_marker = marker.duplicate()

        new_marker.label = f"{new_marker.label} (copy)"
        if not duplicate_brain_target_list:
            new_marker.brain_target_list = []

        self.markers.AddMarker(new_marker, render=True, focus=True)

        if set_target:
            current_target = self.markers.FindTarget()
            if current_target:
                self.markers.UnsetTarget(current_target.marker_id)
            self.markers.SetTarget(new_marker.marker_id)

    def OnMenuDuplicateMarker(self):
        current_item = self.marker_list_ctrl.currentItem()
        marker_idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        if marker_idx == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return
        self._DuplicateMarker(marker_idx)

    def GetEfieldDataStatus(self, efield_data_loaded, indexes_saved_list):
        self.indexes_saved_lists = []
        self.efield_data_saved = efield_data_loaded
        self.indexes_saved_lists = indexes_saved_list

    def CreateMarkerEfield(self, point, orientation):
        from vtkmodules.vtkCommonColor import vtkNamedColors

        vtk_colors = vtkNamedColors()
        position_flip = list(point)
        position_flip[1] = -position_flip[1]

        marker = self.CreateMarker(
            position=position_flip,
            orientation=list(orientation),
            colour=vtk_colors.GetColor3d("Orange"),
            size=2,
            marker_type=MarkerType.COIL_TARGET,
        )
        self.markers.AddMarker(marker, render=True, focus=True)

    def OnMenuShowVectorField(self):
        import invesalius.data.imagedata_utils as imagedata_utils
        import invesalius.data.transformations as tr

        session = ses.Session()
        current_item = self.marker_list_ctrl.currentItem()
        idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        marker = self.__get_marker(idx)
        position = marker.position
        orientation = marker.orientation
        coord = np.concatenate([np.asarray(position, float), np.asarray(orientation, float)])

        m_img = tr.compose_matrix(angles=np.radians(orientation), translate=position)

        position, orientation = imagedata_utils.convert_invesalius_to_world(
            position=position,
            orientation=orientation,
        )
        Publisher.sendMessage(
            "Calculate position and rotation", position=position, orientation=orientation
        )

        Publisher.sendMessage(
            "Update interseccion offline",
            m_img=m_img,
            coord=np.concatenate(
                [np.asarray(position, float), np.radians(np.asarray(orientation, float))]
            ),
            list_index=marker.marker_id,
        )

        if session.GetConfig("debug_efield"):
            enorm = self.navigation.debug_efield_enorm
        else:
            enorm = self.navigation.neuronavigation_api.update_efield_vectorROI(
                position=self.cp, orientation=orientation, T_rot=self.T_rot, id_list=self.ID_list
            )
        enorm_data = [self.T_rot, self.cp, coord, enorm, self.ID_list]
        Publisher.sendMessage("Get enorm", enorm_data=enorm_data, plot_vector=True)
        plot_efield_vectors = self.navigation.plot_efield_vectors
        Publisher.sendMessage(
            "Save target data",
            target_list_index=marker.marker_id,
            position=position,
            orientation=orientation,
            plot_efield_vectors=plot_efield_vectors,
        )

    def GetRotationPosition(self, T_rot, cp, m_img):
        self.T_rot = T_rot
        self.cp = cp
        self.m_img_offline = m_img

    def GetIdList(self, ID_list):
        self.ID_list = ID_list

    def OnMenuSetEfieldTarget(self):
        current_item = self.marker_list_ctrl.currentItem()
        idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        if idx == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return
        marker_id = self.__get_marker_id(idx)
        self.markers.SetTarget(marker_id)
        self.efield_target_idx_origin = marker_id

    def OnMenuSetEfieldTarget2(self):
        current_item = self.marker_list_ctrl.currentItem()
        idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        if idx == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return

        efield_target_idx_2 = self.__get_marker_id(idx)
        target1_origin = self.markers.list[
            self.efield_target_idx_origin
        ].cortex_position_orientation
        target2 = self.markers.list[efield_target_idx_2].cortex_position_orientation
        Publisher.sendMessage(
            "Get targets Ids for mtms", target1_origin=target1_origin, target2=target2
        )

    def OnMenuSaveEfieldTargetData(self):
        current_item = self.marker_list_ctrl.currentItem()
        list_index = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        marker = self.__get_marker(list_index)
        position = marker.position
        orientation = marker.orientation
        plot_efield_vectors = self.navigation.plot_efield_vectors
        Publisher.sendMessage(
            "Save target data",
            target_list_index=marker.marker_id,
            position=position,
            orientation=orientation,
            plot_efield_vectors=plot_efield_vectors,
        )

    def OnClearEfieldSavedData(self):
        Publisher.sendMessage("Clear saved efield data")

    def OnSetEfieldBrainTarget(self):
        current_item = self.marker_list_ctrl.currentItem()
        idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        marker = self.__get_marker(idx)
        position = marker.position
        orientation = marker.orientation
        marker_id = marker.marker_id
        if all([o is None for o in orientation]):
            orientation = [0, 0, 0]

        self.markers.SetPointOfInterest(marker_id)
        Publisher.sendMessage(
            "Send efield target position on brain",
            marker_id=marker_id,
            position=position,
            orientation=orientation,
        )

    def transform_to_mtms(self, coil_position, coil_orientation_euler, brain_position):
        import invesalius.data.transformations as tr

        coil_position = np.array(coil_position)
        brain_position = np.array(brain_position)

        coil_rotation_matrix = tr.euler_matrix(
            coil_orientation_euler[0], coil_orientation_euler[1], coil_orientation_euler[2], "sxyz"
        )

        translated_position = brain_position - coil_position

        brain_position_in_coil_coords = np.dot(coil_rotation_matrix[:3, :3].T, translated_position)

        return brain_position_in_coil_coords

    def OnCreateBrainTargetFromLandmark(self):
        current_item = self.marker_list_ctrl.currentItem()
        list_index = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        marker_coil = self.__get_marker(list_index)
        position_coil = marker_coil.position
        orientation_coil = marker_coil.orientation

        dialog = dlg.CreateBrainTargetDialog(
            marker=position_coil + orientation_coil, brain_actor=self.brain_actor
        )
        if dialog.exec() == QDialog.Accepted:
            (
                coil_position_list,
                coil_orientation_list,
                brain_position_list,
                brain_orientation_list,
            ) = dialog.GetValue()

            position = list(coil_position_list[0])
            orientation = list(coil_orientation_list[0])
            marker = self.CreateMarker(
                position=position,
                orientation=orientation,
                marker_type=MarkerType.BRAIN_TARGET,
                size=1,
                label=str(marker_coil.label),
            )
            marker.marker_uuid = str(uuid.uuid4())
            mtms_coords = self.transform_to_mtms(position_coil, orientation, position)
            marker.x_mtms = np.round(mtms_coords[0], 1)
            marker.y_mtms = np.round(mtms_coords[1], 1)
            marker.r_mtms = np.round(orientation[2], 0)
            marker.intensity_mtms = 10
            marker_coil.brain_target_list.append(marker.to_brain_targets_dict())

            for position, orientation in zip(brain_position_list, brain_orientation_list):
                marker = self.CreateMarker(
                    position=list(position),
                    orientation=list(orientation),
                    marker_type=MarkerType.BRAIN_TARGET,
                    size=1,
                    label=str(marker_coil.label),
                )
                marker.marker_uuid = str(uuid.uuid4())
                mtms_coords = self.transform_to_mtms(position_coil, orientation, position)
                marker.x_mtms = np.round(mtms_coords[0], 1)
                marker.y_mtms = np.round(mtms_coords[1], 1)
                marker.r_mtms = np.round(orientation[2], 0)
                marker.intensity_mtms = 10
                marker_coil.brain_target_list.append(marker.to_brain_targets_dict())

        if marker_coil.brain_target_list:
            item = self.marker_list_ctrl.topLevelItem(list_index)
            _set_tree_item_bg(item, QColor(251, 243, 226), self.marker_list_ctrl.columnCount())
        self.OnMarkerFocused()
        self.markers.SaveState()
        dialog.close()

    def OnMenuRemoveEfieldTarget(self):
        current_item = self.marker_list_ctrl.currentItem()
        idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        marker_id = self.__get_marker_id(idx)

        self.markers.UnsetTarget(marker_id)

        self.efield_target_idx = None

    def OnMenuRemoveEfieldTargetatCortex(self):
        current_item = self.marker_list_ctrl.currentItem()
        idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        marker = self.__get_marker(idx)

        marker.marker_type = MarkerType.LANDMARK

        self.markers.UnsetPointOfInterest(marker.marker_id)
        Publisher.sendMessage("Clear efield target at cortex")

    def OnMenuUnsetTarget(self):
        current_item = self.marker_list_ctrl.currentItem()
        idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        marker_id = self.__get_marker_id(idx)
        self.markers.UnsetTarget(marker_id)

    def OnMenuChangeMEP(self):
        current_item = self.marker_list_ctrl.currentItem()
        idx = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        marker = self.__get_marker(idx)

        new_mep = dlg.ShowEnterMEPValue(
            self.marker_list_ctrl.topLevelItem(idx).text(const.MEP_COLUMN)
        )
        self.markers.ChangeMEP(marker, new_mep)

    def OnMenuChangeMEPBrainTarget(self):
        current_item = self.brain_targets_list_ctrl.currentItem()
        list_index = (
            self.brain_targets_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        )
        if list_index == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return
        marker = self.currently_focused_marker.brain_target_list[list_index]
        if not marker["mep_value"]:
            marker["mep_value"] = "0"
        marker["mep_value"] = dlg.ShowEnterMEPValue(str(marker["mep_value"]))
        self.brain_targets_list_ctrl.topLevelItem(list_index).setText(
            const.BRAIN_MEP_COLUMN, str(marker["mep_value"])
        )
        Publisher.sendMessage(
            "Redraw MEP mapping from brain targets",
            marker_target=marker,
            brain_target_list=self.currently_focused_marker.brain_target_list,
        )

    def _UnsetTarget(self, marker):
        idx = self.__find_marker_index(marker.marker_id)

        Publisher.sendMessage("Press target mode button", pressed=False)

        item = self.marker_list_ctrl.topLevelItem(idx)
        if marker.brain_target_list:
            _set_tree_item_bg(item, QColor(251, 243, 226), self.marker_list_ctrl.columnCount())
        else:
            _set_tree_item_bg(item, QColor("white"), self.marker_list_ctrl.columnCount())
        item.setText(const.TARGET_COLUMN, "")

        target_uuid = marker.marker_uuid
        for key, data in self.itemDataMap.items():
            current_uuid = data[-1]
            if current_uuid == target_uuid:
                self.itemDataMap[key][const.TARGET_COLUMN] = ""

    def __find_marker_index(self, marker_id):
        num_items = self.marker_list_ctrl.topLevelItemCount()
        for idx in range(num_items):
            item_marker_id = self.__get_marker_id(idx)
            if item_marker_id == marker_id:
                return idx
        return None

    def __get_marker_id(self, idx):
        item = self.marker_list_ctrl.topLevelItem(idx)
        if item is None:
            return int(idx)
        current_uuid = item.text(const.UUID)
        for marker in self.markers.list:
            if current_uuid == marker.marker_uuid:
                marker_id = self.markers.list.index(marker)
                return int(marker_id)
        return int(item.text(const.ID_COLUMN))

    def __get_marker(self, idx):
        marker_id = self.__get_marker_id(idx)
        return self.markers.list[marker_id]

    def ChangeColor(self):
        current_item = self.marker_list_ctrl.currentItem()
        index = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        if index == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return
        marker = self.__get_marker(index)

        current_color = marker.colour8bit
        new_color = dlg.ShowColorDialog(color_current=current_color)

        if not new_color:
            return

        self.markers.ChangeColor(marker, new_color)

    def OnSetBrainTarget(self):
        current_item = self.marker_list_ctrl.currentItem()
        index = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        if index == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return
        marker = self.__get_marker(index)

        position = marker.position
        orientation = marker.orientation
        dialog = dlg.CreateBrainTargetDialog(
            mTMS=self.mTMS,
            marker=position + orientation,
            brain_target=True,
            brain_actor=self.brain_actor,
        )

        if dialog.exec() == QDialog.Accepted:
            position_list, orientation_list = dialog.GetValueBrainTarget()
            for position, orientation in zip(position_list, orientation_list):
                new_marker = self.CreateMarker(
                    position=list(position),
                    orientation=list(orientation),
                    size=0.05,
                    marker_type=MarkerType.BRAIN_TARGET,
                )
                new_marker.marker_uuid = str(uuid.uuid4())
                new_marker.label = str(marker.label)
                marker.brain_target_list.append(new_marker.to_brain_targets_dict())
        self.markers.SaveState()
        dialog.close()

    def OnSendBrainTarget(self):
        current_item = self.marker_list_ctrl.currentItem()
        index = self.marker_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        if index == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return
        marker = self.__get_marker(index)
        brain_target = marker.position + marker.orientation
        target = self.markers.FindTarget()
        if target is not None:
            coil_pose = target.position + target.orientation
            if self.navigation.coil_at_target:
                self.mTMS.UpdateTarget(coil_pose, brain_target)
                print("Send brain target to mTMS API")
            else:
                print("The coil is not at the target")
        else:
            print("Target not set")

    def OnSessionChanged(self, value):
        Publisher.sendMessage("Current session changed", new_session_id=value)

    def OnSelectMarkerByActor(self, actor):
        for m, idx in zip(self.markers.list, range(len(self.markers.list))):
            visualization = m.visualization
            if visualization is None:
                continue

            if visualization.get("actor") == actor:
                current_uuid = m.marker_uuid
                for i in range(self.marker_list_ctrl.topLevelItemCount()):
                    item = self.marker_list_ctrl.topLevelItem(i)
                    if current_uuid == item.text(const.UUID):
                        idx = i

                self.marker_list_ctrl.setCurrentItem(self.marker_list_ctrl.topLevelItem(idx))
                self.marker_list_ctrl.topLevelItem(idx).setSelected(True)
                break

    def OnDeleteAllMarkers(self, evt=None):
        if evt is not None:
            result = dlg.ShowConfirmationDialog(msg=_("Delete all markers? Cannot be undone."))
            if result != QDialog.Accepted:
                return
        self.markers.Clear()
        self.itemDataMap.clear()
        Publisher.sendMessage("Set vector field assembly visibility", enabled=False)
        self.brain_targets_list_ctrl.clear()
        self.brain_targets_list_ctrl.hide()

    def OnDeleteFiducialMarker(self, label):
        indexes = []
        if label and (label in self.__list_fiducial_labels()):
            for id_n in range(self.marker_list_ctrl.topLevelItemCount()):
                item = self.marker_list_ctrl.topLevelItem(id_n)
                if item.text(const.LABEL_COLUMN) == label:
                    self.marker_list_ctrl.setCurrentItem(item)
                    indexes = [self.marker_list_ctrl.indexOfTopLevelItem(item)]

        self.__delete_multiple_markers(indexes)

    def OnDeleteSelectedMarkers(self, evt=None):
        indexes = self.__get_selected_items()

        if not indexes:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return

        msg = _("Delete marker?") if len(indexes) == 1 else _("Delete markers?")

        result = dlg.ShowConfirmationDialog(msg=msg)
        if result != QDialog.Accepted:
            return

        self.__delete_multiple_markers(indexes)

        remaining_count = self.marker_list_ctrl.topLevelItemCount()
        if remaining_count > 0:
            focus_index = min(indexes[0], remaining_count - 1)
            self.FocusOnMarker(focus_index)
        else:
            self.currently_focused_marker = None

    def OnDeleteSelectedBrainTarget(self):
        current_item = self.brain_targets_list_ctrl.currentItem()
        list_index = (
            self.brain_targets_list_ctrl.indexOfTopLevelItem(current_item) if current_item else -1
        )
        if list_index == -1:
            QMessageBox.information(self, _("InVesalius 3"), _("No data selected."))
            return
        brain_target_list = self.currently_focused_marker.brain_target_list
        target_uuid = self.brain_targets_list_ctrl.topLevelItem(list_index).text(const.BRAIN_UUID)
        markers = [
            marker for marker in brain_target_list if marker.get("marker_uuid") != target_uuid
        ]
        self.currently_focused_marker.brain_target_list = markers
        self.OnMarkerFocused()
        self.markers.SaveState()

    def GetNextMarkerLabel(self):
        return self.markers.GetNextMarkerLabel()

    def OnCreateMarker(
        self,
        evt=None,
        position=None,
        orientation=None,
        colour=None,
        size=None,
        label=None,
        is_target=False,
        seed=None,
        session_id=None,
        marker_type=None,
        cortex_position_orientation=None,
        mep_value=None,
    ):
        if label is None:
            label = self.GetNextMarkerLabel()

        if self.nav_status and self.navigation.e_field_loaded:
            Publisher.sendMessage("Get Cortex position")

        if marker_type is None:
            marker_type = (
                MarkerType.COIL_TARGET
                if self.nav_status and self.navigation.track_coil
                else MarkerType.LANDMARK
            )
        if not self.nav_status and orientation is None:
            marker_type = MarkerType.LANDMARK
            orientation = None, None, None

        marker = self.CreateMarker(
            position=position,
            orientation=orientation,
            colour=colour,
            size=size,
            label=label,
            is_target=is_target,
            seed=seed,
            session_id=session_id,
            marker_type=marker_type,
            cortex_position_orientation=cortex_position_orientation,
            mep_value=mep_value,
        )
        self.markers.AddMarker(marker, render=True, focus=True)

    def ParseValue(self, value):
        value = value.strip()

        if value == "None":
            return None
        if value == "True":
            return True
        if value == "False":
            return False
        if value == "[]":
            return []

        if value.startswith("[") and value.endswith("]"):
            return self._parse_list(value)
        if value.startswith("{") and value.endswith("}"):
            return self._parse_dict(value)

        try:
            if "." in value or "e" in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            if (value.startswith('"') and value.endswith('"')) or (
                value.startswith("'") and value.endswith("'")
            ):
                return value[1:-1]
            return value

    def _parse_list(self, list_str):
        return [
            self.ParseValue(el.strip())
            for el in self._split_by_outer_commas(list_str[1:-1].strip())
        ]

    def _parse_dict(self, dict_str):
        items = self._split_by_outer_commas(dict_str[1:-1].strip())
        return {
            self.ParseValue(kv.split(":", 1)[0].strip()): self.ParseValue(
                kv.split(":", 1)[1].strip()
            )
            for kv in items
        }

    def _split_by_outer_commas(self, string):
        elements = []
        depth = 0
        current_element = []

        for char in string:
            if char in "[{":
                depth += 1
            elif char in "]}" and depth > 0:
                depth -= 1

            if char == "," and depth == 0:
                elements.append("".join(current_element).strip())
                current_element = []
            else:
                current_element.append(char)

        if current_element:
            elements.append("".join(current_element).strip())

        return elements

    def GetMarkersFromFile(self, filename, overwrite_image_fiducials):
        try:
            with open(filename) as file:
                magick_line = file.readline()
                assert magick_line.startswith(const.MARKER_FILE_MAGICK_STRING)
                version = int(magick_line.split("_")[-1])
                if version not in const.SUPPORTED_MARKER_FILE_VERSIONS:
                    QMessageBox.information(
                        self, _("InVesalius 3"), _("Unknown version of the markers file.")
                    )
                    return

                column_names = file.readline().strip().split("\t")
                column_names_parsed = [self.ParseValue(name) for name in column_names]

                markers_data = []
                for line in file:
                    values = line.strip().split("\t")
                    values_parsed = [self.ParseValue(value) for value in values]
                    marker_data = dict(zip(column_names_parsed, values_parsed))

                    markers_data.append(marker_data)

            self.marker_list_ctrl.hide()

            for data in markers_data:
                marker = Marker(version=version)
                marker.from_dict(data)

                marker.is_target = False

                self.markers.AddMarker(marker, render=False)

                if overwrite_image_fiducials and marker.label in self.__list_fiducial_labels():
                    Publisher.sendMessage(
                        "Load image fiducials", label=marker.label, position=marker.position
                    )

        except Exception as e:
            QMessageBox.information(self, _("InVesalius 3"), _("Invalid markers file."))
            utils.debug(e)

        self.marker_list_ctrl.show()
        Publisher.sendMessage("Render volume viewer")
        Publisher.sendMessage("Update UI for refine tab")
        self.markers.SaveState()

    def OnLoadMarkers(self, evt=None):
        last_directory = ses.Session().GetConfig("last_directory_3d_surface", "")
        dialog = dlg.FileSelectionDialog(
            _("Load markers"), last_directory, const.WILDCARD_MARKER_FILES
        )
        overwrite_checkbox = QCheckBox(_("Overwrite current image fiducials"), dialog)
        dialog.sizer.addWidget(overwrite_checkbox)
        dialog.FitSizers()
        if dialog.exec() == QDialog.Accepted:
            filename = dialog.GetPath()
            self.GetMarkersFromFile(filename, overwrite_checkbox.isChecked())

    def OnShowHideAllMarkers(self, ctrl=None):
        if ctrl is None:
            return
        if ctrl.isChecked():
            Publisher.sendMessage("Hide markers", markers=self.markers.list)
            ctrl.setText("Show all")
        else:
            Publisher.sendMessage("Show markers", markers=self.markers.list)
            ctrl.setText("Hide all")

    def OnSaveMarkers(self, evt=None):
        prj_data = prj.Project()
        timestamp = time.localtime(time.time())
        stamp_date = f"{timestamp.tm_year:0>4d}{timestamp.tm_mon:0>2d}{timestamp.tm_mday:0>2d}"
        stamp_time = f"{timestamp.tm_hour:0>2d}{timestamp.tm_min:0>2d}{timestamp.tm_sec:0>2d}"
        sep = "-"
        parts = [stamp_date, stamp_time, prj_data.name, "markers"]
        default_filename = sep.join(parts) + ".mkss"

        filename = dlg.ShowLoadSaveDialog(
            message=_("Save markers as..."),
            wildcard=const.WILDCARD_MARKER_FILES,
            default_filename=default_filename,
            save_ext=".mkss",
        )

        if not filename:
            return

        version_line = "%s%i\n" % (
            const.MARKER_FILE_MAGICK_STRING,
            const.CURRENT_MARKER_FILE_VERSION,
        )
        header_line = f"{Marker.to_csv_header()}\n"
        data_lines = [marker.to_csv_row() + "\n" for marker in self.markers.list]
        try:
            with open(filename, "w", newline="") as file:
                file.writelines([version_line, header_line])
                file.writelines(data_lines)
                file.close()
        except Exception as e:
            QMessageBox.information(self, _("InVesalius 3"), _("Error writing markers file."))
            utils.debug(str(e))

    def OnSelectColour(self, colour):
        self.marker_colour = [c / 255.0 for c in colour][:3]

    def OnSelectSize(self, ctrl=None):
        if ctrl is not None:
            self.marker_size = ctrl.value()

    def OnChangeCurrentSession(self, new_session_id):
        self.current_session = new_session_id

    def UpdateMarker(self, marker, new_position, new_orientation):
        marker_id = marker.marker_id
        self.markers.list[marker_id].position = new_position
        self.markers.list[marker_id].orientation = new_orientation
        self.UpdateMarkerInList(marker)
        self.markers.SaveState()

    def UpdateMarkerInList(self, marker):
        idx = self.__find_marker_index(marker.marker_id)
        if idx is None:
            return

        z_offset_str = str(marker.z_offset) if marker.z_offset != 0.0 else ""
        self.marker_list_ctrl.topLevelItem(idx).setText(const.Z_OFFSET_COLUMN, z_offset_str)

    def UpdateMarkerOrientation(self, marker_id=None):
        list_index = marker_id if marker_id else 0
        position = self.markers.list[list_index].position
        orientation = self.markers.list[list_index].orientation
        dialog = dlg.CreateBrainTargetDialog(mTMS=self.mTMS, marker=position + orientation)

        if dialog.exec() == QDialog.Accepted:
            orientation = dialog.GetValue()
            Publisher.sendMessage(
                "Update target orientation", target_id=marker_id, orientation=list(orientation)
            )
        dialog.close()

    def AddPeeledSurface(self, flag, actor):
        self.brain_actor = actor

    def CreateMarker(
        self,
        position=None,
        orientation=None,
        colour=None,
        size=None,
        label=None,
        is_target=False,
        seed=None,
        session_id=None,
        marker_type=MarkerType.LANDMARK,
        cortex_position_orientation=None,
        z_offset=0.0,
        z_rotation=0.0,
        mep_value=None,
    ):
        if label is None:
            label = self.GetNextMarkerLabel()

        marker = Marker()

        marker.position = position or self.current_position
        marker.orientation = orientation or self.current_orientation

        marker.colour = colour or self.marker_colour
        marker.size = size or self.marker_size
        marker.label = label
        marker.is_target = is_target
        marker.seed = seed or self.current_seed
        marker.session_id = session_id or self.current_session
        marker.marker_type = marker_type
        marker.cortex_position_orientation = (
            cortex_position_orientation or self.cortex_position_orientation
        )
        marker.z_offset = z_offset
        marker.z_rotation = z_rotation
        marker.mep_value = mep_value

        marker.marker_id = len(self.markers.list)

        marker.marker_uuid = str(uuid.uuid4())

        return marker

    def _AddMarker(self, marker, render, focus):
        num_items = self.marker_list_ctrl.topLevelItemCount()

        list_entry = ["" for _ in range(0, const.X_COLUMN)]
        list_entry[const.ID_COLUMN] = str(num_items)
        list_entry[const.SESSION_COLUMN] = str(marker.session_id)
        list_entry[const.MARKER_TYPE_COLUMN] = marker.marker_type.human_readable
        list_entry[const.LABEL_COLUMN] = marker.label
        list_entry[const.Z_OFFSET_COLUMN] = str(marker.z_offset) if marker.z_offset != 0.0 else ""
        list_entry[const.TARGET_COLUMN] = "Yes" if marker.is_target else ""
        list_entry[const.POINT_OF_INTEREST_TARGET_COLUMN] = (
            "Yes" if marker.is_point_of_interest else ""
        )
        list_entry[const.MEP_COLUMN] = str(marker.mep_value) if marker.mep_value else ""
        list_entry[const.UUID] = str(marker.marker_uuid) if marker.marker_uuid else ""

        if self.session.GetConfig("debug"):
            list_entry.append(str(round(marker.x, 1)))
            list_entry.append(str(round(marker.y, 1)))
            list_entry.append(str(round(marker.z, 1)))

        key = 0
        if len(self.itemDataMap) > 0:
            key = len(self.itemDataMap.keys()) + 1

        tree_item = QTreeWidgetItem([str(e) for e in list_entry])
        self.marker_list_ctrl.addTopLevelItem(tree_item)
        tree_item.setData(0, Qt.UserRole, key)

        data_map_entry = list_entry.copy()
        data_map_entry.append(marker.marker_uuid)
        self.itemDataMap[key] = data_map_entry

        if marker.brain_target_list:
            _set_tree_item_bg(tree_item, QColor(251, 243, 226), self.marker_list_ctrl.columnCount())

        self.marker_list_ctrl.scrollToItem(tree_item)

        if focus:
            self.FocusOnMarker(num_items)
