import os
import re
import sys
from functools import partial

import numpy as np
from matplotlib import colors as mcolors
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.data.vtk_utils as vtk_utils
import invesalius.gui.dialogs as dlg
import invesalius.gui.log as log
import invesalius.gui.widgets.gradient as grad
import invesalius.session as ses
from invesalius import inv_paths, utils
from invesalius.gui.language_dialog import ComboBoxLanguage
from invesalius.i18n import tr as _
from invesalius.navigation.navigation import Navigation
from invesalius.navigation.robot import Robot
from invesalius.navigation.tracker import Tracker
from invesalius.net.neuronavigation_api import NeuronavigationApi
from invesalius.net.pedal_connection import PedalConnector
from invesalius.pubsub import pub as Publisher


def _make_radio_group(parent, choices, orientation=Qt.Horizontal):
    group = QButtonGroup(parent)
    layout = QHBoxLayout() if orientation == Qt.Horizontal else QVBoxLayout()
    buttons = []
    for i, label in enumerate(choices):
        rb = QRadioButton(label, parent)
        group.addButton(rb, i)
        layout.addWidget(rb)
        buttons.append(rb)
    if buttons:
        buttons[0].setChecked(True)
    return group, layout, buttons


class Preferences(QDialog):
    def __init__(
        self,
        parent,
        page,
        id_=-1,
        title=_("Preferences"),
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setSizeGripEnabled(True)

        self.book = QTabWidget(self)

        self.have_log_tab = 0

        self.visualization_tab = VisualizationTab(self.book)
        self.language_tab = LanguageTab(self.book)
        if self.have_log_tab == 1:
            self.logging_tab = LoggingTab(self.book)

        self.book.addTab(self.visualization_tab, _("Visualization"))

        session = ses.Session()
        mode = session.GetConfig("mode")
        if mode == const.MODE_NAVIGATOR:
            tracker = Tracker()
            robot = Robot()
            neuronavigation_api = NeuronavigationApi()
            pedal_connector = PedalConnector(neuronavigation_api, self)
            navigation = Navigation(
                pedal_connector=pedal_connector,
                neuronavigation_api=neuronavigation_api,
            )

            self.navigation_tab = NavigationTab(self.book, navigation)
            self.tracker_tab = TrackerTab(self.book, tracker, robot)
            self.object_tab = ObjectTab(self.book, navigation, tracker, pedal_connector)

            self.book.addTab(self.navigation_tab, _("Navigation"))
            self.book.addTab(self.tracker_tab, _("Tracker"))
            self.book.addTab(self.object_tab, _("TMS Coil"))

        self.book.addTab(self.language_tab, _("Language"))
        if self.have_log_tab == 1:
            self.book.addTab(self.logging_tab, _("Logging"))

        btnsizer = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        btnsizer.accepted.connect(self.OnOK)
        btnsizer.rejected.connect(self.reject)

        self.book.setCurrentIndex(page)

        sizer = QVBoxLayout(self)
        sizer.addWidget(self.book, 1)
        sizer.addWidget(btnsizer)
        self.setLayout(sizer)
        self.__bind_events()

    def __bind_events(self):
        Publisher.subscribe(self.LoadPreferences, "Load Preferences")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.reject()
        elif event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.OnOK()
        else:
            super().keyPressEvent(event)

    def OnOK(self):
        Publisher.sendMessage("Save Preferences")
        self.accept()

    def GetPreferences(self):
        values = {}

        lang = self.language_tab.GetSelection()
        viewer = self.visualization_tab.GetSelection()

        values.update(lang)
        values.update(viewer)

        if self.have_log_tab == 1:
            logging = self.logging_tab.GetSelection()
            values.update(logging)

        return values

    def LoadPreferences(self):
        session = ses.Session()
        rendering = session.GetConfig("rendering")
        surface_interpolation = session.GetConfig("surface_interpolation")
        language = session.GetConfig("language")
        slice_interpolation = not bool(session.GetConfig("slice_interpolation"))

        file_logging = log.invLogger.GetConfig("file_logging")
        file_logging_level = log.invLogger.GetConfig("file_logging_level")
        append_log_file = log.invLogger.GetConfig("append_log_file")
        logging_file = log.invLogger.GetConfig("logging_file")
        console_logging = log.invLogger.GetConfig("console_logging")
        console_logging_level = log.invLogger.GetConfig("console_logging_level")

        values = {
            const.RENDERING: rendering,
            const.SURFACE_INTERPOLATION: surface_interpolation,
            const.LANGUAGE: language,
            const.SLICE_INTERPOLATION: slice_interpolation,
            const.FILE_LOGGING: file_logging,
            const.FILE_LOGGING_LEVEL: file_logging_level,
            const.APPEND_LOG_FILE: append_log_file,
            const.LOGFILE: logging_file,
            const.CONSOLE_LOGGING: console_logging,
            const.CONSOLE_LOGGING_LEVEL: console_logging_level,
        }

        self.visualization_tab.LoadSelection(values)
        self.language_tab.LoadSelection(values)
        if self.have_log_tab == 1:
            self.logging_tab.LoadSelection(values)


class VisualizationTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.session = ses.Session()

        self.colormaps = [str(cmap) for cmap in const.MEP_COLORMAP_DEFINITIONS.keys()]
        self.number_colors = 4
        self.cluster_volume = None

        self.conf = self.session.GetConfig("mep_configuration")
        try:
            self.conf = dict(self.conf)
        except TypeError:
            self.conf = {}
        self.conf["mep_colormap"] = self.conf.get("mep_colormap", "Viridis")

        # 3D Visualization group
        bsizer = QGroupBox(_("3D Visualization"))
        bsizer_layout = QVBoxLayout()

        lbl_inter = QLabel(_("Surface Interpolation "))
        self.rb_inter_group, rb_inter_layout, self.rb_inter_btns = _make_radio_group(
            bsizer, ["Flat", "Gouraud", "Phong"]
        )
        bsizer_layout.addWidget(lbl_inter)
        bsizer_layout.addLayout(rb_inter_layout)

        lbl_rendering = QLabel(_("Volume Rendering"))
        self.rb_rendering_group, rb_rendering_layout, self.rb_rendering_btns = _make_radio_group(
            bsizer, ["CPU", _("GPU (NVidia video cards only)")]
        )
        bsizer_layout.addWidget(lbl_rendering)
        bsizer_layout.addLayout(rb_rendering_layout)
        bsizer.setLayout(bsizer_layout)

        # 2D Visualization group
        bsizer_slices = QGroupBox(_("2D Visualization"))
        bsizer_slices_layout = QVBoxLayout()
        lbl_inter_sl = QLabel(_("Slice Interpolation "))
        self.rb_inter_sl_group, rb_inter_sl_layout, self.rb_inter_sl_btns = _make_radio_group(
            bsizer_slices, [_("Yes"), _("No")]
        )
        bsizer_slices_layout.addWidget(lbl_inter_sl)
        bsizer_slices_layout.addLayout(rb_inter_sl_layout)
        bsizer_slices.setLayout(bsizer_slices_layout)

        border = QVBoxLayout(self)
        border.addWidget(bsizer_slices)
        border.addWidget(bsizer, 1)

        if self.conf.get("mep_enabled") is True:
            self.bsizer_mep = self.InitMEPMapping(None)
            border.addWidget(self.bsizer_mep)

        self.setLayout(border)

    def GetSelection(self):
        options = {
            const.RENDERING: self.rb_rendering_group.checkedId(),
            const.SURFACE_INTERPOLATION: self.rb_inter_group.checkedId(),
            const.SLICE_INTERPOLATION: not bool(self.rb_inter_sl_group.checkedId()),
        }
        return options

    def InitMEPMapping(self, event):
        bsizer_mep = QGroupBox(_("TMS Motor Mapping"))
        bsizer_mep_layout = QVBoxLayout()

        from invesalius import project as prj

        self.proj = prj.Project()

        combo_brain_surface_name = QComboBox()
        combo_brain_surface_name.setMinimumWidth(210)
        combo_brain_surface_name.currentIndexChanged.connect(self.OnComboName)

        for n in range(len(self.proj.surface_dict)):
            combo_brain_surface_name.addItem(str(self.proj.surface_dict[n].name))

        self.combo_brain_surface_name = combo_brain_surface_name

        self._current_colour = QColor(0, 0, 255)
        button_colour = QPushButton()
        button_colour.setFixedWidth(22)
        button_colour.setStyleSheet(f"background-color: {self._current_colour.name()};")
        button_colour.clicked.connect(self.OnSelectColour)
        self.button_colour = button_colour

        line1 = QHBoxLayout()
        line1.addWidget(combo_brain_surface_name, 1)
        line1.addWidget(button_colour)

        surface_sel_lbl = QLabel(_("Brain Surface:"))
        font = surface_sel_lbl.font()
        font.setBold(True)
        surface_sel_lbl.setFont(font)
        surface_sel_sizer = QHBoxLayout()
        surface_sel_sizer.addWidget(surface_sel_lbl)
        surface_sel_sizer.addLayout(line1)

        # Gaussian Radius
        lbl_gaussian_radius = QLabel(_("Gaussian Radius:"))
        self.spin_gaussian_radius = QDoubleSpinBox()
        self.spin_gaussian_radius.setFixedWidth(64)
        self.spin_gaussian_radius.setSingleStep(0.5)
        self.spin_gaussian_radius.setRange(1, 99)
        self.spin_gaussian_radius.setValue(self.conf.get("gaussian_radius"))
        self.spin_gaussian_radius.valueChanged.connect(
            partial(self.OnSelectGaussianRadius, ctrl=self.spin_gaussian_radius)
        )

        line_gaussian_radius = QHBoxLayout()
        line_gaussian_radius.addWidget(lbl_gaussian_radius, 1)
        line_gaussian_radius.addWidget(self.spin_gaussian_radius)

        # Gaussian Standard Deviation
        lbl_std_dev = QLabel(_("Gaussian Standard Deviation:"))
        self.spin_std_dev = QDoubleSpinBox()
        self.spin_std_dev.setFixedWidth(64)
        self.spin_std_dev.setSingleStep(0.01)
        self.spin_std_dev.setRange(0.01, 5.0)
        self.spin_std_dev.setValue(self.conf.get("gaussian_sharpness"))
        self.spin_std_dev.valueChanged.connect(partial(self.OnSelectStdDev, ctrl=self.spin_std_dev))

        line_std_dev = QHBoxLayout()
        line_std_dev.addWidget(lbl_std_dev, 1)
        line_std_dev.addWidget(self.spin_std_dev)

        # Dimensions size
        lbl_dims_size = QLabel(_("Dimensions size:"))
        self.spin_dims_size = QSpinBox()
        self.spin_dims_size.setFixedWidth(64)
        self.spin_dims_size.setSingleStep(5)
        self.spin_dims_size.setRange(10, 100)
        self.spin_dims_size.setValue(self.conf.get("dimensions_size"))
        self.spin_dims_size.valueChanged.connect(
            partial(self.OnSelectDimsSize, ctrl=self.spin_dims_size)
        )

        line_dims_size = QHBoxLayout()
        line_dims_size.addWidget(lbl_dims_size, 1)
        line_dims_size.addWidget(self.spin_dims_size)

        # Colormap
        lbl_colormap = QLabel(_("Select Colormap:"))
        font_bold = lbl_colormap.font()
        font_bold.setBold(True)
        lbl_colormap.setFont(font_bold)

        self.combo_thresh = QComboBox()
        self.combo_thresh.addItems(self.colormaps)
        self.combo_thresh.currentIndexChanged.connect(self.OnSelectColormap)
        self.combo_thresh.setCurrentIndex(self.colormaps.index(self.conf.get("mep_colormap")))

        colors_gradient = self.GenerateColormapColors(
            self.conf.get("mep_colormap"), self.number_colors
        )

        self.gradient = grad.GradientDisp(bsizer_mep, -1, -5000, 5000, -5000, 5000, colors_gradient)

        colormap_gradient_sizer = QHBoxLayout()
        colormap_gradient_sizer.addWidget(self.combo_thresh)
        colormap_gradient_sizer.addWidget(self.gradient, 1)

        colormap_sizer = QVBoxLayout()
        colormap_sizer.addWidget(lbl_colormap)
        colormap_sizer.addLayout(colormap_gradient_sizer)

        colormap_custom = QVBoxLayout()

        lbl_colormap_ranges = QLabel(_("Custom Colormap Ranges"))
        font_bold2 = lbl_colormap_ranges.font()
        font_bold2.setBold(True)
        lbl_colormap_ranges.setFont(font_bold2)

        lbl_min = QLabel(_("Min Value (uV):"))
        self.spin_min = QDoubleSpinBox()
        self.spin_min.setFixedWidth(70)
        self.spin_min.setSingleStep(10)
        self.spin_min.setRange(0, 10000)
        self.spin_min.setValue(self.conf.get("colormap_range_uv").get("min"))

        lbl_low = QLabel(_("Low Value (uV):"))
        self.spin_low = QDoubleSpinBox()
        self.spin_low.setFixedWidth(70)
        self.spin_low.setSingleStep(10)
        self.spin_low.setRange(0, 10000)
        self.spin_low.setValue(self.conf.get("colormap_range_uv").get("low"))

        lbl_mid = QLabel(_("Mid Value (uV):"))
        self.spin_mid = QDoubleSpinBox()
        self.spin_mid.setFixedWidth(70)
        self.spin_mid.setSingleStep(10)
        self.spin_mid.setRange(0, 10000)
        self.spin_mid.setValue(self.conf.get("colormap_range_uv").get("mid"))

        lbl_max = QLabel(_("Max Value (uV):"))
        self.spin_max = QDoubleSpinBox()
        self.spin_max.setFixedWidth(70)
        self.spin_max.setSingleStep(10)
        self.spin_max.setRange(0, 10000)
        self.spin_max.setValue(self.conf.get("colormap_range_uv").get("max"))

        line_cm_texts = QHBoxLayout()
        for lbl in (lbl_min, lbl_low, lbl_mid, lbl_max):
            line_cm_texts.addWidget(lbl, 1)

        line_cm_spins = QHBoxLayout()
        for sp in (self.spin_min, self.spin_low, self.spin_mid, self.spin_max):
            line_cm_spins.addWidget(sp)

        for ctrl, key in [
            (self.spin_min, "min"),
            (self.spin_low, "low"),
            (self.spin_mid, "mid"),
            (self.spin_max, "max"),
        ]:
            ctrl.valueChanged.connect(partial(self.OnSelectColormapRange, ctrl=ctrl, key=key))

        colormap_custom.addWidget(lbl_colormap_ranges)
        colormap_custom.addLayout(line_cm_texts)
        colormap_custom.addLayout(line_cm_spins)

        btn_reset = QPushButton(_("Reset to defaults"))
        btn_reset.clicked.connect(self.ResetMEPSettings)

        colormap_custom.addWidget(btn_reset, 0, Qt.AlignCenter)

        colormap_sizer.addLayout(colormap_custom)

        bsizer_mep_layout.addLayout(surface_sel_sizer)
        bsizer_mep_layout.addLayout(line_gaussian_radius)
        bsizer_mep_layout.addLayout(line_std_dev)
        bsizer_mep_layout.addLayout(line_dims_size)
        bsizer_mep_layout.addLayout(colormap_sizer)

        bsizer_mep.setLayout(bsizer_mep_layout)
        return bsizer_mep

    def ResetMEPSettings(self):
        Publisher.sendMessage("Reset MEP Config")
        self.UpdateMEPFromSession()

    def UpdateMEPFromSession(self):
        self.conf = dict(self.session.GetConfig("mep_configuration"))
        self.spin_gaussian_radius.setValue(self.conf.get("gaussian_radius"))
        self.spin_std_dev.setValue(self.conf.get("gaussian_sharpness"))
        self.spin_dims_size.setValue(self.conf.get("dimensions_size"))

        self.combo_thresh.setCurrentIndex(self.colormaps.index(self.conf.get("mep_colormap")))

        ranges = self.conf.get("colormap_range_uv")
        ranges = dict(ranges)
        self.spin_min.setValue(ranges.get("min"))
        self.spin_low.setValue(ranges.get("low"))
        self.spin_mid.setValue(ranges.get("mid"))
        self.spin_max.setValue(ranges.get("max"))

    def OnSelectStdDev(self, value=None, ctrl=None):
        self.conf["gaussian_sharpness"] = ctrl.value()
        self.session.SetConfig("mep_configuration", self.conf)

    def OnSelectGaussianRadius(self, value=None, ctrl=None):
        self.conf["gaussian_radius"] = ctrl.value()
        self.session.SetConfig("mep_configuration", self.conf)

    def OnSelectDimsSize(self, value=None, ctrl=None):
        self.conf["dimensions_size"] = ctrl.value()
        self.session.SetConfig("mep_configuration", self.conf)

    def OnSelectColormapRange(self, value=None, ctrl=None, key=None):
        self.conf["colormap_range_uv"][key] = ctrl.value()
        self.session.SetConfig("mep_configuration", self.conf)

    def LoadSelection(self, values):
        rendering = values[const.RENDERING]
        surface_interpolation = values[const.SURFACE_INTERPOLATION]
        slice_interpolation = values[const.SLICE_INTERPOLATION]

        self.rb_rendering_btns[int(rendering)].setChecked(True)
        self.rb_inter_btns[int(surface_interpolation)].setChecked(True)
        self.rb_inter_sl_btns[int(slice_interpolation)].setChecked(True)

    def OnSelectColormap(self, index=None):
        self.conf["mep_colormap"] = self.colormaps[self.combo_thresh.currentIndex()]
        colors = self.GenerateColormapColors(self.conf.get("mep_colormap"), self.number_colors)

        self.session.SetConfig("mep_configuration", self.conf)
        Publisher.sendMessage("Save Preferences")
        self.UpdateGradient(self.gradient, colors)

    def GenerateColormapColors(self, colormap_name, number_colors=4):
        color_def = const.MEP_COLORMAP_DEFINITIONS[colormap_name]
        colors = list(color_def.values())
        positions = [0.0, 0.25, 0.5, 1.0]

        cmap = mcolors.LinearSegmentedColormap.from_list(
            colormap_name, list(zip(positions, colors))
        )
        colors_gradient = [
            (
                int(255 * cmap(i)[0]),
                int(255 * cmap(i)[1]),
                int(255 * cmap(i)[2]),
                int(255 * cmap(i)[3]),
            )
            for i in np.linspace(0, 1, number_colors)
        ]

        return colors_gradient

    def UpdateGradient(self, gradient, colors):
        gradient.SetGradientColours(colors)
        gradient.update()

        self.update()
        self.show()

    def OnComboName(self, index=None):
        from invesalius import project as prj

        self.proj = prj.Project()
        surface_index = self.combo_brain_surface_name.currentIndex()
        Publisher.sendMessage("Show single surface", index=surface_index, visibility=True)
        Publisher.sendMessage("Get brain surface actor", index=surface_index)
        Publisher.sendMessage("Press motor map button", pressed=True)

        colour = [int(value * 255) for value in self.proj.surface_dict[surface_index].colour]
        self._current_colour = QColor(*colour[:3])
        self.button_colour.setStyleSheet(f"background-color: {self._current_colour.name()};")

    def OnSelectColour(self):
        colour = QColorDialog.getColor(self._current_colour, self)
        if colour.isValid():
            self._current_colour = colour
            self.button_colour.setStyleSheet(f"background-color: {colour.name()};")
            colour_float = [colour.redF(), colour.greenF(), colour.blueF()]
            Publisher.sendMessage(
                "Set surface colour",
                surface_index=self.combo_brain_surface_name.currentIndex(),
                colour=colour_float,
            )


class LoggingTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        # File Logging Options group
        bsizer_logging = QGroupBox(_("File Logging Options"))
        bsizer_logging_layout = QVBoxLayout()

        bsizer_file_logging = QHBoxLayout()

        lbl_file_logging = QLabel("Do Logging")
        self.rb_file_logging_group, rb_file_logging_layout, self.rb_file_logging_btns = (
            _make_radio_group(self, ["No", "Yes"])
        )
        bsizer_file_logging.addWidget(lbl_file_logging)
        bsizer_file_logging.addLayout(rb_file_logging_layout)

        lbl_append_file = QLabel("Append File")
        self.rb_append_file_group, rb_append_file_layout, self.rb_append_file_btns = (
            _make_radio_group(self, ["No", "Yes"])
        )
        bsizer_file_logging.addWidget(lbl_append_file)
        bsizer_file_logging.addLayout(rb_append_file_layout)

        lbl_file_logging_level = QLabel(_(" Logging Level "))
        self.cb_file_logging_level = QComboBox()
        self.cb_file_logging_level.addItems(const.LOGGING_LEVEL_TYPES)
        bsizer_file_logging.addWidget(lbl_file_logging_level)
        bsizer_file_logging.addWidget(self.cb_file_logging_level)

        bsizer_logging_layout.addLayout(bsizer_file_logging)

        bsizer_log_filename = QHBoxLayout()
        lbl_log_file_label = QLabel(_("File:"))
        self.tc_log_file_name = QLineEdit()
        self.tc_log_file_name.setReadOnly(True)
        self.tc_log_file_name.setMinimumWidth(300)
        palette = self.tc_log_file_name.palette()
        palette.setColor(QPalette.Text, QColor("blue"))
        self.tc_log_file_name.setPalette(palette)

        bt_log_file_select = QPushButton("Modify")
        bt_log_file_select.clicked.connect(self.OnModifyButton)

        bsizer_log_filename.addWidget(lbl_log_file_label)
        bsizer_log_filename.addWidget(self.tc_log_file_name)
        bsizer_log_filename.addWidget(bt_log_file_select)
        bsizer_logging_layout.addLayout(bsizer_log_filename)
        bsizer_logging.setLayout(bsizer_logging_layout)

        # Console Logging Options group
        bsizer_console_logging = QGroupBox(_(" Console Logging Options"))
        bsizer_console_layout = QHBoxLayout()

        lbl_console_logging = QLabel("Do logging")
        self.rb_console_logging_group, rb_console_logging_layout, self.rb_console_logging_btns = (
            _make_radio_group(self, ["No", "Yes"])
        )
        bsizer_console_layout.addWidget(lbl_console_logging)
        bsizer_console_layout.addLayout(rb_console_logging_layout)

        lbl_console_logging_level = QLabel(_(" Logging Level "))
        self.cb_console_logging_level = QComboBox()
        self.cb_console_logging_level.addItems(const.LOGGING_LEVEL_TYPES)
        bsizer_console_layout.addWidget(lbl_console_logging_level)
        bsizer_console_layout.addWidget(self.cb_console_logging_level)
        bsizer_console_logging.setLayout(bsizer_console_layout)

        border = QVBoxLayout(self)
        border.addWidget(bsizer_logging, 1)
        border.addWidget(bsizer_console_logging, 1)
        self.setLayout(border)

    @log.call_tracking_decorator
    def OnModifyButton(self):
        from PySide6.QtWidgets import QFileDialog

        logging_file = self.tc_log_file_name.text()
        path, fname = os.path.split(logging_file)
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Log Contents",
            os.path.join(path, fname),
            "Log files (*.log)",
        )
        if file_path:
            self.tc_log_file_name.setText(file_path)
            return True
        return False

    def GetSelection(self):
        options = {
            const.FILE_LOGGING: self.rb_file_logging_group.checkedId(),
            const.FILE_LOGGING_LEVEL: self.cb_file_logging_level.currentIndex(),
            const.APPEND_LOG_FILE: self.rb_append_file_group.checkedId(),
            const.LOGFILE: self.tc_log_file_name.text(),
            const.CONSOLE_LOGGING: self.rb_console_logging_group.checkedId(),
            const.CONSOLE_LOGGING_LEVEL: self.cb_console_logging_level.currentIndex(),
        }

        file_logging = self.rb_file_logging_group.checkedId()
        log.invLogger.SetConfig("file_logging", file_logging)
        file_logging_level = self.cb_file_logging_level.currentIndex()
        log.invLogger.SetConfig("file_logging_level", file_logging_level)
        append_log_file = self.rb_append_file_group.checkedId()
        log.invLogger.SetConfig("append_log_file", append_log_file)
        logging_file = self.tc_log_file_name.text()
        log.invLogger.SetConfig("logging_file", logging_file)
        console_logging = self.rb_console_logging_group.checkedId()
        log.invLogger.SetConfig("console_logging", console_logging)
        console_logging_level = self.cb_console_logging_level.currentIndex()
        log.invLogger.SetConfig("console_logging_level", console_logging_level)
        log.invLogger.configureLogging()

        return options

    def LoadSelection(self, values):
        file_logging = values[const.FILE_LOGGING]
        file_logging_level = values[const.FILE_LOGGING_LEVEL]
        append_log_file = values[const.APPEND_LOG_FILE]
        logging_file = values[const.LOGFILE]
        console_logging = values[const.CONSOLE_LOGGING]
        console_logging_level = values[const.CONSOLE_LOGGING_LEVEL]

        self.rb_file_logging_btns[int(file_logging)].setChecked(True)
        self.cb_file_logging_level.setCurrentIndex(int(file_logging_level))
        self.rb_append_file_btns[int(append_log_file)].setChecked(True)
        self.tc_log_file_name.setText(logging_file)
        self.rb_console_logging_btns[int(console_logging)].setChecked(True)
        self.cb_console_logging_level.setCurrentIndex(int(console_logging_level))


class NavigationTab(QWidget):
    def __init__(self, parent, navigation):
        super().__init__(parent)

        self.session = ses.Session()
        self.navigation = navigation
        self.sleep_nav = self.navigation.sleep_nav
        self.sleep_coord = const.SLEEP_COORDINATES

        self.LoadConfig()

        text_note = QLabel(_("Note: Using too low sleep times can result in Invesalius crashing!"))

        nav_sleep = QLabel(_("Navigation Sleep (s):"))
        spin_nav_sleep = QDoubleSpinBox()
        spin_nav_sleep.setFixedWidth(80)
        spin_nav_sleep.setSingleStep(0.01)
        spin_nav_sleep.setRange(0.01, 10.0)
        spin_nav_sleep.setValue(self.sleep_nav)
        spin_nav_sleep.valueChanged.connect(partial(self.OnSelectNavSleep, ctrl=spin_nav_sleep))

        coord_sleep = QLabel(_("Coordinate Sleep (s):"))
        spin_coord_sleep = QDoubleSpinBox()
        spin_coord_sleep.setFixedWidth(80)
        spin_coord_sleep.setSingleStep(0.01)
        spin_coord_sleep.setRange(0.01, 10.0)
        spin_coord_sleep.setValue(self.sleep_coord)
        spin_coord_sleep.valueChanged.connect(
            partial(self.OnSelectCoordSleep, ctrl=spin_coord_sleep)
        )

        line_nav_sleep = QHBoxLayout()
        line_nav_sleep.addWidget(nav_sleep, 1)
        line_nav_sleep.addWidget(spin_nav_sleep)

        line_coord_sleep = QHBoxLayout()
        line_coord_sleep.addWidget(coord_sleep, 1)
        line_coord_sleep.addWidget(spin_coord_sleep)

        conf_sizer = QGroupBox(_("Sleep time configuration"))
        conf_layout = QVBoxLayout()
        conf_layout.addWidget(text_note)
        conf_layout.addLayout(line_nav_sleep)
        conf_layout.addLayout(line_coord_sleep)
        conf_sizer.setLayout(conf_layout)

        main_sizer = QVBoxLayout(self)
        main_sizer.addWidget(conf_sizer)
        self.setLayout(main_sizer)

    def OnSelectNavSleep(self, value=None, ctrl=None):
        self.sleep_nav = ctrl.value()
        self.navigation.UpdateNavSleep(self.sleep_nav)
        self.session.SetConfig("sleep_nav", self.sleep_nav)

    def OnSelectCoordSleep(self, value=None, ctrl=None):
        self.sleep_coord = ctrl.value()
        Publisher.sendMessage("Update coord sleep", data=self.sleep_coord)
        self.session.SetConfig("sleep_coord", self.sleep_nav)

    def LoadConfig(self):
        sleep_nav = self.session.GetConfig("sleep_nav")
        sleep_coord = self.session.GetConfig("sleep_coord")

        if sleep_nav is not None:
            self.sleep_nav = sleep_nav

        if sleep_coord is not None:
            self.sleep_coord = sleep_coord


class ObjectTab(QWidget):
    def __init__(self, parent, navigation, tracker, pedal_connector):
        super().__init__(parent)

        self.session = ses.Session()

        self.coil_list = const.COIL

        self.tracker = tracker
        self.pedal_connector = pedal_connector
        self.navigation = navigation
        self.robot = Robot()
        self.coil_registrations = {}
        self.__bind_events()

        # TMS coil registration group
        self.config_lbl = QLabel(_("Current Configuration:"))
        font = self.config_lbl.font()
        font.setBold(True)
        self.config_lbl.setFont(font)

        self.config_txt = QLabel(
            f"{os.path.basename(self.coil_registrations.get('default_coil', {}).get('path', 'None'))}"
        )

        tooltip = _("New TMS coil configuration")
        btn_new = QPushButton(_("New"))
        btn_new.setFixedSize(65, 23)
        btn_new.setToolTip(tooltip)
        btn_new.clicked.connect(self.OnCreateNewCoil)

        tooltip = _("Load TMS coil configuration from an OBR file")
        btn_load = QPushButton(_("Load"))
        btn_load.setFixedSize(65, 23)
        btn_load.setToolTip(tooltip)
        btn_load.clicked.connect(self.OnLoadCoilFromOBR)

        tooltip = _("Save TMS coil configuration to a file")
        btn_save = QPushButton(_("Save"))
        btn_save.setFixedSize(65, 23)
        btn_save.setToolTip(tooltip)
        btn_save.clicked.connect(self.OnSaveCoilToOBR)

        coil_sizer = QGroupBox(_("TMS coil registration"))
        inner_coil_layout = QGridLayout()
        inner_coil_layout.addWidget(self.config_lbl, 0, 0)
        inner_coil_layout.addWidget(self.config_txt, 0, 1)
        inner_coil_layout.addWidget(btn_new, 1, 0)
        inner_coil_layout.addWidget(btn_load, 1, 1)
        inner_coil_layout.addWidget(btn_save, 1, 2)
        coil_sizer.setLayout(inner_coil_layout)

        # Angle/Dist thresholds
        self.angle_threshold = self.session.GetConfig(
            "angle_threshold", const.DEFAULT_ANGLE_THRESHOLD
        )
        self.distance_threshold = self.session.GetConfig(
            "distance_threshold", const.DEFAULT_DISTANCE_THRESHOLD
        )

        text_angles = QLabel(_("Angle threshold (degrees):"))
        spin_size_angles = QDoubleSpinBox()
        spin_size_angles.setFixedWidth(80)
        spin_size_angles.setRange(0.1, 99)
        spin_size_angles.setValue(self.angle_threshold)
        spin_size_angles.valueChanged.connect(
            partial(self.OnSelectAngleThreshold, ctrl=spin_size_angles)
        )

        text_dist = QLabel(_("Distance threshold (mm):"))
        spin_size_dist = QDoubleSpinBox()
        spin_size_dist.setFixedWidth(80)
        spin_size_dist.setRange(0.1, 99)
        spin_size_dist.setValue(self.distance_threshold)
        spin_size_dist.valueChanged.connect(
            partial(self.OnSelectDistanceThreshold, ctrl=spin_size_dist)
        )

        line_angle_threshold = QHBoxLayout()
        line_angle_threshold.addWidget(text_angles, 1)
        line_angle_threshold.addWidget(spin_size_angles)

        line_dist_threshold = QHBoxLayout()
        line_dist_threshold.addWidget(text_dist, 1)
        line_dist_threshold.addWidget(spin_size_dist)

        conf_sizer = QGroupBox(_("Settings"))
        conf_layout = QVBoxLayout()
        conf_layout.addLayout(line_angle_threshold)
        conf_layout.addLayout(line_dist_threshold)
        conf_sizer.setLayout(conf_layout)

        # TMS coil selection
        self.sel_sizer = QGroupBox(
            _(
                f"TMS coil selection ({len(navigation.coil_registrations)} out of {navigation.n_coils})"
            )
        )
        self.inner_sel_layout = QVBoxLayout()

        self.coil_btns = {}
        self.no_coils_lbl = None
        if len(self.coil_registrations) == 0:
            self.no_coils_lbl = QLabel(
                _("No coils found in config.json. Create or load new coils below.")
            )
            self.inner_sel_layout.addWidget(self.no_coils_lbl)
        self.sel_sizer.setLayout(self.inner_sel_layout)

        # Robot coil selection
        self.robot_sizer = QGroupBox(_("Robot coil selection"))
        self.inner_robot_layout = QVBoxLayout()

        self.robot_lbl = QLabel(_("Robot is connected. Coil attached to robot: "))
        self.choice_robot_coil = QComboBox()
        self.choice_robot_coil.setFixedWidth(90)
        self.choice_robot_coil.addItems(list(self.navigation.coil_registrations))
        robot_coil_name = self.robot.GetCoilName() or ""
        idx = self.choice_robot_coil.findText(robot_coil_name)
        if idx >= 0:
            self.choice_robot_coil.setCurrentIndex(idx)
        self.choice_robot_coil.setToolTip("Specify which coil is attached to the robot")
        self.choice_robot_coil.currentTextChanged.connect(self.OnChoiceRobotCoil)

        if not self.robot.IsConnected():
            self.robot_lbl.setText("Robot is not connected")
            self.choice_robot_coil.hide()

        self.inner_robot_layout.addWidget(self.robot_lbl)
        self.inner_robot_layout.addWidget(self.choice_robot_coil)
        self.robot_sizer.setLayout(self.inner_robot_layout)

        # Main sizer
        main_sizer = QVBoxLayout(self)
        main_sizer.addWidget(coil_sizer)
        main_sizer.addWidget(self.sel_sizer)
        main_sizer.addWidget(self.robot_sizer)
        main_sizer.addWidget(conf_sizer)
        self.setLayout(main_sizer)

        self.LoadConfig()

    def __bind_events(self):
        Publisher.subscribe(self.OnSetCoilCount, "Reset coil selection")
        Publisher.subscribe(
            self.OnRobotConnectionStatus, "Robot to Neuronavigation: Robot connection status"
        )

    def OnRobotConnectionStatus(self, data):
        if data is None:
            return
        if data == "Connected":
            self.choice_robot_coil.show()
            self.robot_lbl.setText("Robot is connected. Coil attached to robot: ")
        else:
            self.robot_lbl.setText("Robot is not connected.")

    def OnChoiceRobotCoil(self, text):
        self.robot.SetCoilName(text)

    def AddCoilButton(self, coil_name, show_button=True):
        if self.no_coils_lbl is not None:
            self.no_coils_lbl.deleteLater()
            self.no_coils_lbl = None

        if coil_name not in self.coil_btns:
            coil_btn = QPushButton(coil_name[:8])
            coil_btn.setCheckable(True)
            coil_btn.setFixedSize(88, 17)
            coil_btn.setToolTip(coil_name)
            coil_btn.clicked.connect(lambda checked, name=coil_name: self.OnSelectCoil(name=name))
            coil_btn.setContextMenuPolicy(Qt.CustomContextMenu)
            coil_btn.customContextMenuRequested.connect(
                lambda pos, name=coil_name: self.OnRightClickCoil(pos, name)
            )
            coil_btn.setVisible(show_button)
            self.coil_btns[coil_name] = coil_btn
            self.inner_sel_layout.addWidget(coil_btn)

    def ShowMulticoilGUI(self, show_multicoil):
        self.config_txt.setVisible(not show_multicoil)
        self.config_lbl.setVisible(not show_multicoil)

        self.sel_sizer.setVisible(show_multicoil)
        self.robot_sizer.setVisible(show_multicoil)

        self.choice_robot_coil.setVisible(show_multicoil and self.robot.IsConnected())

    def OnSetCoilCount(self, n_coils):
        multicoil_mode = n_coils > 1

        if multicoil_mode:
            self.sel_sizer.setTitle(f"TMS coil selection (0 out of {n_coils})")

            for btn in self.coil_btns.values():
                btn.setEnabled(True)
                btn.setChecked(False)

        self.ShowMulticoilGUI(multicoil_mode)

    def LoadConfig(self):
        state = self.session.GetConfig("navigation", {})
        n_coils = state.get("n_coils", 1)
        multicoil_mode = n_coils > 1
        self.ShowMulticoilGUI(multicoil_mode)

        self.coil_registrations = self.session.GetConfig("coil_registrations", {})
        for coil_name in self.coil_registrations:
            self.AddCoilButton(coil_name, show_button=multicoil_mode)

        selected_coils = state.get("selected_coils", [])
        for coil_name in selected_coils:
            self.coil_btns[coil_name].setChecked(True)

        self.config_txt.setText(
            f"{os.path.basename(self.coil_registrations.get('default_coil', {}).get('path', 'None'))}"
        )

        n_coils_selected = len(selected_coils)
        self.sel_sizer.setTitle(f"TMS coil selection ({n_coils_selected} out of {n_coils})")

        if n_coils_selected == n_coils:
            self.CoilSelectionDone()

    def CoilSelectionDone(self):
        if self.navigation.n_coils == 1:
            self.robot.SetCoilName(next(iter(self.navigation.coil_registrations)))

        Publisher.sendMessage("Coil selection done", done=True)
        Publisher.sendMessage("Update status text in GUI", label=_("Ready"))

        for btn in self.coil_btns.values():
            btn.setEnabled(btn.isChecked())

    def OnSelectCoil(self, event=None, name=None, select=False):
        if name is None:
            if not select:
                Publisher.sendMessage("Reset coil selection", n_coils=self.navigation.n_coils)
            return

        coil_registration = None
        navigation = self.navigation

        btn = self.coil_btns.get(name)
        is_selected = select or (btn is not None and btn.isChecked())

        if is_selected:
            coil_registration = self.coil_registrations[name]

            obj_id = coil_registration["obj_id"]
            selected_registrations = navigation.coil_registrations
            conflicting_coil_name = next(
                (
                    coil_name
                    for coil_name, registration in selected_registrations.items()
                    if registration["obj_id"] == obj_id
                ),
                None,
            )
            if conflicting_coil_name is not None:
                QMessageBox.warning(
                    self,
                    _("InVesalius 3"),
                    _(
                        f"Cannot select this coil, its index (obj_id = {obj_id}) conflicts with selected coil: {conflicting_coil_name}"
                    ),
                )
                self.coil_btns[name].setChecked(False)
                return

            elif (obj_tracker_id := coil_registration["tracker_id"]) != self.tracker.tracker_id:
                QMessageBox.warning(
                    self,
                    _("InVesalius 3"),
                    _(
                        f"Cannot select this coil, its tracker [{const.TRACKERS[obj_tracker_id - 1]}] does not match the selected tracker [{const.TRACKERS[self.tracker.tracker_id - 1]}]"
                    ),
                )
                self.coil_btns[name].setChecked(False)
                return

            self.coil_btns[name].setChecked(True)

        Publisher.sendMessage("Select coil", coil_name=name, coil_registration=coil_registration)

        n_coils_selected = len(navigation.coil_registrations)
        n_coils = navigation.n_coils

        self.config_txt.setText(
            f"{os.path.basename(self.coil_registrations.get('default_coil', {}).get('path', 'None'))}"
        )
        self.sel_sizer.setTitle(f"TMS coil selection ({n_coils_selected} out of {n_coils})")

        if self.choice_robot_coil is not None:
            self.choice_robot_coil.clear()
            self.choice_robot_coil.addItems(list(navigation.coil_registrations))
            idx = self.choice_robot_coil.findText(self.robot.GetCoilName() or "")
            if idx >= 0:
                self.choice_robot_coil.setCurrentIndex(idx)

        if n_coils_selected == n_coils:
            self.CoilSelectionDone()
        else:
            Publisher.sendMessage("Coil selection done", done=False)
            for btn in self.coil_btns.values():
                btn.setEnabled(True)

    def OnRightClickCoil(self, pos, name):
        def DeleteCoil(name):
            self.OnSelectCoil(name=name, select=False)
            del self.coil_registrations[name]

            self.coil_btns[name].deleteLater()
            del self.coil_btns[name]

            self.session.SetConfig("coil_registrations", self.coil_registrations)
            Publisher.sendMessage("Select coil", coil_name=name, coil_registration=None)

        menu = QMenu(self)
        delete_action = menu.addAction("Delete coil")
        save_action = menu.addAction("Save coil to OBR file")

        delete_action.triggered.connect(lambda: DeleteCoil(name))
        save_action.triggered.connect(lambda: self.OnSaveCoilToOBR(coil_name=name))

        sender = self.coil_btns.get(name)
        if sender:
            menu.exec(sender.mapToGlobal(pos))

    def OnCreateNewCoil(self):
        if self.tracker.IsTrackerInitialized():
            dialog = dlg.ObjectCalibrationDialog(
                self.tracker,
                self.navigation.n_coils,
                self.pedal_connector,
            )
            try:
                if dialog.exec() == QDialog.Accepted:
                    (coil_name, coil_path, obj_fiducials, obj_orients, obj_id, tracker_id) = (
                        dialog.GetValue()
                    )

                    if coil_name in self.coil_registrations and coil_name != "default_coil":
                        from PySide6.QtWidgets import QInputDialog

                        new_name, ok = QInputDialog.getText(
                            self,
                            _("Warning: Coil Name Conflict"),
                            _(
                                "A registration with this name already exists. Enter a new name or overwrite an old coil registration"
                            ),
                            text=coil_name,
                        )
                        if ok:
                            coil_name = new_name.strip()
                        else:
                            return

                    if np.isfinite(obj_fiducials).all() and np.isfinite(obj_orients).all():
                        coil_registration = {
                            "fiducials": obj_fiducials.tolist(),
                            "orientations": obj_orients.tolist(),
                            "obj_id": obj_id,
                            "tracker_id": tracker_id,
                            "path": coil_path.decode(const.FS_ENCODE),
                        }
                        self.coil_registrations[coil_name] = coil_registration
                        self.session.SetConfig("coil_registrations", self.coil_registrations)
                        self.AddCoilButton(coil_name)

                        coil_btn = self.coil_btns[coil_name]
                        if coil_btn.isChecked():
                            coil_btn.setChecked(False)
                            self.OnSelectCoil(name=coil_name, select=False)

                        if len(self.navigation.coil_registrations) < self.navigation.n_coils:
                            self.OnSelectCoil(name=coil_name, select=True)
                        else:
                            coil_btn.setEnabled(False)

                        coil_btn.setVisible(self.navigation.n_coils > 1)

            except Exception:
                pass
            dialog.close()
        else:
            dlg.ShowNavigationTrackerWarning(0, "choose")

    def OnLoadCoilFromOBR(self):
        filename = dlg.ShowLoadSaveDialog(
            message=_("Load object registration"), wildcard=_("Registration files (*.obr)|*.obr")
        )

        try:
            if filename:
                with open(filename, "r") as text_file:
                    data = [s.split("\t") for s in text_file.readlines()]

                registration_coordinates = np.array(data[1:]).astype(np.float32)
                obj_fiducials = registration_coordinates[:, :3]
                obj_orients = registration_coordinates[:, 3:]

                coil_name = data[0][0][2:]
                coil_path = data[0][1].encode(const.FS_ENCODE)
                tracker_id = int(data[0][3])
                obj_id = int(data[0][-1])
                coil_name = "default_coil" if self.navigation.n_coils == 1 else coil_name

                if len(data[0]) < 6:
                    coil_name = "default_coil"
                    tracker_id = self.tracker.tracker_id

                if coil_name in self.coil_registrations and coil_name != "default_coil":
                    from PySide6.QtWidgets import QInputDialog

                    new_name, ok = QInputDialog.getText(
                        self,
                        _("Warning: Coil Name Conflict"),
                        _(
                            "A registration with this name already exists. Enter a new name or overwrite an old coil registration"
                        ),
                        text=coil_name,
                    )
                    if ok:
                        coil_name = new_name.strip()
                    else:
                        return

                if not os.path.exists(coil_path):
                    coil_path = os.path.join(inv_paths.OBJ_DIR, "magstim_fig8_coil.stl")

                polydata = vtk_utils.CreateObjectPolyData(coil_path)
                if not polydata:
                    coil_path = os.path.join(inv_paths.OBJ_DIR, "magstim_fig8_coil.stl")

                if np.isfinite(obj_fiducials).all() and np.isfinite(obj_orients).all():
                    coil_registration = {
                        "fiducials": obj_fiducials.tolist(),
                        "orientations": obj_orients.tolist(),
                        "obj_id": obj_id,
                        "tracker_id": tracker_id,
                        "path": coil_path.decode(const.FS_ENCODE),
                    }
                    self.coil_registrations[coil_name] = coil_registration
                    self.session.SetConfig("coil_registrations", self.coil_registrations)
                    self.AddCoilButton(coil_name)

                    coil_btn = self.coil_btns[coil_name]
                    if coil_btn.isChecked():
                        coil_btn.setChecked(False)
                        self.OnSelectCoil(name=coil_name, select=False)
                    elif self.navigation.CoilSelectionDone():
                        coil_btn.setEnabled(False)

                    if self.navigation.n_coils == 1:
                        self.OnSelectCoil(name="default_coil", select=False)
                        self.OnSelectCoil(name="default_coil", select=True)
                        coil_btn.hide()

                Publisher.sendMessage(
                    "Update status text in GUI", label=_("Object file successfully loaded")
                )

                msg = _("Object file successfully loaded")
                QMessageBox.information(self, _("InVesalius 3"), msg)
        except Exception:
            QMessageBox.warning(
                self, _("InVesalius 3"), _("Object registration file incompatible.")
            )
            Publisher.sendMessage("Update status text in GUI", label="")

    def OnSaveCoilToOBR(self, evt=None, coil_name=None):
        if coil_name is None:
            if self.navigation.n_coils > 1 and self.coil_registrations:
                from PySide6.QtWidgets import QInputDialog

                coil_name, ok = QInputDialog.getItem(
                    self,
                    _("Saving coil registration"),
                    _("Select which coil registration to save"),
                    list(self.coil_registrations),
                    0,
                    False,
                )
                if not ok:
                    return
            else:
                coil_name = next(iter(self.coil_registrations), None)

        coil_registration = self.coil_registrations.get(coil_name, None)

        if coil_registration is None:
            QMessageBox.warning(
                self, _("Save"), _("Failed to save registration: No registration to save!")
            )
            return

        filename = dlg.ShowLoadSaveDialog(
            message=_("Save object registration as..."),
            wildcard=_("Registration files (*.obr)|*.obr"),
            style=1,
            default_filename="object_registration.obr",
            save_ext="obr",
        )
        if filename:
            hdr = (
                coil_name
                + "\t"
                + utils.decode(coil_registration["path"], const.FS_ENCODE)
                + "\t"
                + "Tracker"
                + "\t"
                + str("%d" % coil_registration["tracker_id"])
                + "\t"
                + "Index"
                + "\t"
                + str("%d" % coil_registration["obj_id"])
            )
            data = np.hstack([coil_registration["fiducials"], coil_registration["orientations"]])
            np.savetxt(filename, data, fmt="%.4f", delimiter="\t", newline="\n", header=hdr)
            QMessageBox.information(self, _("Save"), _("Object file successfully saved"))

    def OnSelectAngleThreshold(self, value=None, ctrl=None):
        self.angle_threshold = ctrl.value()
        Publisher.sendMessage("Update angle threshold", angle=self.angle_threshold)
        self.session.SetConfig("angle_threshold", self.angle_threshold)

    def OnSelectDistanceThreshold(self, value=None, ctrl=None):
        self.distance_threshold = ctrl.value()
        Publisher.sendMessage("Update distance threshold", dist_threshold=self.distance_threshold)
        self.session.SetConfig("distance_threshold", self.distance_threshold)


class TrackerTab(QWidget):
    def __init__(self, parent, tracker, robot):
        super().__init__(parent)

        self.session = ses.Session()

        self.__bind_events()

        self.tracker = tracker
        self.robot = robot
        self.robot_ip = None
        self.matrix_tracker_to_robot = None
        self.n_coils = 1
        self.LoadConfig()

        # Number of coils
        n_coils_options = [str(n) for n in range(1, 10)]
        select_n_coils_elem = QComboBox()
        select_n_coils_elem.setFixedWidth(145)
        select_n_coils_elem.addItems(n_coils_options)
        select_n_coils_elem.setToolTip(_("Choose the number of coils to track"))
        select_n_coils_elem.setCurrentIndex(self.n_coils - 1)
        select_n_coils_elem.currentIndexChanged.connect(
            partial(self.OnChooseNoOfCoils, ctrl=select_n_coils_elem)
        )

        select_n_coils_label = QLabel(_("Choose the number of coils to track:"))

        # Tracker selection
        tracker_options = [_("Select")] + self.tracker.get_trackers()
        select_tracker_elem = QComboBox()
        select_tracker_elem.setFixedWidth(145)
        select_tracker_elem.addItems(tracker_options)
        select_tracker_elem.setToolTip(_("Choose the tracking device"))
        select_tracker_elem.setCurrentIndex(self.tracker.tracker_id)
        select_tracker_elem.currentIndexChanged.connect(
            partial(self.OnChooseTracker, ctrl=select_tracker_elem)
        )

        select_tracker_label = QLabel(_("Choose the tracking device: "))

        # Reference mode
        choice_ref = QComboBox()
        choice_ref.setFixedWidth(145)
        choice_ref.addItems(const.REF_MODE)
        choice_ref.setCurrentIndex(const.DEFAULT_REF_MODE)
        choice_ref.setToolTip(_("Choose the navigation reference mode"))
        choice_ref.currentIndexChanged.connect(
            partial(self.OnChooseReferenceMode, ctrl=select_tracker_elem)
        )
        self.choice_ref = choice_ref

        choice_ref_label = QLabel(_("Choose the navigation reference mode: "))

        ref_layout = QGridLayout()
        ref_layout.addWidget(select_n_coils_label, 0, 0)
        ref_layout.addWidget(select_n_coils_elem, 0, 1)
        ref_layout.addWidget(select_tracker_label, 1, 0)
        ref_layout.addWidget(select_tracker_elem, 1, 1)
        ref_layout.addWidget(choice_ref_label, 2, 0)
        ref_layout.addWidget(choice_ref, 2, 1)

        sizer = QGroupBox(_("Setup tracker"))
        sizer_layout = QVBoxLayout()
        sizer_layout.addLayout(ref_layout)
        sizer.setLayout(sizer_layout)

        # Robot IP
        lbl_rob = QLabel(_("IP for robot device: "))

        robot_ip_options = self.robot.robot_ip_options
        choice_IP = QComboBox()
        choice_IP.setEditable(True)
        choice_IP.addItems(robot_ip_options)
        choice_IP.setToolTip(_("Choose or type the robot IP"))

        if self.robot.robot_ip in self.robot.robot_ip_options:
            choice_IP.setCurrentIndex(robot_ip_options.index(self.robot.robot_ip))
            self.robot_ip = choice_IP.currentText()
        elif self.robot.robot_ip is not None:
            choice_IP.setEditText(self.robot.robot_ip)
            self.robot_ip = choice_IP.currentText()
        elif choice_IP.currentText() == "" and choice_IP.count() == 0:
            choice_IP.setEditText(_("Select or type robot IP"))
        elif choice_IP.currentText() == "":
            choice_IP.setCurrentIndex(0)
            self.robot_ip = choice_IP.currentText()

        choice_IP.currentTextChanged.connect(partial(self.OnTxt_Ent, ctrl=choice_IP))
        choice_IP.currentIndexChanged.connect(partial(self.OnChoiceIP, ctrl=choice_IP))
        self.choice_IP = choice_IP

        btn_rob_add_ip = QPushButton("+")
        btn_rob_add_ip.setToolTip("Add a new IP to the list")
        btn_rob_add_ip.setFixedWidth(30)
        btn_rob_add_ip.clicked.connect(self.OnAddIP)
        self.btn_rob_add_ip = btn_rob_add_ip

        btn_rob_rem_ip = QPushButton("-")
        btn_rob_rem_ip.setToolTip("Remove the selected IP from the list")
        btn_rob_rem_ip.setFixedWidth(30)
        btn_rob_rem_ip.clicked.connect(self.OnRemoveIP)
        self.btn_rob_rem_ip = btn_rob_rem_ip

        btn_rob = QPushButton(_("Connect"))
        btn_rob.setToolTip("Connect to the selected IP")
        btn_rob.clicked.connect(self.OnRobotConnect)
        self.btn_rob = btn_rob

        status_text = QLabel("Status")
        self.status_text = status_text

        btn_rob_con = QPushButton(_("Register"))
        btn_rob_con.setToolTip("Register robot tracking")
        btn_rob_con.clicked.connect(self.OnRobotRegister)

        if self.robot.IsConnected():
            self.status_text.setText(_("Robot is connected!"))
            if self.matrix_tracker_to_robot is not None:
                btn_rob_con.setText("Register Again")
        else:
            self.status_text.setText(_("Robot is not connected!"))
            btn_rob_con.hide()

        self.btn_rob_con = btn_rob_con

        rob_ip_layout = QHBoxLayout()
        rob_ip_layout.addWidget(lbl_rob)
        rob_ip_layout.addWidget(btn_rob_add_ip)
        rob_ip_layout.addWidget(btn_rob_rem_ip)
        rob_ip_layout.addWidget(choice_IP, 1)
        rob_ip_layout.addWidget(btn_rob)

        rob_status_layout = QHBoxLayout()
        rob_status_layout.addWidget(status_text, 1)
        rob_status_layout.addStretch(1)
        rob_status_layout.addWidget(btn_rob_con)

        # Pressure force setpoint
        self.pressure_recommended = 10.0
        self.pressure_match_tol = 0.1

        self.pressure_min = 0.0
        self.pressure_max = 40.0
        self.pressure_step = 1
        self.pressure_scale = int(1 / self.pressure_step)

        self.pressure_setpoint = self.session.GetConfig(
            "pressure_setpoint", self.pressure_recommended
        )
        self.pressure_setpoint = max(
            self.pressure_min, min(self.pressure_max, float(self.pressure_setpoint))
        )

        self.pressure_lbl = QLabel(_("Pressure setpoint (N):"))
        self.pressure_val_lbl = QLabel(f"{self.pressure_setpoint:.1f} N")

        self.pressure_warn_threshold = 20.0
        self._pressure_lbl_default_fg = self.pressure_lbl.palette().color(QPalette.WindowText)
        self._pressure_val_default_fg = self.pressure_val_lbl.palette().color(QPalette.WindowText)

        self.pressure_rec_lbl = QLabel(
            _("Recommended: {value} N").format(value=f"{self.pressure_recommended:.1f}")
        )
        f = self.pressure_rec_lbl.font()
        f.setItalic(True)
        ps = f.pointSize()
        if ps > 1:
            f.setPointSize(ps - 1)
        self.pressure_rec_lbl.setFont(f)
        self.pressure_rec_lbl.setStyleSheet("color: rgb(90, 90, 90);")

        self._apply_pressure_color(self.pressure_setpoint)

        self.pressure_slider = QSlider(Qt.Horizontal)
        self.pressure_slider.setRange(
            int(self.pressure_min * self.pressure_scale),
            int(self.pressure_max * self.pressure_scale),
        )
        self.pressure_slider.setValue(int(self.pressure_setpoint * self.pressure_scale))
        self.pressure_slider.setToolTip(_("Set the desired pressure/force setpoint"))
        self.pressure_slider.setTickPosition(QSlider.TicksBelow)
        self.pressure_slider.setTickInterval(self.pressure_scale)
        self.pressure_slider.valueChanged.connect(self.OnPressureSlider)

        self.btn_set_rec = QPushButton(_("Set 10 N"))
        self.btn_set_rec.setFixedSize(70, 23)
        self.btn_set_rec.setToolTip(_("Set pressure to the recommended 5.0 N"))
        self.btn_set_rec.clicked.connect(self.OnSetRecommendedPressure)

        # Pressure sensor
        self.chk_enable_pressure = QCheckBox(_("Enable pressure sensor"))
        if getattr(self.robot, "robot_init_config", None):
            use_pressure_sensor = self.robot.robot_init_config.get("use_pressure_sensor", False)
        else:
            Publisher.sendMessage("Neuronavigation to Robot: Request config")
            use_pressure_sensor = False
        self.chk_enable_pressure.setChecked(use_pressure_sensor)
        self.chk_enable_pressure.stateChanged.connect(self.OnTogglePressureSensor)
        self.chk_enable_pressure.setEnabled(self.robot.IsConnected())
        self._update_pressure_controls_state(self.robot.IsConnected() and use_pressure_sensor)

        pressure_row = QHBoxLayout()
        pressure_row.addWidget(self.pressure_lbl)
        pressure_row.addWidget(self.pressure_slider, 1)
        pressure_row.addWidget(self.pressure_val_lbl)
        pressure_row.addWidget(self.btn_set_rec)

        pressure_hint_row = QHBoxLayout()
        pressure_hint_row.addSpacing(self.pressure_lbl.sizeHint().width())
        pressure_hint_row.addWidget(self.pressure_rec_lbl)

        pressure_box = QGroupBox(_("Pressure Control"))
        pressure_layout = QVBoxLayout()
        pressure_layout.addWidget(self.chk_enable_pressure)
        pressure_layout.addLayout(pressure_row)
        pressure_layout.addLayout(pressure_hint_row)
        pressure_box.setLayout(pressure_layout)

        rob_static_sizer = QGroupBox(_("Setup robot"))
        rob_layout = QVBoxLayout()
        rob_layout.addLayout(rob_ip_layout)
        rob_layout.addLayout(rob_status_layout)
        rob_layout.addWidget(pressure_box)
        rob_static_sizer.setLayout(rob_layout)

        main_sizer = QVBoxLayout(self)
        main_sizer.addWidget(sizer)
        main_sizer.addWidget(rob_static_sizer)
        self.setLayout(main_sizer)

        Publisher.sendMessage("Neuronavigation to Robot: Check connection robot")

    def __bind_events(self):
        Publisher.subscribe(self.ShowParent, "Show preferences dialog")
        Publisher.subscribe(self.OnRobotStatus, "Robot to Neuronavigation: Robot connection status")
        Publisher.subscribe(
            self.OnSetRobotTransformationMatrix,
            "Neuronavigation to Robot: Set robot transformation matrix",
        )
        Publisher.subscribe(self.OnRobotConfigReceived, "Robot to Neuronavigation: Initial config")

    def LoadConfig(self):
        session = ses.Session()
        self.n_coils = session.GetConfig("navigation", {}).get("n_coils", 1)

        state = session.GetConfig("robot", {})

        self.robot_ip = state.get("robot_ip", None)
        self.matrix_tracker_to_robot = state.get("tracker_to_robot", None)
        if self.matrix_tracker_to_robot is not None:
            self.matrix_tracker_to_robot = np.array(self.matrix_tracker_to_robot)

    def OnRobotConfigReceived(self, config):
        use_pressure_sensor = config.get("use_pressure_sensor", False)
        self.chk_enable_pressure.setChecked(use_pressure_sensor)
        self._update_pressure_controls_state(self.robot.IsConnected() and use_pressure_sensor)

    def OnChooseNoOfCoils(self, index=None, ctrl=None):
        old_n_coils = self.n_coils
        if index is not None:
            self.n_coils = index + 1
        else:
            self.n_coils = 1

        if self.n_coils != old_n_coils:
            tracker_id = self.tracker.tracker_id
            self.tracker.DisconnectTracker()
            self.tracker.SetTracker(tracker_id, n_coils=self.n_coils)

        ctrl.setCurrentIndex(self.n_coils - 1)
        Publisher.sendMessage("Reset coil selection", n_coils=self.n_coils)
        Publisher.sendMessage("Coil selection done", done=False)

    def OnChooseTracker(self, index=None, ctrl=None):
        if sys.platform == "darwin":
            QTimer.singleShot(0, lambda: self.window().hide())
        else:
            self.HideParent()
        Publisher.sendMessage("Begin busy cursor")
        Publisher.sendMessage("Update status text in GUI", label=_("Configuring tracker ..."))

        choice = index

        self.tracker.DisconnectTracker()
        self.tracker.ResetTrackerFiducials()
        self.tracker.SetTracker(choice, n_coils=self.n_coils)
        Publisher.sendMessage("Update status text in GUI", label=_("Ready"))
        Publisher.sendMessage("Tracker changed")
        Publisher.sendMessage("Reset coil selection", n_coils=self.n_coils)
        Publisher.sendMessage("Coil selection done", done=False)
        ctrl.setCurrentIndex(self.tracker.tracker_id)
        Publisher.sendMessage("End busy cursor")
        if sys.platform == "darwin":
            QTimer.singleShot(0, lambda: self.window().show())
        else:
            self.ShowParent()

    def OnChooseReferenceMode(self, index=None, ctrl=None):
        Navigation(None, None).SetReferenceMode(index)

    def HideParent(self):
        w = self.window()
        if w:
            w.hide()

    def ShowParent(self):
        w = self.window()
        if w:
            w.show()

    def verifyFormatIP(self, robot_ip):
        robot_ip_strip = robot_ip.strip()
        full_ip_pattern = re.compile(
            r"^(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
            r"\.(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
            r"\.(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)"
            r"\.(25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)$"
        )

        if full_ip_pattern.match(robot_ip_strip):
            return True
        else:
            self.status_text.setText(_("Robot is not connected and invalid IP!"))
            return False

    def OnTxt_Ent(self, text=None, ctrl=None):
        robot_ip_input = ctrl.currentText()
        robot_ip_input = re.sub(r"[^0-9.]", "", robot_ip_input)
        ctrl.setEditText(robot_ip_input)

        msg_box = _("Select or type robot IP:")

        if robot_ip_input == "":
            ctrl.setEditText(msg_box)
        else:
            self.btn_rob_con.hide()
            self.robot_ip = robot_ip_input
            if self.verifyFormatIP(self.robot_ip):
                self.status_text.setText(_("Robot is not connected!"))

    def OnChoiceIP(self, index=None, ctrl=None):
        self.robot_ip = ctrl.currentText()

    def OnAddIP(self):
        if self.robot_ip is not None:
            new_ip = self.choice_IP.currentText()

            if new_ip is not None and self.verifyFormatIP(new_ip):
                if new_ip not in self.robot.robot_ip_options:
                    self.choice_IP.addItem(new_ip)
                    self.robot.robot_ip_options.append(new_ip)
                    self.robot.SaveConfig("robot_ip_options", self.robot.robot_ip_options)
                else:
                    self.choice_IP.setCurrentIndex(self.robot.robot_ip_options.index(new_ip))
            else:
                self.status_text.setText(_("Please select or enter valid IP!"))

    def OnRemoveIP(self):
        if self.robot_ip is not None:
            current_ip = self.choice_IP.currentText()

            result = QMessageBox.question(
                self,
                _("Confirmation"),
                _(f"Do you really want to remove the IP '{current_ip}' from the list?"),
                QMessageBox.Yes | QMessageBox.No,
            )

            if result == QMessageBox.Yes:
                try:
                    index = self.choice_IP.findText(current_ip)
                    if index >= 0:
                        self.choice_IP.removeItem(index)

                    self.robot.robot_ip_options.remove(current_ip)
                    self.robot.SaveConfig("robot_ip_options", self.robot.robot_ip_options)

                    if self.choice_IP.count() > 0:
                        self.choice_IP.setCurrentIndex(0)
                    else:
                        self.choice_IP.setEditText("")

                except Exception as e:
                    QMessageBox.critical(
                        self,
                        "Erro",
                        _(f"An error occurred while removing the IP:\n{e}"),
                    )

    def OnRobotConnect(self):
        if self.robot_ip is not None and self.verifyFormatIP(self.robot_ip):
            self.robot.is_robot_connected = False
            self.status_text.setText(_("Trying to connect to robot..."))
            self.btn_rob_con.hide()
            self.robot.SetRobotIP(self.robot_ip)
            Publisher.sendMessage(
                "Neuronavigation to Robot: Connect to robot", robot_IP=self.robot_ip
            )
        else:
            self.status_text.setText(_("Please select or enter valid IP before connecting!"))

    def OnRobotRegister(self):
        if sys.platform == "darwin":
            QTimer.singleShot(0, lambda: self.window().hide())
        else:
            self.HideParent()
        self.robot.RegisterRobot()
        if sys.platform == "darwin":
            QTimer.singleShot(0, lambda: self.window().show())
        else:
            self.ShowParent()

    def OnRobotStatus(self, data):
        if data == "Connected":
            self.robot.is_robot_connected = True
            self.status_text.setText(_("Setup robot transformation matrix:"))
            self.btn_rob_con.show()
            self.chk_enable_pressure.setEnabled(True)
            self.chk_enable_pressure.setChecked(
                self.robot.robot_init_config.get("use_pressure_sensor", False)
            )
            self._update_pressure_controls_state(self.chk_enable_pressure.isChecked())

            if (
                self.robot.robot_ip not in self.robot.robot_ip_options
                and self.robot.robot_ip is not None
            ):
                self.robot.robot_ip_options.append(self.robot.robot_ip)
        else:
            if self.robot.robot_ip is not None:
                self.status_text.setText(_(f"{data} to robot on {self.robot.robot_ip}"))
            else:
                self.status_text.setText(_(f"{data} to robot"))
            self.btn_rob_con.hide()
            self.chk_enable_pressure.setEnabled(False)
            self._update_pressure_controls_state(False)

    def OnSetRobotTransformationMatrix(self, data):
        if self.robot.matrix_tracker_to_robot is not None:
            self.status_text.setText("Robot is fully setup!")
            self.btn_rob_con.setText("Register Again")
            self.btn_rob_con.show()

    def OnPressureSlider(self, val_i):
        value = val_i / self.pressure_scale

        self.pressure_val_lbl.setText(f"{value:.1f} N")
        self._apply_pressure_color(value)
        self.pressure_setpoint = value

        try:
            self.session.SetConfig("pressure_setpoint", value)
        except Exception:
            pass

        try:
            if hasattr(self.robot, "SetPressureSetpoint"):
                self.robot.SetPressureSetpoint(value)
        except Exception:
            pass

    def OnSetRecommendedPressure(self):
        val_i = int(round(self.pressure_recommended * self.pressure_scale))
        self.pressure_slider.setValue(val_i)

        value = val_i / self.pressure_scale
        self.pressure_val_lbl.setText(f"{value:.1f} N")
        self._apply_pressure_color(value)
        self.pressure_setpoint = value
        try:
            self.session.SetConfig("pressure_setpoint", value)
        except Exception:
            pass
        try:
            if hasattr(self.robot, "SetPressureSetpoint"):
                self.robot.SetPressureSetpoint(value)
        except Exception:
            pass

    def _update_pressure_controls_state(self, slider_enabled: bool):
        """Enable/disable slider, value, button, and label colors."""
        self.pressure_slider.setEnabled(slider_enabled)
        self.btn_set_rec.setEnabled(slider_enabled)

        if slider_enabled:
            self._set_label_color(self.pressure_lbl, self._pressure_lbl_default_fg)
            self._set_label_color(self.pressure_val_lbl, self._pressure_val_default_fg)
            self._apply_pressure_color(self.pressure_setpoint)
        else:
            gray_color = QColor(150, 150, 150)
            self._set_label_color(self.pressure_lbl, gray_color)
            self._set_label_color(self.pressure_val_lbl, gray_color)
            self.pressure_rec_lbl.setStyleSheet("color: rgb(150, 150, 150);")

    def OnTogglePressureSensor(self, state):
        if not self.robot.robot_init_config:
            print("Robot init config not loaded")
            Publisher.sendMessage("Neuronavigation to Robot: Request config")
            self.chk_enable_pressure.setChecked(False)
            return

        enabled = self.chk_enable_pressure.isChecked()
        self._update_pressure_controls_state(enabled)

        self.robot.robot_init_config["use_pressure_sensor"] = enabled

        Publisher.sendMessage("Set visibility robot force visualizer", visible=enabled)
        Publisher.sendMessage(
            "Neuronavigation to Robot: Update config", use_pressure_sensor=enabled
        )

    def _set_label_color(self, label, color):
        palette = label.palette()
        palette.setColor(QPalette.WindowText, color)
        label.setPalette(palette)

    def _apply_pressure_color(self, value: float):
        if value > self.pressure_warn_threshold:
            warn_color = QColor(255, 0, 0)
            self._set_label_color(self.pressure_lbl, warn_color)
            self._set_label_color(self.pressure_val_lbl, warn_color)
        else:
            self._set_label_color(self.pressure_lbl, self._pressure_lbl_default_fg)
            self._set_label_color(self.pressure_val_lbl, self._pressure_val_default_fg)

        if abs(value - self.pressure_recommended) <= self.pressure_match_tol:
            self.pressure_rec_lbl.setStyleSheet("color: rgb(0, 140, 0); font-weight: bold;")
        else:
            self.pressure_rec_lbl.setStyleSheet("color: rgb(90, 90, 90); font-weight: normal;")


class LanguageTab(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        bsizer = QGroupBox(_("Language"))
        bsizer_layout = QVBoxLayout()

        self.lg = lg = ComboBoxLanguage(bsizer)
        self.cmb_lang = cmb_lang = lg.GetComboBox()
        text = QLabel(_("Language settings will be applied \n the next time InVesalius starts."))
        bsizer_layout.addWidget(cmb_lang)
        bsizer_layout.addSpacing(5)
        bsizer_layout.addWidget(text)
        bsizer.setLayout(bsizer_layout)

        border = QVBoxLayout(self)
        border.addWidget(bsizer, 1)
        self.setLayout(border)

    def GetSelection(self):
        selection = self.cmb_lang.currentIndex()
        locales = self.lg.GetLocalesKey()
        options = {const.LANGUAGE: locales[selection]}
        return options

    def LoadSelection(self, values):
        language = values[const.LANGUAGE]
        locales = self.lg.GetLocalesKey()
        selection = locales.index(language)
        self.cmb_lang.setCurrentIndex(int(selection))
