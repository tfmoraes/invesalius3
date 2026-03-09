#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import multiprocessing
import time
from typing import Dict

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFontMetrics
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

import invesalius.data.slice_ as slc
from invesalius.gui import dialogs
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher
from invesalius.segmentation.deep_learning import segment

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import tinygrad

    HAS_TINYGRAD = True
except ImportError:
    HAS_TINYGRAD = False

if HAS_TORCH:
    TORCH_DEVICES = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name()
            device_id = f"cuda:{i}"
            TORCH_DEVICES[name] = device_id
    TORCH_DEVICES["CPU"] = "cpu"

if HAS_TINYGRAD:
    TINYGRAD_DEVICES = {}
    for device in list(tinygrad.Device.get_available_devices()):
        TINYGRAD_DEVICES[device] = device
    if "DSP" in TINYGRAD_DEVICES.keys():
        del TINYGRAD_DEVICES["DSP"]


def _make_radio_group(parent, label, choices, orientation=Qt.Horizontal):
    group = QButtonGroup(parent)
    layout = QHBoxLayout() if orientation == Qt.Horizontal else QVBoxLayout()
    buttons = []
    for i, text in enumerate(choices):
        rb = QRadioButton(text, parent)
        group.addButton(rb, i)
        layout.addWidget(rb)
        buttons.append(rb)
    if buttons:
        buttons[0].setChecked(True)
    return group, layout, buttons


class DeepLearningSegmenterDialog(QDialog):
    def __init__(
        self,
        parent,
        title,
        auto_segment=False,
        has_torch=True,
        has_tinygrad=True,
        segmenter=None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)

        backends = []
        if HAS_TORCH and has_torch:
            backends.append("Pytorch")
        if HAS_TORCH and has_tinygrad:
            backends.append("Tinygrad")
        self.segmenter = segmenter

        if HAS_TORCH:
            self.torch_devices: Dict[str, str] = TORCH_DEVICES

        if HAS_TINYGRAD:
            self.tinygrad_devices: Dict[str, str] = TINYGRAD_DEVICES

        self.auto_segment = auto_segment

        self.backends = backends

        self.ps = None
        self.segmented = False
        self.mask = None

        self.overlap_options = (0, 10, 25, 50)
        self.default_overlap = 50

        self.elapsed_time_timer = QTimer(self)

        self._init_gui()
        self._do_layout()
        self._set_events()

        self.OnSetBackend()
        self.HideProgress()

        if self.auto_segment:
            self.OnSegment(self)

    def _init_gui(self):
        self.cb_backends = QComboBox(self)
        self.cb_backends.addItems(self.backends)
        if self.backends:
            self.cb_backends.setCurrentText(self.backends[0])
        fm = QFontMetrics(self.font())
        max_text = "MM" * (1 + max(len(i) for i in self.backends))
        self.cb_backends.setMinimumWidth(fm.horizontalAdvance(max_text))

        self.chk_use_gpu = QCheckBox(_("Use GPU"), self)
        choices = []
        value = ""

        if HAS_TORCH or HAS_TINYGRAD:
            if HAS_TORCH:
                choices = list(self.torch_devices.keys())
                value = choices[0]
            else:
                choices = list(self.tinygrad_devices.keys())
                value = choices[0]

        self.lbl_device = QLabel(_("Device"), self)
        self.cb_devices = QComboBox(self)
        self.cb_devices.addItems(choices)
        if value:
            self.cb_devices.setCurrentText(value)

        self.sld_threshold = QSlider(Qt.Horizontal, self)
        self.sld_threshold.setRange(0, 100)
        self.sld_threshold.setValue(75)
        w = fm.horizontalAdvance("M" * 20)
        self.sld_threshold.setMinimumWidth(w)

        self.txt_threshold = QLineEdit(self)
        w2 = fm.horizontalAdvance("MMMMM")
        self.txt_threshold.setMinimumWidth(w2)

        self.chk_new_mask = QCheckBox(_("Create new mask"), self)
        self.chk_new_mask.setChecked(True)
        self.chk_apply_wwwl = QCheckBox(_("Apply WW&WL"), self)
        self.chk_apply_wwwl.setChecked(False)

        # Overlap radio group
        self.overlap_group, self.overlap_layout, self.overlap_btns = _make_radio_group(
            self, _("Overlap"), [f"{i}%" for i in self.overlap_options]
        )
        self.overlap_group_box = QGroupBox(_("Overlap"), self)
        self.overlap_group_box.setFlat(True)
        self.overlap_group_box.setLayout(self.overlap_layout)
        self.overlap_btns[self.overlap_options.index(self.default_overlap)].setChecked(True)

        self.progress = QProgressBar(self)
        self.lbl_progress_caption = QLabel(_("Elapsed time:"), self)
        self.lbl_time = QLabel(_("00:00:00"), self)
        self.btn_segment = QPushButton(_("Segment"), self)
        self.btn_stop = QPushButton(_("Stop"), self)
        self.btn_stop.setEnabled(False)
        self.btn_close = QPushButton(_("Close"), self)

        self.txt_threshold.setText(f"{self.sld_threshold.value():3d}%")

    def _do_layout(self):
        main_sizer = QVBoxLayout(self)
        sizer_backends = QHBoxLayout()
        label_1 = QLabel(_("Backend"), self)
        sizer_backends.addWidget(label_1)
        sizer_backends.addWidget(self.cb_backends, 1)
        main_sizer.addLayout(sizer_backends)
        main_sizer.addWidget(self.chk_use_gpu)

        sizer_devices = QHBoxLayout()
        if HAS_TORCH or HAS_TINYGRAD:
            sizer_devices.addWidget(self.lbl_device)
            sizer_devices.addWidget(self.cb_devices, 1)
        main_sizer.addLayout(sizer_devices)
        main_sizer.addWidget(self.overlap_group_box)

        label_5 = QLabel(_("Level of certainty"), self)
        main_sizer.addWidget(label_5)
        sizer_3 = QHBoxLayout()
        sizer_3.addWidget(self.sld_threshold, 1)
        sizer_3.addWidget(self.txt_threshold)
        main_sizer.addLayout(sizer_3)
        main_sizer.addWidget(self.chk_apply_wwwl)
        main_sizer.addWidget(self.chk_new_mask)
        main_sizer.addWidget(self.progress)

        time_sizer = QHBoxLayout()
        time_sizer.addWidget(self.lbl_progress_caption)
        time_sizer.addWidget(self.lbl_time, 1)
        main_sizer.addLayout(time_sizer)

        sizer_buttons = QHBoxLayout()
        sizer_buttons.addStretch()
        sizer_buttons.addWidget(self.btn_close)
        sizer_buttons.addWidget(self.btn_stop)
        sizer_buttons.addWidget(self.btn_segment)
        main_sizer.addLayout(sizer_buttons)

        self.main_sizer = main_sizer
        self.setLayout(main_sizer)
        self.adjustSize()

    def _set_events(self):
        self.cb_backends.currentIndexChanged.connect(self.OnSetBackend)
        self.sld_threshold.valueChanged.connect(self.OnScrollThreshold)
        self.txt_threshold.editingFinished.connect(self.OnKillFocus)
        self.btn_segment.clicked.connect(self.OnSegment)
        self.btn_stop.clicked.connect(self.OnStop)
        self.btn_close.clicked.connect(self.OnBtnClose)
        self.elapsed_time_timer.timeout.connect(self.OnTickTimer)

    def apply_segment_threshold(self):
        threshold = self.sld_threshold.value() / 100.0
        if self.ps is not None:
            self.ps.apply_segment_threshold(threshold)
            slc.Slice().discard_all_buffers()
            Publisher.sendMessage("Reload actual slice")

    def OnSetBackend(self, index=None):
        backend_text = self.cb_backends.currentText().lower()
        if backend_text == "pytorch":
            if HAS_TORCH:
                choices = list(self.torch_devices.keys())
                self.cb_devices.clear()
                self.cb_devices.addItems(choices)
                self.cb_devices.setCurrentText(choices[0])
                self.lbl_device.show()
                self.cb_devices.show()
            self.chk_use_gpu.hide()
        elif backend_text == "tinygrad":
            if HAS_TINYGRAD:
                choices = list(self.tinygrad_devices)
                self.cb_devices.clear()
                self.cb_devices.addItems(choices)
                self.cb_devices.setCurrentText(choices[0])
                self.lbl_device.show()
                self.cb_devices.show()
            self.chk_use_gpu.hide()
        else:
            raise TypeError("Wrong backend")

        self.adjustSize()

    def OnScrollThreshold(self, value):
        self.txt_threshold.setText(f"{value:3d}%")
        if self.segmented:
            self.apply_segment_threshold()

    def OnKillFocus(self):
        value = self.txt_threshold.text()
        value = value.replace("%", "")
        try:
            value = int(value)
        except ValueError:
            value = self.sld_threshold.value()
        self.sld_threshold.setValue(value)
        self.txt_threshold.setText(f"{value:3d}%")

        if self.segmented:
            self.apply_segment_threshold()

    def OnSegment(self, evt=None):
        self.ShowProgress()
        self.t0 = time.time()
        self.elapsed_time_timer.start(1000)
        image = slc.Slice().matrix
        backend = self.cb_backends.currentText()
        if backend.lower() == "pytorch":
            try:
                device_id = self.torch_devices[self.cb_devices.currentText()]
            except (KeyError, AttributeError):
                device_id = "cpu"
        elif backend.lower() == "tinygrad":
            try:
                device_id = self.tinygrad_devices[self.cb_devices.currentText()]
            except (KeyError, AttributeError):
                device_id = tinygrad.Device.DEFAULT
        else:
            raise TypeError("Wrong backend")
        apply_wwwl = self.chk_apply_wwwl.isChecked()
        create_new_mask = self.chk_new_mask.isChecked()
        use_gpu = self.chk_use_gpu.isChecked()
        self.btn_close.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_segment.setEnabled(False)
        self.chk_new_mask.setEnabled(False)
        self.chk_apply_wwwl.setEnabled(False)

        window_width = slc.Slice().window_width
        window_level = slc.Slice().window_level

        overlap = self.overlap_options[self.overlap_group.checkedId()]

        try:
            self.ps = self.segmenter(
                image,
                create_new_mask,
                backend,
                device_id,
                use_gpu,
                overlap,
                apply_wwwl,
                window_width,
                window_level,
            )
            self.ps.start()
        except (multiprocessing.ProcessError, OSError, ValueError) as err:
            self.OnStop(None)
            self.HideProgress()
            dlg = dialogs.ErrorMessageBox(
                None,
                "It was not possible to start brain segmentation because:" + "\n" + str(err),
                "Brain segmentation error",
            )
            dlg.exec()

    def OnStop(self, evt=None):
        if self.ps is not None:
            self.ps.terminate()
        self.btn_close.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_segment.setEnabled(True)
        self.chk_new_mask.setEnabled(True)
        self.elapsed_time_timer.stop()

    def OnBtnClose(self):
        self.close()

    def AfterSegment(self):
        self.segmented = True
        self.btn_close.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_segment.setEnabled(False)
        self.chk_new_mask.setEnabled(False)
        self.sld_threshold.setEnabled(False)
        self.txt_threshold.setEnabled(False)
        self.cb_backends.setEnabled(False)
        self.cb_devices.setEnabled(False)
        self.overlap_group_box.setEnabled(False)
        self.chk_apply_wwwl.setEnabled(False)
        self.chk_use_gpu.setEnabled(False)

        self.elapsed_time_timer.stop()
        self.apply_segment_threshold()

        if self.auto_segment:
            self.close()
            Publisher.sendMessage("Brain segmentation completed")

    def SetProgress(self, progress):
        if self.progress:
            self.progress.setValue(int(progress * 100))

    def OnTickTimer(self):
        fmt = "%H:%M:%S"
        self.lbl_time.setText(time.strftime(fmt, time.gmtime(time.time() - self.t0)))
        if self.ps is not None:
            if not self.ps.is_alive() and self.ps.exception is not None:
                error, traceback = self.ps.exception
                self.OnStop(None)
                self.HideProgress()
                dlg = dialogs.ErrorMessageBox(
                    None,
                    "Brain segmentation error",
                    "It was not possible to use brain segmentation because:"
                    + "\n"
                    + str(error)
                    + "\n"
                    + traceback,
                )
                dlg.exec()
                return

            progress = self.ps.get_completion()
            if progress == np.inf or progress >= 1.0:
                progress = 1.0
                self.AfterSegment()
            else:
                progress = max(0.0, min(progress, 1.0))
            self.SetProgress(progress)

    def closeEvent(self, event):
        self.btn_stop.setEnabled(False)
        self.btn_segment.setEnabled(True)
        self.chk_new_mask.setEnabled(True)
        self.progress.setValue(0)

        if self.ps is not None:
            self.ps.terminate()
            self.ps = None

        super().closeEvent(event)

    def HideProgress(self):
        self.progress.hide()
        self.lbl_progress_caption.hide()
        self.lbl_time.hide()
        self.adjustSize()

    def ShowProgress(self):
        self.progress.show()
        self.lbl_progress_caption.show()
        self.lbl_time.show()
        self.adjustSize()


class BrainSegmenterDialog(DeepLearningSegmenterDialog):
    def __init__(self, parent, auto_segment=False):
        super().__init__(
            parent=parent,
            title=_("Brain segmentation"),
            has_torch=True,
            has_tinygrad=True,
            segmenter=segment.BrainSegmentProcess,
            auto_segment=auto_segment,
        )


class SubpartSegmenterDialog(DeepLearningSegmenterDialog):
    def __init__(self, parent, auto_segment=False):
        self.mask_types = {
            "cortical": _("Cortical"),
            "subcortical": _("Subcortical"),
            "white_matter": _("White Matter"),
            "cerebellum": _("Cerebellum"),
            "ventricles": _("Ventricles"),
            "brain_stem": _("Brain Stem"),
            "choroid_plexus": _("Choroid Plexus"),
        }

        self.selected_mask_types = []

        super().__init__(
            parent=parent,
            title=_("Brain subpart Segmentation"),
            has_torch=True,
            has_tinygrad=True,
            segmenter=segment.SubpartSegmentProcess,
            auto_segment=auto_segment,
        )

    def _init_gui(self):
        """Override _init_gui to add specific UI elements"""
        super()._init_gui()

        self.mask_types_box = QGroupBox(_("Mask Types to Generate"), self)
        self.mask_types_sizer = QVBoxLayout()

        self.chk_whole_brain = QCheckBox(_("Whole Brain"), self)
        self.chk_whole_brain.setChecked(True)
        self.chk_whole_brain.setEnabled(False)

        self.separator = QFrame(self)
        self.separator.setFrameShape(QFrame.HLine)
        self.separator.setFrameShadow(QFrame.Sunken)

        self.mask_checkboxes = {}
        for mask_id, mask_label in self.mask_types.items():
            self.mask_checkboxes[mask_id] = QCheckBox(mask_label, self)
            self.mask_checkboxes[mask_id].setChecked(False)

    def _do_layout(self):
        """Override _do_layout to arrange FastSurferCNN specific UI elements"""
        super()._do_layout()

        self.mask_types_sizer.addWidget(self.chk_whole_brain)
        self.mask_types_sizer.addWidget(self.separator)

        for checkbox in self.mask_checkboxes.values():
            self.mask_types_sizer.addWidget(checkbox)

        self.mask_types_box.setLayout(self.mask_types_sizer)

        # Insert mask types after the overlap widget
        items_count = self.main_sizer.count()
        overlap_index = -1
        for i in range(items_count):
            item = self.main_sizer.itemAt(i)
            if item and item.widget() == self.overlap_group_box:
                overlap_index = i
                break

        if overlap_index >= 0:
            self.main_sizer.insertWidget(overlap_index + 1, self.mask_types_box)
        else:
            self.main_sizer.addWidget(self.mask_types_box)

        self.adjustSize()

    def OnSegment(self, evt=None):
        self.selected_mask_types = [
            mask_id for mask_id, checkbox in self.mask_checkboxes.items() if checkbox.isChecked()
        ]

        self.ShowProgress()
        self.t0 = time.time()
        self.elapsed_time_timer.start(1000)

        image = slc.Slice().matrix
        backend = self.cb_backends.currentText()
        create_new_mask = self.chk_new_mask.isChecked()

        if backend.lower() == "pytorch":
            try:
                device_id = self.torch_devices[self.cb_devices.currentText()]
            except (KeyError, AttributeError):
                device_id = "cpu"
            use_gpu = True if "cpu" not in device_id.lower() else False
        elif backend.lower() == "tinygrad":
            try:
                device_id = self.tinygrad_devices[self.cb_devices.currentText()]
                if device_id == "GPU":
                    device_id = "cuda"
            except (KeyError, AttributeError):
                device_id = "cpu"
            use_gpu = "cuda" in device_id.lower()
        else:
            device_id = "cpu"
            use_gpu = False
        overlap = self.overlap_options[self.overlap_group.checkedId()]
        apply_wwwl = self.chk_apply_wwwl.isChecked()
        window_width = slc.Slice().window_width
        window_level = slc.Slice().window_level

        self.btn_close.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_segment.setEnabled(False)
        self.chk_new_mask.setEnabled(False)
        self.chk_use_gpu.setEnabled(False)
        self.overlap_group_box.setEnabled(False)
        self.cb_backends.setEnabled(False)
        self.cb_devices.setEnabled(False)

        for checkbox in self.mask_checkboxes.values():
            checkbox.setEnabled(False)

        try:
            self.ps = self.segmenter(
                image,
                create_new_mask,
                backend,
                device_id,
                use_gpu,
                overlap,
                apply_wwwl,
                window_width,
                window_level,
                selected_mask_types=self.selected_mask_types,
            )
            self.ps.start()
        except (multiprocessing.ProcessError, OSError, ValueError) as err:
            self.OnStop(None)
            self.HideProgress()
            dlg = dialogs.ErrorMessageBox(
                None,
                _("It was not possible to start Subpart segmentation because:") + "\n" + str(err),
                _("FastSurfer segmentation error"),
            )
            dlg.exec()

    def AfterSegment(self):
        super().AfterSegment()

        for checkbox in self.mask_checkboxes.values():
            checkbox.setEnabled(True)

    def apply_segment_threshold(self):
        threshold = self.sld_threshold.value() / 100.0
        self.ps.apply_segment_threshold(threshold)
        slc.Slice().discard_all_buffers()
        Publisher.sendMessage("Reload actual slice")

    def OnStop(self, evt=None):
        super().OnStop(evt)

        for checkbox in self.mask_checkboxes.values():
            checkbox.setEnabled(True)


class TracheaSegmenterDialog(DeepLearningSegmenterDialog):
    def __init__(self, parent):
        super().__init__(
            parent=parent,
            title=_("Trachea segmentation"),
            has_torch=True,
            segmenter=segment.TracheaSegmentProcess,
        )


class MandibleSegmenterDialog(DeepLearningSegmenterDialog):
    def __init__(self, parent):
        super().__init__(
            parent=parent,
            title=_("Mandible segmentation (CT)"),
            has_torch=True,
            segmenter=segment.MandibleCTSegmentProcess,
        )

    def _init_gui(self):
        super()._init_gui()

        self.chk_apply_resize_by_spacing = QCheckBox(_("Resize by spacing"), self)
        self.chk_apply_resize_by_spacing.setChecked(True)

        self.patch_txt = QLabel("Patch size:", self)

        patch_size = [
            "48",
            "96",
            "160",
            "192",
            "240",
            "288",
            "320",
            "336",
            "384",
            "432",
            "480",
            "528",
        ]

        self.patch_cmb = QComboBox(self)
        self.patch_cmb.addItems(patch_size)
        self.patch_cmb.setCurrentText("192")

        self.path_sizer = QHBoxLayout()
        self.path_sizer.addWidget(self.patch_txt)
        self.path_sizer.addWidget(self.patch_cmb, 2)

    def _do_layout(self):
        super()._do_layout()
        items_count = self.main_sizer.count()
        insert_idx = min(8, items_count)
        self.main_sizer.insertWidget(insert_idx, self.chk_apply_resize_by_spacing)
        self.main_sizer.insertLayout(insert_idx + 1, self.path_sizer)
        self.adjustSize()

    def OnSegment(self, evt=None):
        self.ShowProgress()
        self.t0 = time.time()
        self.elapsed_time_timer.start(1000)
        image = slc.Slice().matrix
        backend = self.cb_backends.currentText()
        if backend.lower() == "pytorch":
            try:
                device_id = self.torch_devices[self.cb_devices.currentText()]
            except (KeyError, AttributeError):
                device_id = "cpu"
        else:
            raise TypeError("Wrong backend")
        apply_wwwl = self.chk_apply_wwwl.isChecked()
        create_new_mask = self.chk_new_mask.isChecked()
        use_gpu = self.chk_use_gpu.isChecked()
        resize_by_spacing = self.chk_apply_resize_by_spacing.isChecked()

        self.btn_close.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_segment.setEnabled(False)
        self.chk_new_mask.setEnabled(False)
        self.chk_apply_resize_by_spacing.setEnabled(False)

        window_width = slc.Slice().window_width
        window_level = slc.Slice().window_level

        overlap = self.overlap_options[self.overlap_group.checkedId()]
        patch_size = int(self.patch_cmb.currentText())

        try:
            self.ps = self.segmenter(
                image,
                create_new_mask,
                backend,
                device_id,
                use_gpu,
                overlap,
                apply_wwwl,
                window_width,
                window_level,
                patch_size=patch_size,
                resize_by_spacing=resize_by_spacing,
                image_spacing=slc.Slice().spacing,
            )
            self.ps.start()
        except (multiprocessing.ProcessError, OSError, ValueError) as err:
            self.OnStop(None)
            self.HideProgress()
            dlg = dialogs.ErrorMessageBox(
                None,
                "It was not possible to start brain segmentation because:" + "\n" + str(err),
                "Brain segmentation error",
            )
            dlg.exec()

    def OnStop(self, evt=None):
        super().OnStop(evt)
        self.chk_apply_resize_by_spacing.setEnabled(True)


class ImplantSegmenterDialog(DeepLearningSegmenterDialog):
    def __init__(self, parent):
        super().__init__(
            parent=parent,
            title=_("Implant prediction (CT)"),
            has_torch=True,
            segmenter=segment.ImplantCTSegmentProcess,
        )

    def _init_gui(self):
        super()._init_gui()

        self.patch_txt = QLabel("Patch size:", self)

        patch_size = [
            "48",
            "96",
            "160",
            "192",
            "240",
            "288",
            "320",
            "336",
            "384",
            "432",
            "480",
            "528",
        ]

        self.patch_cmb = QComboBox(self)
        self.patch_cmb.addItems(patch_size)
        self.patch_cmb.setCurrentText("480")

        self.path_sizer = QHBoxLayout()
        self.path_sizer.addWidget(self.patch_txt)
        self.path_sizer.addWidget(self.patch_cmb, 2)

        self.method_group, self.method_layout, self.method_btns = _make_radio_group(
            self, "Method:", ["Binary", "Gray"]
        )
        self.method_box = QGroupBox("Method:", self)
        self.method_box.setFlat(True)
        self.method_box.setLayout(self.method_layout)

    def _do_layout(self):
        super()._do_layout()
        items_count = self.main_sizer.count()
        insert_idx = min(8, items_count)
        self.main_sizer.insertLayout(insert_idx, self.path_sizer)
        self.main_sizer.insertWidget(insert_idx + 1, self.method_box)
        self.adjustSize()

    def OnSegment(self, evt=None):
        self.ShowProgress()
        self.t0 = time.time()
        self.elapsed_time_timer.start(1000)
        image = slc.Slice().matrix
        backend = self.cb_backends.currentText()
        if backend.lower() == "pytorch":
            try:
                device_id = self.torch_devices[self.cb_devices.currentText()]
            except (KeyError, AttributeError):
                device_id = "cpu"
        else:
            raise TypeError("Wrong backend")
        apply_wwwl = self.chk_apply_wwwl.isChecked()
        create_new_mask = self.chk_new_mask.isChecked()
        use_gpu = self.chk_use_gpu.isChecked()
        method = self.method_group.checkedId()

        self.btn_close.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.btn_segment.setEnabled(False)
        self.chk_new_mask.setEnabled(False)

        window_width = slc.Slice().window_width
        window_level = slc.Slice().window_level

        overlap = self.overlap_options[self.overlap_group.checkedId()]

        patch_size = int(self.patch_cmb.currentText())

        try:
            self.ps = self.segmenter(
                image,
                create_new_mask,
                backend,
                device_id,
                use_gpu,
                overlap,
                apply_wwwl,
                window_width,
                window_level,
                method=method,
                patch_size=patch_size,
                resize_by_spacing=True,
                image_spacing=slc.Slice().spacing,
            )
            self.ps.start()
        except (multiprocessing.ProcessError, OSError, ValueError) as err:
            self.OnStop(None)
            self.HideProgress()
            dlg = dialogs.ErrorMessageBox(
                None,
                "It was not possible to start brain segmentation because:" + "\n" + str(err),
                "Brain segmentation error",
            )
            dlg.exec()

    def OnStop(self, evt=None):
        super().OnStop(evt)
