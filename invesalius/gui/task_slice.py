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
import os
import sys

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.data.slice_ as slice_
import invesalius.gui.dialogs as dlg
import invesalius.gui.widgets.gradient as grad
import invesalius.session as ses
from invesalius import inv_paths
from invesalius.gui.default_viewers import ColourSelectButton
from invesalius.gui.widgets.inv_spinctrl import InvFloatSpinCtrl, InvSpinCtrl
from invesalius.i18n import tr as _
from invesalius.project import Project
from invesalius.pubsub import pub as Publisher

MASK_LIST = []


class TaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        inner_panel = InnerTaskPanel(self)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(7, 0, 7, 7)
        layout.addWidget(inner_panel, 1)


class InnerTaskPanel(QScrollArea):
    def __init__(self, parent):
        super().__init__(parent)
        self.select_all_active = False
        Publisher.subscribe(self.update_create_surface_button, "Update create surface button")

        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        container.setStyleSheet("background-color: white;")

        # Button for creating new mask
        BMP_ADD = QPixmap(os.path.join(inv_paths.ICON_DIR, "object_add.png"))
        button_new_mask = QPushButton("")
        button_new_mask.setIcon(QIcon(BMP_ADD))
        button_new_mask.setFlat(True)
        button_new_mask.clicked.connect(self.OnLinkNewMask)

        tooltip = _("Create mask for slice segmentation and editing")
        link_new_mask = QPushButton(_("Create new mask"))
        link_new_mask.setFlat(True)
        link_new_mask.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; color: black; }"
        )
        link_new_mask.setToolTip(tooltip)
        link_new_mask.clicked.connect(self.OnLinkNewMask)

        Publisher.subscribe(self.OnLinkNewMask, "New mask from shortcut")

        line_new = QHBoxLayout()
        line_new.addWidget(link_new_mask, 1)
        line_new.addWidget(button_new_mask, 0)

        fold_panel = FoldPanel(container)
        self.fold_panel = fold_panel

        self.button_next = QPushButton(_("Create surface"))
        check_box = QCheckBox(_("Overwrite last surface"))
        self.check_box = check_box
        self.button_next.clicked.connect(self.OnButtonNextTask)

        next_btn_sizer = QVBoxLayout()
        next_btn_sizer.addWidget(self.button_next, 0, Qt.AlignRight)

        line_sizer = QHBoxLayout()
        line_sizer.addWidget(check_box, 0, Qt.AlignLeft)
        line_sizer.setContentsMargins(5, 0, 5, 0)
        line_sizer.addLayout(next_btn_sizer, 1)

        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.addLayout(line_new)
        main_layout.addWidget(fold_panel, 1)
        main_layout.addLayout(line_sizer)
        main_layout.addSpacing(5)

        self.setWidget(container)

    def OnButton(self):
        self.OnLinkNewMask()

    def update_create_surface_button(self, select_all_active):
        """Update the create surface button text based on select all state"""
        self.select_all_active = select_all_active

        if select_all_active:
            self.button_next.setText("Create All Surfaces")
            self.button_next.setToolTip("Create surfaces for all selected masks")
        else:
            self.button_next.setText("Create Surface")
            self.button_next.setToolTip("Create surface from selected mask")

    def get_current_surface_parameters(self):
        """Get current surface creation parameters from the UI"""
        overwrite = self.check_box.isChecked()
        return {
            "method": {
                "algorithm": "Default",
                "options": {},
            },
            "options": {
                "index": None,
                "name": "",
                "quality": _("Optimal *"),
                "fill": False,
                "keep_largest": False,
                "overwrite": overwrite,
            },
        }

    def OnButtonNextTask(self):
        if self.select_all_active:
            dlgs = dlg.SurfaceDialog()
            if dlgs.exec() == QDialog.Accepted:
                algorithm = dlgs.GetAlgorithmSelected()
                options = dlgs.GetOptions()
                surface_parameters_template = self.get_current_surface_parameters()
                surface_parameters_template["method"]["algorithm"] = algorithm
                surface_parameters_template["method"]["options"] = options

                Publisher.sendMessage(
                    "Create surfaces for all masks",
                    surface_template=surface_parameters_template,
                )
            return

        overwrite = self.check_box.isChecked()
        algorithm = "Default"
        options = {}
        to_generate = True
        if self.GetMaskSelected() != -1:
            sl = slice_.Slice()
            if sl.current_mask.was_edited:
                dlgs = dlg.SurfaceDialog()
                if dlgs.exec() == QDialog.Accepted:
                    algorithm = dlgs.GetAlgorithmSelected()
                    options = dlgs.GetOptions()
                else:
                    to_generate = False

            if to_generate:
                proj = Project()
                for idx in proj.mask_dict:
                    if proj.mask_dict[idx] is sl.current_mask:
                        mask_index = idx
                        break
                else:
                    return

                mask_name = ""
                if hasattr(self.fold_panel, "inner_panel") and hasattr(
                    self.fold_panel.inner_panel, "mask_prop_panel"
                ):
                    mask_prop_panel = self.fold_panel.inner_panel.mask_prop_panel
                    if hasattr(mask_prop_panel, "combo_mask_name"):
                        mask_selection = mask_prop_panel.combo_mask_name.currentIndex()
                        if mask_selection >= 0:
                            mask_name = mask_prop_panel.combo_mask_name.itemText(mask_selection)

                method = {"algorithm": algorithm, "options": options}
                srf_options = {
                    "index": mask_index,
                    "name": mask_name,
                    "quality": _("Optimal *"),
                    "fill": False,
                    "keep_largest": False,
                    "overwrite": overwrite,
                }

                Publisher.sendMessage(
                    "Create surface from index",
                    surface_parameters={"method": method, "options": srf_options},
                )
                Publisher.sendMessage("Fold surface task")

        else:
            dlg.InexistentMask()

    def OnLinkNewMask(self, evt=None):
        try:
            evt.data
            evt = None
        except Exception:
            pass

        dialog = dlg.NewMask()

        try:
            if dialog.exec() == QDialog.Accepted:
                ok = 1
            else:
                ok = 0
        except Exception:
            ok = 1

        if ok:
            mask_name, thresh, colour = dialog.GetValue()
            if mask_name:
                Publisher.sendMessage(
                    "Create new mask", mask_name=mask_name, thresh=thresh, colour=colour
                )

    def GetMaskSelected(self):
        return self.fold_panel.GetMaskSelected()


class FoldPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        inner_panel = InnerFoldPanel(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(inner_panel, 1)

        self.inner_panel = inner_panel

    def GetMaskSelected(self):
        return self.inner_panel.GetMaskSelected()


class InnerFoldPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.last_size = None

        tool_box = QToolBox(self)
        self.tool_box = tool_box

        self.mask_prop_panel = MaskProperties(tool_box)
        tool_box.addItem(self.mask_prop_panel, _("Mask properties"))

        etw = EditionTools(tool_box)
        self.__idx_editor = tool_box.addItem(etw, _("Manual edition"))

        wtw = WatershedTool(tool_box)
        self.__idx_watershed = tool_box.addItem(wtw, _("Watershed"))

        tool_box.setCurrentIndex(0)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(tool_box, 1)

        self.last_style = None
        self.last_panel_opened = None

        self.__bind_evt()
        self.__bind_pubsub_evt()

    def __bind_evt(self):
        self.tool_box.currentChanged.connect(self.OnFoldPressCaption)

    def __bind_pubsub_evt(self):
        Publisher.subscribe(self.OnRetrieveStyle, "Retrieve task slice style")
        Publisher.subscribe(self.OnDisableStyle, "Disable task slice style")
        Publisher.subscribe(self.OnCloseProject, "Close project data")
        Publisher.subscribe(self.OnColapsePanel, "Show panel")

    def OnFoldPressCaption(self, index):
        if index == self.__idx_editor:
            Publisher.sendMessage("Enable style", style=const.SLICE_STATE_EDITOR)
            self.last_style = const.SLICE_STATE_EDITOR
        elif index == self.__idx_watershed:
            Publisher.sendMessage("Enable style", style=const.SLICE_STATE_WATERSHED)
            self.last_style = const.SLICE_STATE_WATERSHED
        else:
            Publisher.sendMessage("Disable style", style=const.SLICE_STATE_EDITOR)
            self.last_style = None

    def OnRetrieveStyle(self):
        if self.last_style == const.SLICE_STATE_EDITOR:
            Publisher.sendMessage("Enable style", style=const.SLICE_STATE_EDITOR)

    def OnDisableStyle(self):
        if self.last_style == const.SLICE_STATE_EDITOR:
            Publisher.sendMessage("Disable style", style=const.SLICE_STATE_EDITOR)

    def OnCloseProject(self):
        self.tool_box.setCurrentIndex(0)

    def OnColapsePanel(self, panel_id):
        panel_seg_id = {
            const.ID_THRESHOLD_SEGMENTATION: 0,
            const.ID_MANUAL_SEGMENTATION: 1,
            const.ID_WATERSHED_SEGMENTATION: 2,
        }

        try:
            _id = panel_seg_id[panel_id]
            self.tool_box.setCurrentIndex(_id)
        except KeyError:
            pass

    def GetMaskSelected(self):
        return self.mask_prop_panel.GetMaskSelected()


class MaskProperties(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        ## LINE 1
        combo_mask_name = QComboBox(self)
        combo_mask_name.setEditable(False)
        self.combo_mask_name = combo_mask_name

        button_colour = ColourSelectButton(self, colour=(0, 255, 0), size=(22, 22))
        self.button_colour = button_colour

        line1 = QHBoxLayout()
        line1.addWidget(combo_mask_name, 1)
        line1.addWidget(button_colour, 0)

        ### LINE 2
        text_thresh = QLabel(_("Set predefined or manual threshold:"))

        ### LINE 3
        THRESHOLD_LIST = [""]
        combo_thresh = QComboBox(self)
        combo_thresh.setEditable(False)
        combo_thresh.addItems(THRESHOLD_LIST)
        combo_thresh.setCurrentIndex(0)
        self.combo_thresh = combo_thresh

        ## LINE 4
        gradient = grad.GradientCtrl(self, -1, -5000, 5000, 0, 5000, (0, 255, 0, 100))
        self.gradient = gradient

        layout = QVBoxLayout(self)
        layout.addSpacing(7)
        layout.addLayout(line1)
        layout.addSpacing(5)
        layout.addWidget(text_thresh)
        layout.addSpacing(2)
        layout.addWidget(combo_thresh)
        layout.addSpacing(5)
        layout.addWidget(gradient, 1)
        layout.addSpacing(7)

        proj = Project()
        self.threshold_modes = proj.threshold_modes
        self.threshold_modes_names = []
        self.bind_evt_gradient = True
        self.__bind_events()
        self.__bind_events_wx()

    def __bind_events(self):
        Publisher.subscribe(self.AddMask, "Add mask")
        Publisher.subscribe(self.SetThresholdBounds, "Update threshold limits")
        Publisher.subscribe(self.SetThresholdModes, "Set threshold modes")
        Publisher.subscribe(self.SetItemsColour, "Set GUI items colour")
        Publisher.subscribe(self.SetThresholdValues, "Set threshold values in gradient")
        Publisher.subscribe(self.SelectMaskName, "Select mask name in combo")
        Publisher.subscribe(self.ChangeMaskName, "Change mask name")
        Publisher.subscribe(self.OnRemoveMasks, "Remove masks")
        Publisher.subscribe(self.OnCloseProject, "Close project data")
        Publisher.subscribe(self.SetThresholdValues2, "Set threshold values")

    def OnCloseProject(self):
        self.CloseProject()

    def CloseProject(self):
        self.combo_mask_name.clear()
        self.combo_thresh.clear()

    def OnRemoveMasks(self, mask_indexes):
        self.combo_mask_name.blockSignals(True)
        try:
            count = self.combo_mask_name.count()
            if len(mask_indexes) >= count:
                self.combo_mask_name.clear()
            else:
                for i in sorted(set(mask_indexes), reverse=True):
                    if 0 <= i < self.combo_mask_name.count():
                        self.combo_mask_name.removeItem(i)
        finally:
            self.combo_mask_name.blockSignals(False)

        if self.combo_mask_name.count() == 0:
            self.combo_mask_name.setCurrentText("")
            self.setEnabled(False)

    def __bind_events_wx(self):
        self.gradient.threshold_changed.connect(self.OnSlideChanged)
        self.gradient.threshold_changing.connect(self.OnSlideChanging)
        self.combo_thresh.currentIndexChanged.connect(self.OnComboThresh)
        self.combo_mask_name.currentIndexChanged.connect(self.OnComboName)
        self.button_colour.colour_selected.connect(self.OnSelectColour)

    def SelectMaskName(self, index):
        if index >= 0:
            self.combo_mask_name.setCurrentIndex(index)
        else:
            self.combo_mask_name.setCurrentText("")

    def ChangeMaskName(self, index, name):
        self.combo_mask_name.setItemText(index, name)

    def SetThresholdValues(self, threshold_range):
        thresh_min, thresh_max = threshold_range
        self.bind_evt_gradient = False
        self.gradient.SetMinValue(thresh_min)
        self.gradient.SetMaxValue(thresh_max)

        self.bind_evt_gradient = True
        thresh = (thresh_min, thresh_max)
        if thresh in Project().threshold_modes.values():
            preset_name = Project().threshold_modes.get_key(thresh)
            index = self.threshold_modes_names.index(preset_name)
            self.combo_thresh.setCurrentIndex(index)
        else:
            index = self.threshold_modes_names.index(_("Custom"))
            self.combo_thresh.setCurrentIndex(index)
            Project().threshold_modes[_("Custom")] = (thresh_min, thresh_max)

    def SetThresholdValues2(self, threshold_range):
        thresh_min, thresh_max = threshold_range
        self.gradient.SetMinValue(thresh_min)
        self.gradient.SetMaxValue(thresh_max)
        thresh = (thresh_min, thresh_max)
        if thresh in Project().threshold_modes.values():
            preset_name = Project().threshold_modes.get_key(thresh)
            index = self.threshold_modes_names.index(preset_name)
            self.combo_thresh.setCurrentIndex(index)
        else:
            index = self.threshold_modes_names.index(_("Custom"))
            self.combo_thresh.setCurrentIndex(index)
            Project().threshold_modes[_("Custom")] = (thresh_min, thresh_max)

    def SetItemsColour(self, colour):
        self.gradient.SetColour(colour)
        self.button_colour.SetColour(colour)

    def AddMask(self, mask):
        if self.combo_mask_name.count() == 0:
            self.setEnabled(True)
        mask_name = mask.name
        self.combo_mask_name.addItem(mask_name)

    def GetMaskSelected(self):
        return self.combo_mask_name.currentIndex()

    def SetThresholdModes(self, thresh_modes_names, default_thresh):
        self.combo_thresh.blockSignals(True)
        self.combo_thresh.clear()
        self.combo_thresh.addItems(thresh_modes_names)
        self.combo_thresh.blockSignals(False)
        self.threshold_modes_names = thresh_modes_names
        proj = Project()
        if isinstance(default_thresh, int):
            self.combo_thresh.setCurrentIndex(default_thresh)
            (thresh_min, thresh_max) = self.threshold_modes[thresh_modes_names[default_thresh]]
        elif default_thresh in proj.threshold_modes.keys():
            index = self.threshold_modes_names.index(default_thresh)
            self.combo_thresh.setCurrentIndex(index)
            thresh_min, thresh_max = self.threshold_modes[default_thresh]

        elif default_thresh in proj.threshold_modes.values():
            preset_name = proj.threshold_modes.get_key(default_thresh)
            index = self.threshold_modes_names.index(preset_name)
            self.combo_thresh.setCurrentIndex(index)
            thresh_min, thresh_max = default_thresh
        else:
            index = self.threshold_modes_names.index(_("Custom"))
            self.combo_thresh.setCurrentIndex(index)
            thresh_min, thresh_max = default_thresh
            proj.threshold_modes[_("Custom")] = (thresh_min, thresh_max)

        self.gradient.SetMinValue(thresh_min)
        self.gradient.SetMaxValue(thresh_max)

    def SetThresholdBounds(self, threshold_range):
        thresh_min = threshold_range[0]
        thresh_max = threshold_range[1]
        self.gradient.SetMinRange(thresh_min)
        self.gradient.SetMaxRange(thresh_max)

    def OnComboName(self, mask_index):
        if mask_index < 0:
            return
        Publisher.sendMessage("Change mask selected", index=mask_index)
        Publisher.sendMessage("Show mask", index=mask_index, value=True)

    def OnComboThresh(self, index):
        if index < 0:
            return
        thresh_name = self.combo_thresh.itemText(index)
        if thresh_name and thresh_name in Project().threshold_modes:
            (thresh_min, thresh_max) = Project().threshold_modes[thresh_name]
            self.gradient.SetMinValue(thresh_min)
            self.gradient.SetMaxValue(thresh_max)
            self.OnSlideChanging(None)
            self.OnSlideChanged(None)

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

    def OnSelectColour(self, colour):
        colour = colour[:3]
        self.gradient.SetColour(colour)
        Publisher.sendMessage("Change mask colour", colour=colour)


class EditionTools(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.unit = "mm"

        ## LINE 1
        text1 = QLabel(_("Choose brush type, size or operation:"))

        ## LINE 2
        CIRCLE_ICON = QIcon(QPixmap(os.path.join(inv_paths.ICON_DIR, "brush_circle.png")))
        SQUARE_ICON = QIcon(QPixmap(os.path.join(inv_paths.ICON_DIR, "brush_square.png")))

        bmp_brush_format = {const.BRUSH_CIRCLE: CIRCLE_ICON, const.BRUSH_SQUARE: SQUARE_ICON}
        selected_icon = bmp_brush_format[const.DEFAULT_BRUSH_FORMAT]

        btn_brush_format = QPushButton("")
        btn_brush_format.setIcon(selected_icon)
        btn_brush_format.setFlat(True)

        brush_menu = QMenu(self)
        act_circle = brush_menu.addAction(CIRCLE_ICON, _("Circle"))
        act_square = brush_menu.addAction(SQUARE_ICON, _("Square"))
        act_circle.setData(const.BRUSH_CIRCLE)
        act_square.setData(const.BRUSH_SQUARE)
        btn_brush_format.setMenu(brush_menu)
        self.btn_brush_format = btn_brush_format
        self._brush_icons = bmp_brush_format

        spin_brush_size = InvSpinCtrl(
            self, -1, value=const.BRUSH_SIZE, min_value=1, max_value=1000, spin_button=False
        )
        spin_brush_size.CalcSizeFromTextSize("MMMM")
        spin_brush_size.valueChanged.connect(self.OnBrushSize)
        self.spin = spin_brush_size

        self.txt_unit = QLabel("mm")
        self.txt_unit.setContextMenuPolicy(Qt.CustomContextMenu)
        self.txt_unit.customContextMenuRequested.connect(self.OnContextMenu)

        combo_brush_op = QComboBox(self)
        combo_brush_op.setEditable(False)
        combo_brush_op.addItems(const.BRUSH_OP_NAME)
        combo_brush_op.setCurrentIndex(const.DEFAULT_BRUSH_OP)
        self.combo_brush_op = combo_brush_op

        line2 = QHBoxLayout()
        line2.addWidget(btn_brush_format, 0)
        line2.addSpacing(5)
        line2.addWidget(spin_brush_size, 0)
        line2.addWidget(self.txt_unit, 0)
        line2.addWidget(combo_brush_op, 1)

        ## LINE 3
        text_thresh = QLabel(_("Brush threshold range:"))

        ## LINE 4
        gradient_thresh = grad.GradientCtrl(self, -1, 0, 5000, 0, 5000, (0, 0, 255, 100))
        self.gradient_thresh = gradient_thresh
        self.bind_evt_gradient = True

        ## LINE 5
        m3ediv = QFrame(self)
        m3ediv.setFrameShape(QFrame.HLine)
        m3ediv.setFrameShadow(QFrame.Sunken)

        cbox_mask_edit_3d = QCheckBox("Edit in 3D", self)
        btn_clear_3d_poly = QPushButton(_("Clear Polygons"), self)

        line5 = QHBoxLayout()
        line5.addWidget(cbox_mask_edit_3d, 0)
        line5.addStretch(1)
        line5.addWidget(btn_clear_3d_poly, 0)

        ## LINE 6
        txt_edit_op = QLabel(_("Operation:"))
        combo_mask_edit_3d_op = QComboBox(self)
        combo_mask_edit_3d_op.setEditable(False)
        combo_mask_edit_3d_op.addItems(const.MASK_3D_EDIT_OP_NAME)
        combo_mask_edit_3d_op.setCurrentIndex(const.MASK_3D_EDIT_INCLUDE)

        txt_depth_value = QLabel(_("Depth:"))
        spin_mask_edit_3d_depth = InvFloatSpinCtrl(
            self, value=1.0, min_value=0.0, max_value=1.0, increment=0.05
        )

        line6 = QHBoxLayout()
        line6.addWidget(txt_edit_op, 0)
        line6.addSpacing(1)
        line6.addWidget(combo_mask_edit_3d_op, 0)
        line6.addSpacing(15)
        line6.addWidget(txt_depth_value, 0)
        line6.addSpacing(1)
        line6.addWidget(spin_mask_edit_3d_depth, 0)

        self.cbox_mask_edit_3d = cbox_mask_edit_3d
        self.combo_mask_edit_3d_op = combo_mask_edit_3d_op
        self.spin_mask_edit_3d_depth = spin_mask_edit_3d_depth
        self.btn_clear_3d_poly = btn_clear_3d_poly

        layout = QVBoxLayout(self)
        layout.addSpacing(7)
        layout.addWidget(text1)
        layout.addSpacing(2)
        layout.addLayout(line2)
        layout.addSpacing(5)
        layout.addWidget(text_thresh)
        layout.addSpacing(5)
        layout.addWidget(gradient_thresh)
        layout.addSpacing(5)
        layout.addWidget(m3ediv)
        layout.addSpacing(3)
        layout.addLayout(line5)
        layout.addSpacing(3)
        layout.addLayout(line6)
        layout.addSpacing(7)

        self.__bind_events()
        self.__bind_events_wx()

    def __bind_events_wx(self):
        self.btn_brush_format.menu().triggered.connect(self.OnMenu)
        self.gradient_thresh.threshold_changed.connect(self.OnGradientChanged)
        self.combo_brush_op.currentIndexChanged.connect(self.OnComboBrushOp)
        self.cbox_mask_edit_3d.stateChanged.connect(self.OnCheckboxMaskEdit3D)
        self.combo_mask_edit_3d_op.currentIndexChanged.connect(self.OnComboMaskEdit3DMode)
        self.spin_mask_edit_3d_depth.valueChanged.connect(self.OnSpinDepthMaskEdit3D)
        self.btn_clear_3d_poly.clicked.connect(self.OnClearPolyMaskEdit3D)

    def __bind_events(self):
        Publisher.subscribe(self.SetThresholdBounds, "Update threshold limits")
        Publisher.subscribe(self.ChangeMaskColour, "Change mask colour")
        Publisher.subscribe(self.SetGradientColour, "Add mask")
        Publisher.subscribe(self._set_brush_size, "Set edition brush size")
        Publisher.subscribe(self._set_threshold_range_gui, "Set edition threshold gui")
        Publisher.subscribe(self.ChangeMaskColour, "Set GUI items colour")
        Publisher.subscribe(self.OnAskMaskEdit3DMode, "M3E ask for edit mode")
        Publisher.subscribe(self.OnAskDepthMaskEdit3D, "M3E ask for depth value")

    def ChangeMaskColour(self, colour):
        self.gradient_thresh.SetColour(colour)

    def SetGradientColour(self, mask):
        qt_colour = [c * 255 for c in mask.colour]
        self.gradient_thresh.SetColour(qt_colour)

    def SetThresholdValues(self, threshold_range):
        thresh_min, thresh_max = threshold_range
        self.bind_evt_gradient = False
        self.gradient_thresh.SetMinValue(thresh_min)
        self.gradient_thresh.SetMaxValue(thresh_max)
        self.bind_evt_gradient = True

    def SetThresholdBounds(self, threshold_range):
        thresh_min = threshold_range[0]
        thresh_max = threshold_range[1]
        self.gradient_thresh.SetMinRange(thresh_min)
        self.gradient_thresh.SetMaxRange(thresh_max)
        self.gradient_thresh.SetMinValue(thresh_min)
        self.gradient_thresh.SetMaxValue(thresh_max)

    def OnGradientChanged(self, evt):
        thresh_min = self.gradient_thresh.GetMinValue()
        thresh_max = self.gradient_thresh.GetMaxValue()
        if self.bind_evt_gradient:
            Publisher.sendMessage(
                "Set edition threshold values", threshold_range=(thresh_min, thresh_max)
            )

    def OnMenu(self, action):
        brush_format = action.data()
        self.btn_brush_format.setIcon(self._brush_icons[brush_format])
        Publisher.sendMessage("Set brush format", cursor_format=brush_format)

    def OnBrushSize(self):
        Publisher.sendMessage("Set edition brush size", size=self.spin.GetValue())

    def OnContextMenu(self, pos):
        menu = QMenu(self)
        mm_action = menu.addAction("mm")
        mm_action.setCheckable(True)
        um_action = menu.addAction("µm")
        um_action.setCheckable(True)
        px_action = menu.addAction("px")
        px_action.setCheckable(True)

        if self.unit == "mm":
            mm_action.setChecked(True)
        elif self.unit == "µm":
            um_action.setChecked(True)
        else:
            px_action.setChecked(True)

        action = menu.exec_(self.txt_unit.mapToGlobal(pos))
        if action:
            self.txt_unit.setText(action.text())
            self.unit = action.text()
            Publisher.sendMessage("Set edition brush unit", unit=self.unit)

    def _set_threshold_range_gui(self, threshold_range):
        self.SetThresholdValues(threshold_range)

    def _set_brush_size(self, size):
        self.spin.SetValue(size)

    def OnComboBrushOp(self, brush_op_id):
        if brush_op_id < 0:
            return
        Publisher.sendMessage("Set edition operation", operation=brush_op_id)
        if brush_op_id == const.BRUSH_THRESH:
            self.gradient_thresh.setEnabled(True)
        else:
            self.gradient_thresh.setEnabled(False)

    def OnCheckboxMaskEdit3D(self, state):
        style_id = const.STATE_MASK_3D_EDIT
        spin_val = self.spin_mask_edit_3d_depth.GetValue()

        if self.cbox_mask_edit_3d.isChecked():
            Publisher.sendMessage("Enable style", style=style_id)
            Publisher.sendMessage("M3E set depth value", value=spin_val)
            self.btn_clear_3d_poly.setEnabled(True)
        else:
            Publisher.sendMessage("Disable style", style=style_id)
            self.btn_clear_3d_poly.setEnabled(False)

    def OnComboMaskEdit3DMode(self, op_id):
        if op_id < 0:
            return
        Publisher.sendMessage("M3E set edit mode", mode=op_id)

    def OnAskMaskEdit3DMode(self):
        op_id = self.combo_mask_edit_3d_op.currentIndex()
        Publisher.sendMessage("M3E set edit mode", mode=op_id)

    def OnSpinDepthMaskEdit3D(self):
        spin_val = self.spin_mask_edit_3d_depth.GetValue()
        Publisher.sendMessage("M3E set depth value", value=spin_val)

    def OnAskDepthMaskEdit3D(self):
        spin_val = self.spin_mask_edit_3d_depth.GetValue()
        Publisher.sendMessage("M3E set depth value", value=spin_val)

    def OnClearPolyMaskEdit3D(self):
        if self.cbox_mask_edit_3d.isChecked():
            Publisher.sendMessage("M3E clear polygons")


class WatershedTool(EditionTools):
    def __init__(self, parent):
        QWidget.__init__(self, parent)

        ## LINE 1
        text1 = QLabel(_("Choose brush type, size or operation:"))

        self.unit = "mm"

        ## LINE 2
        CIRCLE_ICON = QIcon(QPixmap(os.path.join(inv_paths.ICON_DIR, "brush_circle.png")))
        SQUARE_ICON = QIcon(QPixmap(os.path.join(inv_paths.ICON_DIR, "brush_square.png")))

        bmp_brush_format = {const.BRUSH_CIRCLE: CIRCLE_ICON, const.BRUSH_SQUARE: SQUARE_ICON}
        selected_icon = bmp_brush_format[const.DEFAULT_BRUSH_FORMAT]

        btn_brush_format = QPushButton("")
        btn_brush_format.setIcon(selected_icon)
        btn_brush_format.setFlat(True)

        brush_menu = QMenu(self)
        act_circle = brush_menu.addAction(CIRCLE_ICON, _("Circle"))
        act_square = brush_menu.addAction(SQUARE_ICON, _("Square"))
        act_circle.setData(const.BRUSH_CIRCLE)
        act_square.setData(const.BRUSH_SQUARE)
        btn_brush_format.setMenu(brush_menu)
        self.btn_brush_format = btn_brush_format
        self._brush_icons = bmp_brush_format

        spin_brush_size = InvSpinCtrl(
            self, -1, value=const.BRUSH_SIZE, min_value=1, max_value=1000, spin_button=False
        )
        spin_brush_size.CalcSizeFromTextSize("MMMM")
        spin_brush_size.valueChanged.connect(self.OnBrushSize)
        self.spin = spin_brush_size

        self.txt_unit = QLabel("mm")
        self.txt_unit.setContextMenuPolicy(Qt.CustomContextMenu)
        self.txt_unit.customContextMenuRequested.connect(self.OnContextMenu)

        combo_brush_op = QComboBox(self)
        combo_brush_op.setEditable(False)
        combo_brush_op.addItems([_("Foreground"), _("Background"), _("Erase")])
        combo_brush_op.setCurrentIndex(0)
        self.combo_brush_op = combo_brush_op

        line2 = QHBoxLayout()
        line2.addWidget(btn_brush_format, 0)
        line2.addSpacing(5)
        line2.addWidget(spin_brush_size, 0)
        line2.addWidget(self.txt_unit, 0)
        line2.addWidget(combo_brush_op, 1)

        # LINE 5
        check_box = QCheckBox(_("Overwrite mask"), self)
        ww_wl_cbox = QCheckBox(_("Use WW&WL"), self)
        ww_wl_cbox.setChecked(True)
        self.check_box = check_box
        self.ww_wl_cbox = ww_wl_cbox

        # Line 6
        bmp = QPixmap(os.path.join(inv_paths.ICON_DIR, "configuration.png"))
        self.btn_wconfig = QPushButton("")
        self.btn_wconfig.setIcon(QIcon(bmp))
        self.btn_wconfig.setFixedSize(bmp.width() + 10, bmp.height() + 10)
        self.btn_exp_watershed = QPushButton(_("Expand watershed to 3D"))

        sizer_btns = QHBoxLayout()
        sizer_btns.addWidget(self.btn_wconfig, 0, Qt.AlignLeft)
        sizer_btns.addWidget(self.btn_exp_watershed, 1)

        layout = QVBoxLayout(self)
        layout.addSpacing(7)
        layout.addWidget(text1)
        layout.addSpacing(2)
        layout.addLayout(line2)
        layout.addSpacing(5)
        layout.addWidget(check_box)
        layout.addSpacing(2)
        layout.addWidget(ww_wl_cbox)
        layout.addSpacing(5)
        layout.addLayout(sizer_btns)
        layout.addSpacing(7)

        self.__bind_events_wx()
        self.__bind_pubsub_evt()

    def __bind_events_wx(self):
        self.btn_brush_format.menu().triggered.connect(self.OnMenu)
        self.combo_brush_op.currentIndexChanged.connect(self.OnComboBrushOp)
        self.check_box.stateChanged.connect(self.OnCheckOverwriteMask)
        self.ww_wl_cbox.stateChanged.connect(self.OnCheckWWWL)
        self.btn_exp_watershed.clicked.connect(self.OnExpandWatershed)
        self.btn_wconfig.clicked.connect(self.OnConfig)

    def __bind_pubsub_evt(self):
        Publisher.subscribe(self._set_brush_size, "Set watershed brush size")

    def ChangeMaskColour(self, colour):
        self.gradient_thresh.SetColour(colour)

    def SetGradientColour(self, mask):
        vtk_colour = mask.colour
        qt_colour = [c * 255 for c in vtk_colour]
        self.gradient_thresh.SetColour(qt_colour)

    def SetThresholdValues(self, threshold_range):
        thresh_min, thresh_max = threshold_range
        self.bind_evt_gradient = False
        self.gradient_thresh.SetMinValue(thresh_min)
        self.gradient_thresh.SetMaxValue(thresh_max)
        self.bind_evt_gradient = True

    def SetThresholdBounds(self, threshold_range):
        thresh_min = threshold_range[0]
        thresh_max = threshold_range[1]
        self.gradient_thresh.SetMinRange(thresh_min)
        self.gradient_thresh.SetMaxRange(thresh_max)
        self.gradient_thresh.SetMinValue(thresh_min)
        self.gradient_thresh.SetMaxValue(thresh_max)

    def OnMenu(self, action):
        brush_format = action.data()
        self.btn_brush_format.setIcon(self._brush_icons[brush_format])
        Publisher.sendMessage("Set watershed brush format", brush_format=brush_format)

    def OnBrushSize(self):
        Publisher.sendMessage("Set watershed brush size", size=self.spin.GetValue())

    def OnContextMenu(self, pos):
        menu = QMenu(self)
        mm_action = menu.addAction("mm")
        mm_action.setCheckable(True)
        um_action = menu.addAction("µm")
        um_action.setCheckable(True)
        px_action = menu.addAction("px")
        px_action.setCheckable(True)

        if self.unit == "mm":
            mm_action.setChecked(True)
        elif self.unit == "µm":
            um_action.setChecked(True)
        else:
            px_action.setChecked(True)

        action = menu.exec_(self.txt_unit.mapToGlobal(pos))
        if action:
            self.txt_unit.setText(action.text())
            self.unit = action.text()
            Publisher.sendMessage("Set watershed brush unit", unit=self.unit)

    def _set_brush_size(self, size):
        self.spin.SetValue(size)

    def OnComboBrushOp(self, index):
        if index < 0:
            return
        brush_op = self.combo_brush_op.currentText()
        Publisher.sendMessage("Set watershed operation", operation=brush_op)

    def OnCheckOverwriteMask(self, state):
        value = self.check_box.isChecked()
        Publisher.sendMessage("Set overwrite mask", flag=value)

    def OnCheckWWWL(self, state):
        value = self.ww_wl_cbox.isChecked()
        Publisher.sendMessage("Set use ww wl", use_ww_wl=value)

    def OnConfig(self):
        from invesalius.data.styles import WatershedConfig

        config = WatershedConfig()
        dlg.WatershedOptionsDialog(config).show()

    def OnExpandWatershed(self):
        Publisher.sendMessage("Expand watershed to 3D AXIAL")
