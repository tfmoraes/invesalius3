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

import time
from functools import partial

import numpy as np

try:
    import Trekker  # noqa: F401

    has_trekker = True
except ImportError:
    has_trekker = False

try:
    from invesalius.navigation.mtms import mTMS

    mTMS()
    has_mTMS = True
except Exception:
    has_mTMS = False

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.data.brainmesh_handler as brain
import invesalius.gui.dialogs as dlg
import invesalius.session as ses
from invesalius.i18n import tr as _
from invesalius.navigation.navigation import Navigation
from invesalius.net.neuronavigation_api import NeuronavigationApi
from invesalius.net.pedal_connection import PedalConnector
from invesalius.pubsub import pub as Publisher


class TaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        neuronavigation_api = NeuronavigationApi()
        pedal_connector = PedalConnector(neuronavigation_api, self)
        navigation = Navigation(
            pedal_connector=pedal_connector,
            neuronavigation_api=neuronavigation_api,
        )

        inner_panel = InnerTaskPanel(self, navigation)

        sizer = QHBoxLayout(self)
        sizer.setContentsMargins(7, 0, 7, 7)
        sizer.addWidget(inner_panel)


class InnerTaskPanel(QWidget):
    def __init__(self, parent, navigation):
        super().__init__(parent)
        self.__bind_events()

        self.e_field_loaded = False
        self.e_field_brain = None
        self.e_field_mesh = None
        self.cortex_file = None
        self.meshes_file = None
        self.multilocus_coil = None
        self.coil = None
        self.ci = None
        self.co = None
        self.sleep_nav = const.SLEEP_NAVIGATION
        self.navigation = navigation
        self.session = ses.Session()

        enable_efield = QCheckBox(_("Enable E-field"), self)
        enable_efield.setChecked(False)
        enable_efield.setEnabled(True)
        enable_efield.stateChanged.connect(lambda: self.OnEnableEfield(ctrl=enable_efield))
        self.enable_efield = enable_efield

        efield_for_targeting = QCheckBox(_("E-fields for targeting"), self)
        efield_for_targeting.setChecked(False)
        efield_for_targeting.setEnabled(True)
        efield_for_targeting.stateChanged.connect(
            lambda: self.OnEfieldsForTargeting(ctrl=efield_for_targeting)
        )

        show_area = QCheckBox(_("Show area above threshold"), self)
        show_area.setChecked(False)
        show_area.setEnabled(True)
        show_area.stateChanged.connect(lambda: self.OnEnableShowAreaAboveThreshold(ctrl=show_area))

        efield_tools = QCheckBox(_("Enable Efield targeting tools"), self)
        efield_tools.setChecked(False)
        efield_tools.setEnabled(True)
        efield_tools.stateChanged.connect(
            lambda: self.OnEnableEfieldTargetingTools(ctrl=efield_tools)
        )

        efield_cortex_markers = QCheckBox(_("View cortex Markers"), self)
        efield_cortex_markers.setChecked(True)
        efield_cortex_markers.setEnabled(True)
        efield_cortex_markers.stateChanged.connect(
            lambda: self.OnViewCortexMarkers(ctrl=efield_cortex_markers)
        )

        efield_save_automatically = QCheckBox(_("Save Automatically"), self)
        efield_save_automatically.setChecked(False)
        efield_save_automatically.setEnabled(True)
        efield_save_automatically.stateChanged.connect(
            lambda: self.OnSaveEfieldAutomatically(ctrl=efield_save_automatically)
        )

        btn_act2 = QPushButton(_("Load Config"), self)
        btn_act2.setFixedSize(100, 23)
        btn_act2.setToolTip(_("Load Brain Json config"))
        btn_act2.setEnabled(True)
        btn_act2.clicked.connect(self.OnAddConfig)

        self.btn_save = QPushButton(_("Save Efield"), self)
        self.btn_save.setFixedWidth(80)
        self.btn_save.setToolTip(_("Save Efield"))
        self.btn_save.clicked.connect(self.OnSaveEfield)
        self.btn_save.setEnabled(False)

        self.btn_all_save = QPushButton(_("Save All Efield"), self)
        self.btn_all_save.setFixedWidth(80)
        self.btn_all_save.setToolTip(_("Save All Efield"))
        self.btn_all_save.clicked.connect(self.OnSaveAllDataEfield)
        self.btn_all_save.setEnabled(False)

        text_sleep = QLabel(_("Sleep (s):"), self)
        spin_sleep = QDoubleSpinBox(self)
        spin_sleep.setFixedSize(50, 23)
        spin_sleep.setEnabled(True)
        spin_sleep.setRange(0.05, 10.0)
        spin_sleep.setSingleStep(0.01)
        spin_sleep.setValue(self.sleep_nav)
        spin_sleep.valueChanged.connect(lambda: self.OnSelectSleep(ctrl=spin_sleep))

        text_threshold = QLabel(_("Threshold:"), self)
        spin_threshold = QDoubleSpinBox(self)
        spin_threshold.setFixedSize(50, 23)
        spin_threshold.setEnabled(True)
        spin_threshold.setRange(0.1, 1)
        spin_threshold.setSingleStep(0.01)
        spin_threshold.setValue(const.EFIELD_MAX_RANGE_SCALE)
        spin_threshold.valueChanged.connect(lambda: self.OnSelectThreshold(ctrl=spin_threshold))

        text_ROI_size = QLabel(_("ROI size:"), self)
        spin_ROI_size = QDoubleSpinBox(self)
        spin_ROI_size.setFixedSize(50, 23)
        spin_ROI_size.setEnabled(True)
        spin_ROI_size.setSingleStep(0.01)
        spin_ROI_size.setValue(const.EFIELD_ROI_SIZE)
        spin_ROI_size.valueChanged.connect(lambda: self.OnSelectROISize(ctrl=spin_ROI_size))

        combo_surface_name_title = QLabel(_("Change coil:"), self)
        self.combo_change_coil = QComboBox(self)
        self.combo_change_coil.setFixedSize(100, 23)
        self.combo_change_coil.activated.connect(self.OnComboCoil)
        self.combo_change_coil.addItem("Select coil:")
        self.combo_change_coil.setEnabled(False)
        self.combo_change_coil.showPopup = self._wrap_show_popup(self.combo_change_coil)

        value = str(0)

        self.input_dt = QLineEdit(str(1), self)
        self.input_dt.setFixedWidth(60)
        self.input_dt.setAlignment(Qt.AlignCenter)
        self.input_dt.setToolTip(_("dt(\u03bc s)"))

        self.input_coil1 = QLineEdit(value, self)
        self.input_coil1.setFixedWidth(60)
        self.input_coil1.setAlignment(Qt.AlignCenter)
        self.input_coil1.setToolTip(_("dI"))

        self.input_coil2 = QLineEdit(value, self)
        self.input_coil2.setFixedWidth(60)
        self.input_coil2.setAlignment(Qt.AlignCenter)
        self.input_coil2.setToolTip(_("dI"))

        self.input_coil3 = QLineEdit(value, self)
        self.input_coil3.setFixedWidth(60)
        self.input_coil3.setAlignment(Qt.AlignCenter)
        self.input_coil3.setToolTip(_("dI"))

        self.input_coil4 = QLineEdit(value, self)
        self.input_coil4.setFixedWidth(60)
        self.input_coil4.setAlignment(Qt.AlignCenter)
        self.input_coil4.setToolTip(_("dI"))

        self.input_coil5 = QLineEdit(value, self)
        self.input_coil5.setFixedWidth(60)
        self.input_coil5.setAlignment(Qt.AlignCenter)
        self.input_coil5.setToolTip(_("dI"))

        text_input_coord = QLabel(_("mtms coords:"), self)
        self.input_coord = QLineEdit(value, self)
        self.input_coord.setFixedWidth(60)
        self.input_coord.setAlignment(Qt.AlignCenter)
        self.input_coord.setToolTip(_("mtms coords"))

        btn_enter_mtms_coord = QPushButton(_("Enter mtms coord"), self)
        btn_enter_mtms_coord.setFixedWidth(80)
        btn_enter_mtms_coord.setToolTip(_("Enter mtms coord"))
        btn_enter_mtms_coord.clicked.connect(self.OnEnterMtmsCoords)
        btn_enter_mtms_coord.setEnabled(True)

        btn_enter = QPushButton(_("Enter"), self)
        btn_enter.setFixedWidth(80)
        btn_enter.setToolTip(_("Enter Values"))
        btn_enter.clicked.connect(self.OnEnterdIPerdt)
        btn_enter.setEnabled(True)

        btn_reset = QPushButton(_("Reset"), self)
        btn_reset.setFixedWidth(80)
        btn_reset.setToolTip(_("Reset Values"))
        btn_reset.clicked.connect(self.OnReset)
        btn_reset.setEnabled(True)

        line_checkboxes = QHBoxLayout()
        line_checkboxes.addWidget(enable_efield)
        line_checkboxes.addWidget(show_area)
        line_checkboxes.addWidget(efield_tools)

        line_change_coil_input_coord_text = QHBoxLayout()
        line_change_coil_input_coord_text.addWidget(combo_surface_name_title)
        line_change_coil_input_coord_text.addWidget(text_input_coord)

        line_change_coil_input_coord = QHBoxLayout()
        line_change_coil_input_coord.addWidget(self.combo_change_coil, 1)
        line_change_coil_input_coord.addWidget(self.input_coord, 1)
        line_change_coil_input_coord.addWidget(btn_enter_mtms_coord, 1)

        line_sleep = QHBoxLayout()
        line_sleep.addWidget(text_sleep, 1)
        line_sleep.addWidget(spin_sleep)
        line_sleep.addWidget(text_threshold, 1)
        line_sleep.addWidget(spin_threshold)
        line_sleep.addWidget(text_ROI_size, 1)
        line_sleep.addWidget(spin_ROI_size)

        line_btns = QHBoxLayout()
        line_btns.addWidget(btn_act2, 1)

        line_btns_save = QHBoxLayout()
        line_btns_save.addWidget(self.input_dt, 1)
        line_btns_save.addWidget(self.btn_save, 1)
        line_btns_save.addWidget(self.btn_all_save, 1)

        text_mtms = QLabel(_("dI"), self)
        line_mtms = QHBoxLayout()
        line_mtms.addWidget(self.input_coil1)
        line_mtms.addWidget(self.input_coil2)
        line_mtms.addWidget(self.input_coil3)
        line_mtms.addWidget(self.input_coil4)
        line_mtms.addWidget(self.input_coil5)

        line_mtms_buttoms = QHBoxLayout()
        line_mtms_buttoms.addWidget(btn_enter)
        line_mtms_buttoms.addWidget(btn_reset)

        line_cortex_markers = QHBoxLayout()
        line_cortex_markers.addWidget(efield_cortex_markers)
        line_cortex_markers.addWidget(efield_save_automatically)
        line_cortex_markers.addWidget(efield_for_targeting)

        main_sizer = QVBoxLayout(self)
        main_sizer.addLayout(line_btns)
        main_sizer.addLayout(line_checkboxes)
        main_sizer.addLayout(line_change_coil_input_coord_text)
        main_sizer.addLayout(line_change_coil_input_coord)
        main_sizer.addLayout(line_sleep)
        main_sizer.addLayout(line_btns_save)
        main_sizer.addWidget(text_mtms)
        main_sizer.addLayout(line_mtms)
        main_sizer.addLayout(line_mtms_buttoms)
        main_sizer.addLayout(line_cortex_markers)

    def _wrap_show_popup(self, combo):
        original = combo.__class__.showPopup

        def wrapper():
            self.OnComboCoilNameClic()
            original(combo)

        return wrapper

    def __bind_events(self):
        Publisher.subscribe(self.UpdateNavigationStatus, "Navigation status")
        Publisher.subscribe(self.OnGetEfieldActor, "Get Efield actor from json")
        Publisher.subscribe(self.OnGetEfieldPaths, "Get Efield paths")
        Publisher.subscribe(self.OnGetMultilocusCoils, "Get multilocus paths from json")
        Publisher.subscribe(self.SendNeuronavigationApi, "Send Neuronavigation Api")
        Publisher.subscribe(self.GetEfieldDataStatus, "Get status of Efield saved data")
        Publisher.subscribe(self.GetIds, "Get dI for mtms")

    def OnAddConfig(self):
        filename = dlg.LoadConfigEfield()
        if filename:
            convert_to_inv = dlg.ImportMeshCoordSystem()
            Publisher.sendMessage("Update status in GUI", value=50, label="Loading E-field...")
            Publisher.sendMessage("Update convert_to_inv flag", convert_to_inv=convert_to_inv)
            Publisher.sendMessage(
                "Read json config file for efield", filename=filename, convert_to_inv=convert_to_inv
            )
            self.e_field_brain = brain.E_field_brain(self.e_field_mesh)
            self.Init_efield()

    def Init_efield(self):
        self.navigation.neuronavigation_api.initialize_efield(
            cortex_model_path=self.cortex_file,
            mesh_models_paths=self.meshes_file,
            coil_model_path=self.coil,
            coil_set=False,
            conductivities_inside=self.ci,
            conductivities_outside=self.co,
            dI_per_dt=self.dIperdt_list,
        )
        Publisher.sendMessage("Update status in GUI", value=0, label="Ready")
        self.Send_dI_per_dt_to_report(self.dIperdt_list, self.ci, self.co)
        self.Send_meshes_coil_paths_to_report()

    def OnEnableEfield(self, ctrl):
        efield_enabled = ctrl.isChecked()
        self.plot_efield_vectors = ctrl.isChecked()
        self.navigation.plot_efield_vectors = self.plot_efield_vectors
        if efield_enabled:
            if self.session.GetConfig("debug_efield"):
                debug_efield_enorm = dlg.ShowLoadCSVDebugEfield()
                if isinstance(debug_efield_enorm, np.ndarray):
                    self.navigation.debug_efield_enorm = debug_efield_enorm
                else:
                    dlg.Efield_debug_Enorm_warning()
                    self.enable_efield.setChecked(False)
                    self.e_field_loaded = False
                    self.navigation.e_field_loaded = self.e_field_loaded
                    return
            else:
                if not self.navigation.neuronavigation_api.connection:
                    dlg.Efield_connection_warning()
                    self.enable_efield.setEnabled(False)
                    self.e_field_loaded = False
                    return
            Publisher.sendMessage("Initialize E-field brain", e_field_brain=self.e_field_brain)

            Publisher.sendMessage("Initialize color array")
            self.e_field_loaded = True
            self.combo_change_coil.setEnabled(True)
            self.btn_all_save.setEnabled(True)

        else:
            Publisher.sendMessage("Recolor again")
            self.e_field_loaded = False
        self.navigation.e_field_loaded = self.e_field_loaded

    def OnEnablePlotVectors(self, ctrl):
        self.plot_efield_vectors = ctrl.isChecked()
        self.navigation.plot_efield_vectors = self.plot_efield_vectors

    def OnEnableShowAreaAboveThreshold(self, ctrl):
        enable = ctrl.isChecked()
        Publisher.sendMessage("Show area above threshold", enable=enable)

    def OnEnableEfieldTargetingTools(self, ctrl):
        enable = ctrl.isChecked()
        Publisher.sendMessage("Enable Efield tools", enable=enable)

    def OnViewCortexMarkers(self, ctrl):
        enable = ctrl.isChecked()
        Publisher.sendMessage("Display efield markers at cortex", display_flag=enable)

    def OnComboNameClic(self):
        import invesalius.project as prj

        proj = prj.Project()
        self.combo_change_coil.clear()
        for n in range(len(proj.surface_dict)):
            self.combo_change_coil.addItem(str(proj.surface_dict[n].name))

    def OnComboCoilNameClic(self):
        self.combo_change_coil.clear()
        if self.multilocus_coil is not None:
            for elements in range(len(self.multilocus_coil)):
                coil_name = self.multilocus_coil[elements].split("/")[-1].split(".bin")[0]
                self.combo_change_coil.addItem(coil_name)

    def OnComboCoil(self, index):
        coil_index = index
        if coil_index == 6:
            coil_set = True
        else:
            coil_set = False
        self.OnChangeCoil(self.multilocus_coil[coil_index], coil_set)

    def OnChangeCoil(self, coil_model_path, coil_set):
        self.navigation.neuronavigation_api.efield_coil(
            coil_model_path=coil_model_path, coil_set=coil_set
        )
        self.coil = coil_model_path
        self.Send_meshes_coil_paths_to_report()

    def UpdateNavigationStatus(self, nav_status, vis_status):
        if nav_status:
            self.enable_efield.setEnabled(False)
            self.btn_save.setEnabled(True)
        else:
            self.enable_efield.setEnabled(True)
            self.btn_save.setEnabled(False)

    def OnSelectSleep(self, ctrl):
        self.sleep_nav = ctrl.value()
        Publisher.sendMessage("Update sleep", data=self.sleep_nav)

    def OnSelectThreshold(self, ctrl):
        threshold = ctrl.value()
        Publisher.sendMessage("Update Efield Threshold", data=threshold)

    def OnSelectROISize(self, ctrl):
        ROI_size = ctrl.value()
        Publisher.sendMessage("Update Efield ROI size", data=ROI_size)

    def OnGetEfieldActor(self, efield_actor, surface_index_cortex):
        self.e_field_mesh = efield_actor
        self.surface_index = surface_index_cortex
        Publisher.sendMessage("Get Actor", surface_index=self.surface_index)

    def OnGetEfieldPaths(self, path_meshes, cortex_file, meshes_file, coil, ci, co, dIperdt_list):
        self.path_meshes = path_meshes
        self.cortex_file = cortex_file
        self.meshes_file = meshes_file
        self.ci = ci
        self.co = co
        self.coil = coil
        self.dIperdt_list = dIperdt_list

    def OnGetMultilocusCoils(self, multilocus_coil_list):
        self.multilocus_coil = multilocus_coil_list

    def OnSaveEfieldAutomatically(self, ctrl):
        enable = ctrl.isChecked()
        Publisher.sendMessage(
            "Save automatically efield data",
            enable=enable,
            path_meshes=self.path_meshes,
            plot_efield_vectors=self.navigation.plot_efield_vectors,
        )

    def OnSaveEfield(self):
        import invesalius.project as prj

        proj = prj.Project()
        timestamp = time.localtime(time.time())
        stamp_date = f"{timestamp.tm_year:0>4d}{timestamp.tm_mon:0>2d}{timestamp.tm_mday:0>2d}"
        stamp_time = f"{timestamp.tm_hour:0>2d}{timestamp.tm_min:0>2d}{timestamp.tm_sec:0>2d}"
        sep = "-"
        if self.path_meshes is None:
            import os

            current_folder_path = os.getcwd()
        else:
            current_folder_path = self.path_meshes
        parts = [current_folder_path, "/", stamp_date, stamp_time, proj.name, "Efield"]
        default_filename = sep.join(parts) + ".csv"

        filename = dlg.ShowLoadSaveDialog(
            message=_("Save markers as..."),
            wildcard="(*.csv)|*.csv",
            default_filename=default_filename,
            save_ext="csv",
        )

        if not filename:
            return
        plot_efield_vectors = self.navigation.plot_efield_vectors
        Publisher.sendMessage(
            "Save Efield data",
            filename=filename,
            plot_efield_vectors=plot_efield_vectors,
            marker_id=None,
        )

    def OnSaveAllDataEfield(self):
        Publisher.sendMessage("Check efield data")
        if self.efield_data_saved:
            import invesalius.project as prj

            proj = prj.Project()
            timestamp = time.localtime(time.time())
            stamp_date = f"{timestamp.tm_year:0>4d}{timestamp.tm_mon:0>2d}{timestamp.tm_mday:0>2d}"
            stamp_time = f"{timestamp.tm_hour:0>2d}{timestamp.tm_min:0>2d}{timestamp.tm_sec:0>2d}"
            sep = "-"
            if self.path_meshes is None:
                import os

                current_folder_path = os.getcwd()
            else:
                current_folder_path = self.path_meshes
            parts = [current_folder_path, "/", stamp_date, stamp_time, proj.name, "Efield"]
            default_filename = sep.join(parts) + ".csv"

            filename = dlg.ShowLoadSaveDialog(
                message=_("Save markers as..."),
                wildcard="(*.csv)|*.csv",
                default_filename=default_filename,
                save_ext="csv",
            )

            if not filename:
                return

            Publisher.sendMessage("Save all Efield data", filename=filename)
        else:
            dlg.Efield_no_data_to_save_warning()

    def SendNeuronavigationApi(self):
        Publisher.sendMessage(
            "Get Neuronavigation Api", neuronavigation_api=self.navigation.neuronavigation_api
        )

    def GetEfieldDataStatus(self, efield_data_loaded, indexes_saved_list):
        self.efield_data_saved = efield_data_loaded

    def OnEnterdIPerdt(self):
        input_dt = 1 / (float(self.input_dt.text()) * 1e-6)
        self.input_coils = [
            float(self.input_coil1.text()),
            float(self.input_coil2.text()),
            float(self.input_coil3.text()),
            float(self.input_coil4.text()),
            float(self.input_coil5.text()),
        ]
        self.input_coils = np.array(self.input_coils) * input_dt
        self.input_coils = self.input_coils.tolist()
        self.navigation.neuronavigation_api.set_dIperdt(
            dIperdt=self.input_coils,
        )
        self.Send_dI_per_dt_to_report(self.input_coils, self.ci, self.co)

    def OnEnterMtmsCoords(self):
        input_coord_str = self.input_coord.text()
        input_coord = [int(i) for i in input_coord_str.split(",") if i]
        Publisher.sendMessage("Send mtms coords", mtms_coord=input_coord)

    def SenddI(self, dIs):
        self.OnChangeCoil(self.multilocus_coil[6], True)
        input_dt = 1 / (float(self.input_dt.text()) * 1e-6)
        self.input_coils = dIs
        self.input_coils = np.array(self.input_coils) * input_dt
        self.input_coil1.setText(str(dIs[0]))
        self.input_coil2.setText(str(dIs[1]))
        self.input_coil3.setText(str(dIs[2]))
        self.input_coil4.setText(str(dIs[3]))
        self.input_coil5.setText(str(dIs[4]))

        self.navigation.neuronavigation_api.set_dIperdt(
            dIperdt=self.input_coils,
        )
        self.Send_dI_per_dt_to_report(self.input_coils, self.ci, self.co)

    def OnEfieldsForTargeting(self, ctrl):
        if ctrl.isChecked():
            self.navigation.neuronavigation_api.set_dIperdt(
                dIperdt=[1, 1, 1, 1, 1],
            )
            self.Send_dI_per_dt_to_report([1, 1, 1, 1, 1], self.ci, self.co)

    def GetIds(self, dIs):
        self.SenddI(dIs)

    def OnReset(self):
        Publisher.sendMessage("Get targets Ids for mtms", target1_origin=[0, 0], target2=[0, 0])

    def Send_dI_per_dt_to_report(self, diperdt, ci, co):
        Publisher.sendMessage(
            "Get diperdt used in efield calculation", diperdt=diperdt, ci=self.ci, co=self.co
        )

    def Send_meshes_coil_paths_to_report(self):
        Publisher.sendMessage(
            "Get path meshes",
            path_meshes=self.path_meshes,
            cortex_file=self.cortex_file,
            meshes_file=self.meshes_file,
            coilmodel=self.coil,
        )
