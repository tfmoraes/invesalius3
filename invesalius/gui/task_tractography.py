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

import nibabel as nb
import numpy as np

try:
    import Trekker

    has_trekker = True
except ImportError:
    has_trekker = False

try:
    from invesalius.navigation.mtms import mTMS

    mTMS()
    has_mTMS = True
except Exception:
    has_mTMS = False

import multiprocessing
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.data.brainmesh_handler as brain
import invesalius.data.slice_ as sl
import invesalius.data.tractography as dti
import invesalius.data.vtk_utils as vtk_utils
import invesalius.gui.dialogs as dlg
import invesalius.project as prj
import invesalius.utils as utils
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher


class TaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        inner_panel = InnerTaskPanel(self)

        sizer = QHBoxLayout(self)
        sizer.setContentsMargins(7, 0, 7, 7)
        sizer.addWidget(inner_panel)


class InnerTaskPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.affine = np.identity(4)
        self.affine_vtk = None
        self.trekker = None
        self.n_tracts = const.N_TRACTS
        self.peel_depth = const.PEEL_DEPTH
        self.view_tracts = False
        self.seed_offset = const.SEED_OFFSET
        self.seed_radius = const.SEED_RADIUS
        self.brain_opacity = const.BRAIN_OPACITY
        self.brain_peel = None
        self.brain_actor = None
        self.n_peels = const.MAX_PEEL_DEPTH
        self.p_old = np.array([[0.0, 0.0, 0.0]])
        self.tracts_run = None
        self.trekker_cfg = const.TREKKER_CONFIG
        self.nav_status = False
        self.peel_loaded = False
        self.__bind_events()

        btn_load = QPushButton(_("FOD"), self)
        btn_load.setFixedSize(50, 23)
        btn_load.setToolTip(_("Load FOD"))
        btn_load.setEnabled(True)
        btn_load.clicked.connect(self.OnLinkFOD)

        btn_load_cfg = QPushButton(_("Configure"), self)
        btn_load_cfg.setFixedSize(65, 23)
        btn_load_cfg.setToolTip(_("Load Trekker configuration parameters"))
        btn_load_cfg.setEnabled(True)
        btn_load_cfg.clicked.connect(self.OnLoadParameters)

        btn_mask = QPushButton(_("Peel"), self)
        btn_mask.setFixedSize(50, 23)
        btn_mask.setToolTip(_("Create peel"))
        btn_mask.setEnabled(True)
        btn_mask.clicked.connect(self.OnCreatePeel)

        btn_act = QPushButton(_("ACT"), self)
        btn_act.setFixedSize(50, 23)
        btn_act.setToolTip(_("Load anatomical labels"))
        btn_act.setEnabled(True)
        btn_act.clicked.connect(self.OnLoadACT)

        line_btns = QHBoxLayout()
        line_btns.addWidget(btn_load, 1)
        line_btns.addWidget(btn_load_cfg, 1)
        line_btns.addWidget(btn_mask, 1)
        line_btns.addWidget(btn_act, 1)

        text_peel_depth = QLabel(_("Peeling depth (mm):"), self)
        spin_peel_depth = QSpinBox(self)
        spin_peel_depth.setFixedSize(50, 23)
        spin_peel_depth.setEnabled(True)
        spin_peel_depth.setRange(0, const.MAX_PEEL_DEPTH)
        spin_peel_depth.setValue(const.PEEL_DEPTH)
        spin_peel_depth.valueChanged.connect(
            lambda: self.OnSelectPeelingDepth(ctrl=spin_peel_depth)
        )

        text_ntracts = QLabel(_("Number tracts:"), self)
        spin_ntracts = QSpinBox(self)
        spin_ntracts.setFixedSize(50, 23)
        spin_ntracts.setEnabled(True)
        spin_ntracts.setRange(1, 2000)
        spin_ntracts.setValue(const.N_TRACTS)
        spin_ntracts.valueChanged.connect(lambda: self.OnSelectNumTracts(ctrl=spin_ntracts))

        text_offset = QLabel(_("Seed offset (mm):"), self)
        spin_offset = QDoubleSpinBox(self)
        spin_offset.setFixedSize(50, 23)
        spin_offset.setEnabled(True)
        spin_offset.setRange(0, 100.0)
        spin_offset.setSingleStep(0.1)
        spin_offset.setValue(self.seed_offset)
        spin_offset.valueChanged.connect(lambda: self.OnSelectOffset(ctrl=spin_offset))

        text_radius = QLabel(_("Seed radius (mm):"), self)
        spin_radius = QDoubleSpinBox(self)
        spin_radius.setFixedSize(50, 23)
        spin_radius.setEnabled(True)
        spin_radius.setRange(0, 100.0)
        spin_radius.setSingleStep(0.1)
        spin_radius.setValue(self.seed_radius)
        spin_radius.valueChanged.connect(lambda: self.OnSelectRadius(ctrl=spin_radius))

        text_opacity = QLabel(_("Brain opacity:"), self)
        spin_opacity = QDoubleSpinBox(self)
        spin_opacity.setFixedSize(50, 23)
        spin_opacity.setEnabled(False)
        spin_opacity.setRange(0, 1.0)
        spin_opacity.setSingleStep(0.1)
        spin_opacity.setValue(self.brain_opacity)
        spin_opacity.valueChanged.connect(lambda: self.OnSelectOpacity(ctrl=spin_opacity))
        self.spin_opacity = spin_opacity

        border = 1
        line_peel_depth = QHBoxLayout()
        line_peel_depth.addWidget(text_peel_depth, 1)
        line_peel_depth.addWidget(spin_peel_depth)

        line_ntracts = QHBoxLayout()
        line_ntracts.addWidget(text_ntracts, 1)
        line_ntracts.addWidget(spin_ntracts)

        line_offset = QHBoxLayout()
        line_offset.addWidget(text_offset, 1)
        line_offset.addWidget(spin_offset)

        line_radius = QHBoxLayout()
        line_radius.addWidget(text_radius, 1)
        line_radius.addWidget(spin_radius)

        line_opacity = QHBoxLayout()
        line_opacity.addWidget(text_opacity, 1)
        line_opacity.addWidget(spin_opacity)

        checktracts = QCheckBox(_("Enable tracts"), self)
        checktracts.setChecked(False)
        checktracts.setEnabled(False)
        checktracts.stateChanged.connect(lambda: self.OnEnableTracts(ctrl=checktracts))
        self.checktracts = checktracts

        checkpeeling = QCheckBox(_("Peel surface"), self)
        checkpeeling.setChecked(False)
        checkpeeling.setEnabled(False)
        checkpeeling.stateChanged.connect(lambda: self.OnShowPeeling(ctrl=checkpeeling))
        self.checkpeeling = checkpeeling

        checkACT = QCheckBox(_("ACT"), self)
        checkACT.setChecked(False)
        checkACT.setEnabled(False)
        checkACT.stateChanged.connect(lambda: self.OnEnableACT(ctrl=checkACT))
        self.checkACT = checkACT

        line_checks = QHBoxLayout()
        line_checks.addWidget(checktracts)
        line_checks.addWidget(checkpeeling)
        line_checks.addWidget(checkACT)

        main_sizer = QVBoxLayout(self)
        main_sizer.addLayout(line_btns)
        main_sizer.addLayout(line_peel_depth)
        main_sizer.addLayout(line_ntracts)
        main_sizer.addLayout(line_offset)
        main_sizer.addLayout(line_radius)
        main_sizer.addLayout(line_opacity)
        main_sizer.addLayout(line_checks)

    def __bind_events(self):
        Publisher.subscribe(self.OnCloseProject, "Close project data")
        Publisher.subscribe(self.OnUpdateTracts, "Set cross focal point")
        Publisher.subscribe(self.UpdateNavigationStatus, "Navigation status")

    def OnSelectPeelingDepth(self, ctrl):
        self.peel_depth = ctrl.value()
        if self.checkpeeling.isChecked():
            actor = self.brain_peel.get_actor(self.peel_depth)
            Publisher.sendMessage("Update peel", flag=True, actor=actor)
            Publisher.sendMessage(
                "Get peel centers and normals",
                centers=self.brain_peel.peel_centers,
                normals=self.brain_peel.peel_normals,
            )
            Publisher.sendMessage("Get init locator", locator=self.brain_peel.locator)
            self.peel_loaded = True

    def OnSelectNumTracts(self, ctrl):
        self.n_tracts = ctrl.value()
        Publisher.sendMessage("Update number of tracts", data=self.n_tracts)

    def OnSelectOffset(self, ctrl):
        self.seed_offset = ctrl.value()
        Publisher.sendMessage("Update seed offset", data=self.seed_offset)

    def OnSelectRadius(self, ctrl):
        self.seed_radius = ctrl.value()
        Publisher.sendMessage("Update seed radius", data=self.seed_radius)

    def OnSelectOpacity(self, ctrl):
        self.brain_actor.GetProperty().SetOpacity(ctrl.value())
        Publisher.sendMessage("Update peel", flag=True, actor=self.brain_actor)

    def OnShowPeeling(self, ctrl):
        if ctrl.isChecked():
            actor = self.brain_peel.get_actor(self.peel_depth)
            self.peel_loaded = True
            Publisher.sendMessage("Update peel visualization", data=self.peel_loaded)
        else:
            actor = None
            self.peel_loaded = False
            Publisher.sendMessage("Update peel visualization", data=self.peel_loaded)

        Publisher.sendMessage("Update peel", flag=ctrl.isChecked(), actor=actor)

    def OnEnableTracts(self, ctrl):
        self.view_tracts = ctrl.isChecked()
        Publisher.sendMessage("Update tracts visualization", data=self.view_tracts)
        if not self.view_tracts:
            Publisher.sendMessage("Remove tracts")
            Publisher.sendMessage("Update marker offset state", create=False)

    def OnEnableACT(self, ctrl):
        Publisher.sendMessage("Enable ACT", data=ctrl.isChecked())

    def UpdateNavigationStatus(self, nav_status, vis_status):
        self.nav_status = nav_status

    def UpdateDialog(self, msg):
        while self.tp.running:
            self.tp.dlg.Pulse(msg)
            if not self.tp.running:
                break

    def OnCreatePeel(self, event=None):
        Publisher.sendMessage("Begin busy cursor")
        from PySide6.QtWidgets import QApplication

        inv_proj = prj.Project()
        peels_dlg = dlg.PeelsCreationDlg(QApplication.activeWindow())
        ret = peels_dlg.exec()
        method = peels_dlg.method
        if ret == peels_dlg.Accepted:
            t_init = time.time()
            msg = "Creating peeled surface..."
            self.tp = dlg.TractographyProgressWindow(msg)

            slic = sl.Slice()
            ww = slic.window_width
            wl = slic.window_level
            affine = np.eye(4)
            if method == peels_dlg.FROM_FILES:
                try:
                    affine = slic.affine.copy()
                except AttributeError:
                    pass

            self.brain_peel = brain.Brain(self.n_peels, ww, wl, affine, inv_proj)
            if method == peels_dlg.FROM_MASK:
                choices = [i for i in inv_proj.mask_dict.values()]
                mask_index = peels_dlg.cb_masks.currentIndex()
                mask = choices[mask_index]
                option = 1
            else:
                mask = peels_dlg.mask_path
                option = 2
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as exec:
                futures = [
                    exec.submit(self.UpdateDialog, msg),
                    exec.submit(self.BrainLoading, option, mask),
                ]
                done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                self.tp.running = False

            t_end = time.time()
            print(f"Elapsed time - {t_end - t_init}")
            self.tp.Close()
            if self.tp.error:
                QMessageBox.critical(None, "Exception!", self.tp.error)

            self.brain_actor = self.brain_peel.get_actor(self.peel_depth)
            self.brain_actor.GetProperty().SetOpacity(self.brain_opacity)
            Publisher.sendMessage("Update peel", flag=True, actor=self.brain_actor)
            Publisher.sendMessage(
                "Get peel centers and normals",
                centers=self.brain_peel.peel_centers,
                normals=self.brain_peel.peel_normals,
            )
            Publisher.sendMessage("Get init locator", locator=self.brain_peel.locator)
            self.checkpeeling.setEnabled(True)
            self.checkpeeling.setChecked(True)
            self.spin_opacity.setEnabled(True)
            Publisher.sendMessage("Update status text in GUI", label=_("Brain model loaded"))
            self.peel_loaded = True
            Publisher.sendMessage("Update peel visualization", data=self.peel_loaded)
            del self.tp
            QMessageBox.information(None, _("InVesalius 3"), _("Peeled surface created"))
        peels_dlg.close()
        Publisher.sendMessage("End busy cursor")

    def BrainLoading(self, option, mask):
        if option == 1:
            self.brain_peel.from_mask(mask)
        else:
            self.brain_peel.from_mask_file(mask)

    def OnLinkFOD(self, event=None):
        Publisher.sendMessage("Begin busy cursor")
        filename = dlg.ShowImportOtherFilesDialog(
            const.ID_NIFTI_IMPORT, msg=_("Import Trekker FOD")
        )
        filename = utils.decode(filename, const.FS_ENCODE)

        if not self.affine_vtk:
            slic = sl.Slice()
            self.affine, self.affine_vtk, img_shift = slic.get_world_to_invesalius_vtk_affine()

        if filename:
            Publisher.sendMessage("Update status text in GUI", label=_("Busy"))
            t_init = time.time()
            try:
                msg = "Setting up FOD ... "
                self.tp = dlg.TractographyProgressWindow(msg)

                self.trekker = None
                file = filename.encode("utf-8")
                with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as exec:
                    futures = [
                        exec.submit(self.UpdateDialog, msg),
                        exec.submit(Trekker.initialize, file),
                    ]
                    done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                    completed_future = done.pop()
                    self.TrekkerCallback(completed_future)

                t_end = time.time()
                print(f"Elapsed time - {t_end - t_init}")
                self.tp.Close()
                if self.tp.error:
                    QMessageBox.critical(None, "Exception!", self.tp.error)
                del self.tp
                QMessageBox.information(None, _("InVesalius 3"), _("FOD Import successful"))
            except Exception:
                Publisher.sendMessage(
                    "Update status text in GUI", label=_("Trekker initialization failed.")
                )
                QMessageBox.warning(None, _("InVesalius 3"), _("Unable to load FOD."))

        Publisher.sendMessage("End busy cursor")

    def TrekkerCallback(self, trekker):
        self.tp.running = False
        print("Import Complete")
        if trekker is not None:
            self.trekker = trekker.result()
            self.trekker, n_threads = dti.set_trekker_parameters(self.trekker, self.trekker_cfg)

            self.checktracts.setEnabled(True)
            self.checktracts.setChecked(True)
            self.view_tracts = True

            Publisher.sendMessage("Update Trekker object", data=self.trekker)
            Publisher.sendMessage("Update number of threads", data=n_threads)
            Publisher.sendMessage("Update tracts visualization", data=1)
            Publisher.sendMessage("Update status text in GUI", label=_("Trekker initialized"))
            self.tp.running = False

    def OnLoadACT(self, event=None):
        if self.trekker:
            Publisher.sendMessage("Begin busy cursor")
            filename = dlg.ShowImportOtherFilesDialog(
                const.ID_NIFTI_IMPORT, msg=_("Import anatomical labels")
            )
            filename = utils.decode(filename, const.FS_ENCODE)

            if not self.affine_vtk:
                slic = sl.Slice()
                self.affine, self.affine_vtk, img_shift = slic.get_world_to_invesalius_vtk_affine()

            try:
                t_init = time.time()
                msg = "Setting up ACT..."
                self.tp = dlg.TractographyProgressWindow(msg)

                Publisher.sendMessage("Update status text in GUI", label=_("Busy"))
                if filename:
                    with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as exec:
                        futures = [
                            exec.submit(self.UpdateDialog, msg),
                            exec.submit(self.ACTLoading, filename),
                        ]
                        done, not_done = wait(futures, return_when=FIRST_COMPLETED)
                        self.tp.running = False

                    t_end = time.time()
                    print(f"Elapsed time - {t_end - t_init}")
                    self.tp.Close()
                    if self.tp.error:
                        QMessageBox.critical(None, "Exception!", self.tp.error)
                    del self.tp
                    self.checkACT.setEnabled(True)
                    self.checkACT.setChecked(True)
                    Publisher.sendMessage("Update ACT data", data=self.act_data_arr)
                    Publisher.sendMessage("Enable ACT", data=True)
                    Publisher.sendMessage(
                        "Update status text in GUI", label=_("Trekker ACT loaded")
                    )
                    QMessageBox.information(None, _("InVesalius 3"), _("ACT Import successful"))
            except Exception:
                Publisher.sendMessage(
                    "Update status text in GUI", label=_("ACT initialization failed.")
                )
                QMessageBox.warning(None, _("InVesalius 3"), _("Unable to load ACT."))

            Publisher.sendMessage("End busy cursor")
        else:
            QMessageBox.warning(None, _("InVesalius 3"), _("Load FOD image before the ACT."))

    def ACTLoading(self, filename):
        act_data = nb.squeeze_image(nb.load(filename))
        act_data = nb.as_closest_canonical(act_data)
        act_data.update_header()
        self.act_data_arr = act_data.get_fdata()
        self.trekker.pathway_stop_at_entry(filename.encode("utf-8"), -1)
        self.trekker.pathway_discard_if_ends_inside(filename.encode("utf-8"), 1)
        self.trekker.pathway_discard_if_enters(filename.encode("utf-8"), 0)

    def OnLoadParameters(self, event=None):
        import json

        filename = dlg.ShowLoadSaveDialog(
            message=_("Load Trekker configuration"), wildcard=_("JSON file (*.json)|*.json")
        )
        try:
            if filename:
                with open(filename) as json_file:
                    self.trekker_cfg = json.load(json_file)
                assert all(name in self.trekker_cfg for name in const.TREKKER_CONFIG)
                if self.trekker:
                    self.trekker, n_threads = dti.set_trekker_parameters(
                        self.trekker, self.trekker_cfg
                    )
                    Publisher.sendMessage("Update Trekker object", data=self.trekker)
                    Publisher.sendMessage("Update number of threads", data=n_threads)

                Publisher.sendMessage("Update status text in GUI", label=_("Trekker config loaded"))

        except (AssertionError, json.decoder.JSONDecodeError):
            self.trekker_cfg = const.TREKKER_CONFIG
            QMessageBox.warning(
                None,
                _("InVesalius 3"),
                _("File incompatible, using default configuration."),
            )
            Publisher.sendMessage("Update status text in GUI", label="")

    def OnUpdateTracts(self, position):
        """
        Minimal working version of tract computation. Updates when cross sends Pubsub message to update.
        Position refers to the coordinates in InVesalius 2D space. To represent the same coordinates in the 3D space,
        flip_x the coordinates and multiply the z coordinate by -1. This is all done in the flix_x function.

        :param arg: event for pubsub
        :param position: list or array with the x, y, and z coordinates in InVesalius space
        """
        if self.view_tracts and not self.nav_status:
            coord_flip = list(position[:3])
            coord_flip[1] = -coord_flip[1]
            dti.compute_and_visualize_tracts(
                self.trekker, coord_flip, self.affine, self.affine_vtk, self.n_tracts
            )

    def OnCloseProject(self):
        self.trekker = None
        self.trekker_cfg = const.TREKKER_CONFIG

        self.checktracts.setChecked(False)
        self.checktracts.setEnabled(False)
        self.checkpeeling.setChecked(False)
        self.checkpeeling.setEnabled(False)
        self.checkACT.setChecked(False)
        self.checkACT.setEnabled(False)

        self.spin_opacity.setValue(const.BRAIN_OPACITY)
        self.spin_opacity.setEnabled(False)
        Publisher.sendMessage("Update peel", flag=False, actor=self.brain_actor)

        self.peel_depth = const.PEEL_DEPTH
        self.n_tracts = const.N_TRACTS

        Publisher.sendMessage("Remove tracts")
