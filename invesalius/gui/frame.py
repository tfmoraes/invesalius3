# --------------------------------------------------------------------
# Software:     InVesalius - Software de Reconstrucao 3D de Imagens Medicas
# Copyright:    (C) 2001  Centro de Pesquisas Renato Archer
# Homepage:     http://www.softwarepublico.gov.br
# Contact:      invesalius@cti.gov.br
# License:      GNU - GPL 2 (LICENSE.txt/LICENCA.txt)
# --------------------------------------------------------------------
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
# --------------------------------------------------------------------

import errno
import os.path
import platform
import subprocess
import sys
import webbrowser

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QStatusBar,
    QSystemTrayIcon,
    QToolBar,
    QWidget,
)

import invesalius.constants as const
import invesalius.gui.default_tasks as tasks
import invesalius.gui.default_viewers as viewers
import invesalius.gui.dialogs as dlg
import invesalius.gui.import_bitmap_panel as imp_bmp
import invesalius.gui.import_panel as imp
import invesalius.gui.log as log
import invesalius.gui.preferences as preferences

#  import invesalius.gui.import_network_panel as imp_net
import invesalius.project as prj
import invesalius.session as ses
import invesalius.utils as utils
from invesalius import inv_paths
from invesalius.data.slice_ import Slice
from invesalius.gui import project_properties
from invesalius.gui.interactive_shell import InteractiveShellFrame
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher

VIEW_TOOLS = [ID_LAYOUT, ID_TEXT, ID_RULER] = [const._new_id_ref() for number in range(3)]

[ID_SHOW_LOG_VIEWER, ID_INTERACTIVE_SHELL] = [const._new_id_ref() for number in range(2)]

WILDCARD_EXPORT_SLICE = "HDF5 (*.hdf5);;NIfTI 1 (*.nii);;Compressed NIfTI (*.nii.gz)"

IDX_EXT = {0: ".hdf5", 1: ".nii", 2: ".nii.gz"}


class MessageWatershed(QWidget):
    def __init__(self, prnt, msg):
        super().__init__(prnt, Qt.Popup)
        self.txt = QLabel(msg, self)

        layout = QHBoxLayout(self)
        layout.addWidget(self.txt)
        self.setLayout(layout)
        self.adjustSize()


class Frame(QMainWindow):
    """
    Main frame of the whole software.
    """

    def __init__(self, prnt):
        """
        Initialize frame, given its parent.
        """
        super().__init__(prnt)
        self.setObjectName("Frame")
        self.resize(1024, 748)
        self.setWindowTitle("InVesalius 3")

        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(
                (geo.width() - self.width()) // 2,
                (geo.height() - self.height()) // 2,
            )

        icon_path = inv_paths.ICON_DIR.joinpath("invesalius.ico")
        self.setWindowIcon(QIcon(str(icon_path)))

        self.mw = None
        self._last_viewer_orientation_focus = const.AXIAL_STR

        if sys.platform != "darwin":
            self.showMaximized()

        self.sizeChanged = True

        self._show_navigator_message = True
        self.edit_data_notebook_label = False

        main_menu = MenuBar(self)
        self.actived_interpolated_slices = main_menu.view_menu
        self.actived_navigation_mode = main_menu.mode_menu
        self.actived_dbs_mode = main_menu.mode_dbs_action
        self.tools_menu = main_menu.tools_menu

        self.setMenuBar(main_menu)
        self.setStatusBar(StatusBar(self))

        self._menu_bar = main_menu

        self.__init_layout()
        self.__bind_events()

        self._idle_timer = QTimer(self)
        self._idle_timer.setInterval(100)
        self._idle_timer.timeout.connect(self._OnIdle)
        self._idle_timer.start()

    def __bind_events(self):
        """
        Bind events related to pubsub.
        """
        sub = Publisher.subscribe
        sub(self._BeginBusyCursor, "Begin busy cursor")
        sub(self._ShowContentPanel, "Cancel DICOM load")
        sub(self._EndBusyCursor, "End busy cursor")
        sub(self._HideContentPanel, "Hide content panel")
        sub(self._HideImportPanel, "Hide import panel")
        sub(self._HideTask, "Hide task panel")
        sub(self._ShowTask, "Show task panel")
        sub(self._SetProjectName, "Set project name")
        sub(self._ShowContentPanel, "Show content panel")
        sub(self._ShowImportPanel, "Show import panel in frame")
        sub(self.ShowPreferences, "Open preferences menu")
        sub(self._ShowImportNetwork, "Show retrieve dicom panel")
        sub(self._ShowImportBitmap, "Show import bitmap panel in frame")
        sub(self._ShowTask, "Show task panel")
        sub(self._UpdateAUI, "Update AUI")
        sub(self._UpdateViewerFocus, "Set viewer orientation focus")
        sub(self._Exit, "Exit")

    def __init_layout(self):
        """
        Build layout using QMainWindow dock widgets and central widget.
        """
        self._viewers_panel = viewers.Panel(self)
        self._task_panel = tasks.Panel(self)
        self._import_panel = imp.Panel(self)
        self._import_bitmap_panel = imp_bmp.Panel(self)

        self.setCentralWidget(self._viewers_panel)
        self._viewers_panel.hide()

        self._task_dock = QDockWidget("", self)
        self._task_dock.setTitleBarWidget(QWidget())
        self._task_dock.setWidget(self._task_panel)
        self._task_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self.addDockWidget(Qt.LeftDockWidgetArea, self._task_dock)

        self._import_dock = QDockWidget(_("Preview medical data to be reconstructed"), self)
        self._import_dock.setWidget(self._import_panel)
        self._import_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self._import_dock.hide()
        self.addDockWidget(Qt.RightDockWidgetArea, self._import_dock)

        self._import_bmp_dock = QDockWidget(_("Preview bitmap to be reconstructed"), self)
        self._import_bmp_dock.setWidget(self._import_bitmap_panel)
        self._import_bmp_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)
        self._import_bmp_dock.hide()
        self.addDockWidget(Qt.RightDockWidgetArea, self._import_bmp_dock)

        t1 = ProjectToolBar(self)
        t2 = HistoryToolBar(self)
        t3 = LayoutToolBar(self)
        t4 = ObjectToolBar(self)
        t5 = SliceToolBar(self)

        self.addToolBar(Qt.TopToolBarArea, t1)
        self.addToolBar(Qt.TopToolBarArea, t2)
        self.addToolBar(Qt.TopToolBarArea, t3)
        self.addToolBar(Qt.TopToolBarArea, t4)
        self.addToolBar(Qt.TopToolBarArea, t5)

    def _BeginBusyCursor(self):
        """
        Start busy cursor.
        """
        QApplication.setOverrideCursor(Qt.WaitCursor)

    def _EndBusyCursor(self):
        """
        End busy cursor.
        """
        try:
            QApplication.restoreOverrideCursor()
        except Exception:
            pass

    def _Exit(self):
        """
        Exit InVesalius.
        """
        for w in QApplication.topLevelWidgets():
            w.close()
        self.close()
        if hasattr(sys, "frozen") and sys.platform == "darwin":
            sys.exit(0)

    def _HideContentPanel(self):
        """
        Hide data and tasks panels.
        """
        self._viewers_panel.hide()
        self._task_dock.show()

    def _HideImportPanel(self):
        """
        Hide import panel and show tasks.
        """
        self._import_dock.hide()
        self._viewers_panel.hide()
        self._task_dock.show()

    def _HideTask(self):
        """
        Hide task panel.
        """
        if self._task_dock.isVisible():
            self._task_dock.hide()
            QApplication.processEvents()
            Publisher.sendMessage("Set layout button full")

    def _SetProjectName(self, proj_name=""):
        """
        Set project name into frame's title.
        """
        if not (proj_name):
            self.setWindowTitle("InVesalius 3")
        else:
            self.setWindowTitle(f"{proj_name} - InVesalius 3")

    def _ShowContentPanel(self):
        """
        Show viewers and task, hide import panel.
        """
        Publisher.sendMessage("Set layout button full")
        self._import_dock.hide()
        self._import_bmp_dock.hide()
        self._viewers_panel.show()
        self._task_dock.show()

    def _ShowImportNetwork(self):
        """
        Show viewers and task, hide import panel.
        """
        Publisher.sendMessage("Set layout button full")
        # self._retrieve_dock.show()
        self._viewers_panel.hide()
        self._task_dock.hide()
        self._import_dock.hide()

    def _ShowImportBitmap(self):
        """
        Show bitmap import panel.
        """
        Publisher.sendMessage("Set layout button full")
        self._import_bmp_dock.show()
        self._viewers_panel.hide()
        self._task_dock.hide()
        self._import_dock.hide()

    def _ShowHelpMessage(self, message):
        pos = self._viewers_panel.mapToGlobal(self._viewers_panel.pos())
        self.mw = MessageWatershed(self, message)
        self.mw.move(pos)
        self.mw.show()

    def _ShowImportPanel(self):
        """
        Show only DICOM import panel.
        """
        Publisher.sendMessage("Set layout button data only")
        self._import_dock.show()
        self._viewers_panel.hide()
        self._task_dock.hide()

    def _ShowTask(self):
        """
        Show task panel.
        """
        self._task_dock.show()

    def _UpdateAUI(self):
        """
        Refresh layout (no-op in Qt, kept for pubsub compatibility).
        """
        pass

    def _UpdateViewerFocus(self, orientation):
        if orientation in (const.AXIAL_STR, const.CORONAL_STR, const.SAGITAL_STR):
            self._last_viewer_orientation_focus = orientation

    def CloseProject(self):
        Publisher.sendMessage("Close Project")

    def ExitDialog(self):
        msg = _("Are you sure you want to exit?")
        cb_text = "Store session"
        box = QMessageBox(
            QMessageBox.Question,
            "Invesalius 3",
            msg,
            QMessageBox.Yes | QMessageBox.No,
            self,
        )
        box.setDefaultButton(QMessageBox.No)
        cb = box.checkBox()
        if cb is None:
            from PySide6.QtWidgets import QCheckBox

            cb = QCheckBox(cb_text)
            box.setCheckBox(cb)

        answer = box.exec()
        save = cb.isChecked()

        if not save and answer == QMessageBox.Yes:
            log.invLogger.closeLogging()
            return 1
        elif save and answer == QMessageBox.Yes:
            log.invLogger.closeLogging()
            return 2
        else:
            return 0

    def closeEvent(self, event):
        """
        Exit InVesalius: disconnect tracker and send 'Exit' message.
        """
        status = self.ExitDialog()
        if status:
            Publisher.sendMessage("Disconnect tracker")
            Publisher.sendMessage("Exit")

            try:
                from invesalius import enhanced_logging

                enhanced_logging.enhanced_logger.cleanup()
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(f"Error cleaning up log viewer: {e}")

            if status == 1:
                Publisher.sendMessage("Exit session")
            event.accept()
        else:
            event.ignore()

    def OnMenuClick(self, action):
        """
        Capture event from action trigger on menu / toolbar.
        Uses action's data() to get the associated ID.
        """
        id = action.data()
        if id is None:
            return

        if id == const.ID_DICOM_IMPORT:
            self.ShowImportDicomPanel()
        elif id == const.ID_PROJECT_OPEN:
            self.ShowOpenProject()
        elif id == const.ID_ANALYZE_IMPORT:
            self.ShowImportOtherFiles(id)
        elif id == const.ID_NIFTI_IMPORT:
            self.ShowImportOtherFiles(id)
        elif id == const.ID_PARREC_IMPORT:
            self.ShowImportOtherFiles(id)
        elif id == const.ID_TIFF_JPG_PNG:
            self.ShowBitmapImporter()
        elif id == const.ID_PROJECT_SAVE:
            session = ses.Session()
            if session.temp_item:
                self.ShowSaveAsProject()
            else:
                self.SaveProject()
        elif id == const.ID_PROJECT_SAVE_AS:
            self.ShowSaveAsProject()
        elif id == const.ID_EXPORT_SLICE:
            self.ExportProject()
        elif id == const.ID_PROJECT_PROPERTIES:
            self.ShowProjectProperties()
        elif id == const.ID_PROJECT_CLOSE:
            self.CloseProject()
        elif id == const.ID_EXIT:
            self.close()
        elif id == const.ID_ABOUT:
            self.ShowAbout()
        elif id == const.ID_START:
            self.ShowGettingStarted()
        elif id in (const.ID_PREFERENCES, const.ID_PREFERENCES_TOOLBAR):
            self.ShowPreferences()
        elif id == const.ID_DICOM_NETWORK:
            self.ShowRetrieveDicomPanel()
        elif id in (const.ID_FLIP_X, const.ID_FLIP_Y, const.ID_FLIP_Z):
            axis = {const.ID_FLIP_X: 2, const.ID_FLIP_Y: 1, const.ID_FLIP_Z: 0}[id]
            self.FlipVolume(axis)
        elif id in (const.ID_SWAP_XY, const.ID_SWAP_XZ, const.ID_SWAP_YZ):
            axes = {
                const.ID_SWAP_XY: (2, 1),
                const.ID_SWAP_XZ: (2, 0),
                const.ID_SWAP_YZ: (1, 0),
            }[id]
            self.SwapAxes(axes)
        elif id == const.ID_REORIENT_IMG:
            self.OnReorientImg()
        elif id == const.ID_EDIT_UNDO:
            self.OnUndo()
        elif id == const.ID_EDIT_REDO:
            self.OnRedo()
        elif id == const.ID_GOTO_SLICE:
            self.OnGotoSlice()
        elif id == const.ID_GOTO_COORD:
            self.GoToDialogScannerCoord()
        elif id == const.ID_BOOLEAN_MASK:
            self.OnMaskBoolean()
        elif id == const.ID_CLEAN_MASK:
            self.OnCleanMask()
        elif id == const.ID_MASK_DENSITY_MEASURE:
            ddlg = dlg.MaskDensityDialog(self)
            ddlg.show()
        elif id == const.ID_MANUAL_WWWL:
            wwwl_dlg = dlg.ManualWWWLDialog(self)
            wwwl_dlg.show()
        elif id == const.ID_THRESHOLD_SEGMENTATION:
            Publisher.sendMessage("Show panel", panel_id=const.ID_THRESHOLD_SEGMENTATION)
            Publisher.sendMessage("Disable actual style")
            Publisher.sendMessage("Enable style", style=const.STATE_DEFAULT)
        elif id == const.ID_MANUAL_SEGMENTATION:
            Publisher.sendMessage("Show panel", panel_id=const.ID_MANUAL_SEGMENTATION)
            Publisher.sendMessage("Disable actual style")
            Publisher.sendMessage("Enable style", style=const.SLICE_STATE_EDITOR)
        elif id == const.ID_WATERSHED_SEGMENTATION:
            Publisher.sendMessage("Show panel", panel_id=const.ID_WATERSHED_SEGMENTATION)
            Publisher.sendMessage("Disable actual style")
            Publisher.sendMessage("Enable style", style=const.SLICE_STATE_WATERSHED)
        elif id == const.ID_FILL_HOLE_AUTO:
            self.OnFillHolesAutomatically()
        elif id == const.ID_FLOODFILL_MASK:
            self.OnFillHolesManually()
        elif id == const.ID_REMOVE_MASK_PART:
            self.OnRemoveMaskParts()
        elif id == const.ID_SELECT_MASK_PART:
            self.OnSelectMaskParts()
        elif id == const.ID_FLOODFILL_SEGMENTATION:
            self.OnFFillSegmentation()
        elif id == const.ID_SEGMENTATION_BRAIN:
            self.OnBrainSegmentation()
        elif id == const.ID_SEGMENTATION_SUBPART:
            self.OnSubpartSegmentation()
        elif id == const.ID_SEGMENTATION_TRACHEA:
            self.OnTracheSegmentation()
        elif id == const.ID_SEGMENTATION_MANDIBLE_CT:
            self.OnMandibleCTSegmentation()
        elif id == const.ID_PLANNING_CRANIOPLASTY:
            self.OnImplantCTSegmentation()
        elif id == const.ID_VIEW_INTERPOLATED:
            action_obj = self._menu_bar._action_map.get(const.ID_VIEW_INTERPOLATED)
            if action_obj:
                self.OnInterpolatedSlices(action_obj.isChecked())
        elif id == const.ID_MODE_NAVIGATION:
            Publisher.sendMessage("Hide dbs folder")
            Publisher.sendMessage("Show target button")
            dbs_action = self._menu_bar._action_map.get(const.ID_MODE_DBS)
            if dbs_action:
                dbs_action.setChecked(False)
            nav_action = self._menu_bar._action_map.get(const.ID_MODE_NAVIGATION)
            if nav_action:
                self.OnNavigationMode(nav_action.isChecked())
        elif id == const.ID_MODE_DBS:
            self.OnDbsMode()
        elif id == const.ID_CROP_MASK:
            self.OnCropMask()
        elif id == const.ID_MASK_3D_PREVIEW:
            action_obj = self._menu_bar._action_map.get(const.ID_MASK_3D_PREVIEW)
            if action_obj:
                self.OnEnableMask3DPreview(value=action_obj.isChecked())
        elif id == const.ID_MASK_3D_AUTO_RELOAD:
            session = ses.Session()
            action_obj = self._menu_bar._action_map.get(const.ID_MASK_3D_AUTO_RELOAD)
            if action_obj:
                session.SetConfig("auto_reload_preview", action_obj.isChecked())
        elif id == const.ID_MASK_3D_RELOAD:
            self.OnUpdateMaskPreview()
        elif id == const.ID_CREATE_SURFACE:
            Publisher.sendMessage("Open create surface dialog")
        elif id == const.ID_REMOVE_NON_VISIBLE_FACES:
            dialog = dlg.RemoveNonVisibleFacesDialog(self)
            dialog.show()
        elif id == const.ID_CREATE_MASK:
            Publisher.sendMessage("New mask from shortcut")
        elif id == const.ID_PLUGINS_SHOW_PATH:
            self.ShowPluginsFolder()
        elif id == ID_SHOW_LOG_VIEWER:
            self.OnShowLogViewer()
        elif id == ID_INTERACTIVE_SHELL:
            self.OnInteractiveShell()
        elif id == const.ID_TASK_BAR:
            if self._task_dock.isVisible():
                self._HideTask()
            else:
                self._ShowTask()
            self.setFocus()

    def keyPressEvent(self, event):
        """
        Handle all key events at a global level.
        """
        keycode = event.key()
        modifiers = event.modifiers()

        focused = QApplication.focusWidget()
        is_search_field = False
        is_shell_focused = False

        if focused and isinstance(focused, (QLineEdit,)):
            is_search_field = True

        try:
            from PySide6.QtWidgets import QTextEdit

            if focused and isinstance(focused, QTextEdit):
                is_shell_focused = True
        except Exception:
            pass

        if modifiers & Qt.ControlModifier:
            text = event.text()
            if text.lower() in ("s", "q"):
                super().keyPressEvent(event)
                return

        qt_movement_map = {
            Qt.Key_Left: const.MOVE_MARKER_LEFT_KEYCODE
            if hasattr(const, "MOVE_MARKER_LEFT_KEYCODE")
            else None,
            Qt.Key_Right: const.MOVE_MARKER_RIGHT_KEYCODE
            if hasattr(const, "MOVE_MARKER_RIGHT_KEYCODE")
            else None,
            Qt.Key_Up: const.MOVE_MARKER_ANTERIOR_KEYCODE
            if hasattr(const, "MOVE_MARKER_ANTERIOR_KEYCODE")
            else None,
            Qt.Key_Down: const.MOVE_MARKER_POSTERIOR_KEYCODE
            if hasattr(const, "MOVE_MARKER_POSTERIOR_KEYCODE")
            else None,
        }

        mapped_keycode = qt_movement_map.get(keycode)
        if (
            mapped_keycode is not None
            and mapped_keycode in const.MOVEMENT_KEYCODES
            and not self.edit_data_notebook_label
            and not is_search_field
            and not is_shell_focused
        ):
            Publisher.sendMessage("Move marker by keyboard", keycode=mapped_keycode)
            return

        if (
            keycode == Qt.Key_Delete
            and not self.edit_data_notebook_label
            and not is_search_field
            and not is_shell_focused
        ):
            Publisher.sendMessage("Delete selected markers")
            return

        super().keyPressEvent(event)

    def resizeEvent(self, event):
        """
        Refresh GUI when frame is resized.
        """
        super().resizeEvent(event)
        self.sizeChanged = True

    def _OnIdle(self):
        if self.sizeChanged:
            self.sizeChanged = False

    def OnDbsMode(self):
        dbs_action = self._menu_bar._action_map.get(const.ID_MODE_DBS)
        st = dbs_action.isChecked() if dbs_action else False
        Publisher.sendMessage("Hide target button")
        if st:
            self.OnNavigationMode(st)
            Publisher.sendMessage("Show dbs folder")
        else:
            self.OnNavigationMode(st)
            Publisher.sendMessage("Hide dbs folder")
        nav_action = self._menu_bar._action_map.get(const.ID_MODE_NAVIGATION)
        if nav_action:
            nav_action.setChecked(False)

    def OnNavigationMode(self, status):
        if status and self._show_navigator_message and sys.platform != "win32":
            QMessageBox.information(
                self,
                "Info",
                _("Currently the Navigation mode is only working on Windows"),
            )
            self._show_navigator_message = False
        Publisher.sendMessage("Set navigation mode", status=status)
        if not status:
            Publisher.sendMessage("Remove sensors ID")

    def ShowPreferences(self, page=0):
        preferences_dialog = preferences.Preferences(self, page)
        preferences_dialog.LoadPreferences()

        if hasattr(preferences_dialog, "Center"):
            preferences_dialog.Center()

        from PySide6.QtWidgets import QDialog

        if preferences_dialog.exec() == QDialog.Accepted:
            values = preferences_dialog.GetPreferences()

            session = ses.Session()

            rendering = values[const.RENDERING]
            surface_interpolation = values[const.SURFACE_INTERPOLATION]
            language = values[const.LANGUAGE]
            slice_interpolation = values.get(const.SLICE_INTERPOLATION, 0)
            file_logging = values.get(const.FILE_LOGGING, 0)
            file_logging_level = values.get(const.FILE_LOGGING_LEVEL, 0)
            append_log_file = values.get(const.APPEND_LOG_FILE, 0)
            logging_file = values.get(const.LOGFILE, "")
            console_logging = values.get(const.CONSOLE_LOGGING, 0)
            console_logging_level = values.get(const.CONSOLE_LOGGING_LEVEL, 0)
            logging = values.get(const.LOGGING, 0)
            logging_level = values.get(const.LOGGING_LEVEL, 0)

            session.SetConfig("rendering", rendering)
            session.SetConfig("surface_interpolation", surface_interpolation)
            session.SetConfig("language", language)
            session.SetConfig("slice_interpolation", slice_interpolation)
            session.SetConfig("file_logging", file_logging)
            session.SetConfig("file_logging_level", file_logging_level)
            session.SetConfig("append_log_file", append_log_file)
            session.SetConfig("logging_file", logging_file)
            session.SetConfig("console_logging", console_logging)
            session.SetConfig("console_logging_level", console_logging_level)
            session.SetConfig("do_logging", logging)
            session.SetConfig("logging_level", logging_level)
            session.SetConfig("append_log_file", append_log_file)
            session.SetConfig("logging_file", logging_file)

            Publisher.sendMessage("Remove Volume")
            Publisher.sendMessage("Reset Raycasting")
            Publisher.sendMessage("Update Slice Interpolation")
            Publisher.sendMessage("Update Slice Interpolation MenuBar")
            Publisher.sendMessage("Update Navigation Mode MenuBar")
            Publisher.sendMessage("Update Surface Interpolation")

    def ShowAbout(self):
        dlg.ShowAboutDialog(self)

    def SaveProject(self):
        Publisher.sendMessage("Show save dialog", save_as=False)

    def ShowGettingStarted(self):
        webbrowser.open("https://invesalius.github.io/docs/user_guide/user_guide.html")

    def ShowImportDicomPanel(self):
        Publisher.sendMessage("Show import directory dialog")

    def ShowImportOtherFiles(self, id_file):
        Publisher.sendMessage("Show import other files dialog", id_type=id_file)

    def ShowRetrieveDicomPanel(self):
        Publisher.sendMessage("Show retrieve dicom panel")

    def ShowOpenProject(self):
        Publisher.sendMessage("Show open project dialog")

    def ShowSaveAsProject(self):
        Publisher.sendMessage("Show save dialog", save_as=True)

    def ExportProject(self):
        p = prj.Project()
        session = ses.Session()
        last_directory = session.GetConfig("last_directory_export_prj", "")

        filename, selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export slice ...",
            os.path.join(last_directory, os.path.split(p.name)[-1]),
            WILDCARD_EXPORT_SLICE,
        )

        if filename:
            filters = WILDCARD_EXPORT_SLICE.split(";;")
            try:
                filter_idx = filters.index(selected_filter)
            except ValueError:
                filter_idx = 0
            ext = IDX_EXT[filter_idx]
            dirpath = os.path.split(filename)[0]
            if not filename.endswith(ext):
                filename += ext
            try:
                p.export_project(filename)
            except OSError as err:
                if err.errno == errno.EACCES:
                    message = f"It was not possible to save because you don't have permission to write at {dirpath}"
                else:
                    message = "It was not possible to save because"
                d = dlg.ErrorMessageBox(None, "Save project error", f"{message}:\n{err}")
                d.exec()
            else:
                session.SetConfig("last_directory_export_prj", dirpath)

    def ShowProjectProperties(self):
        from PySide6.QtWidgets import QDialog

        window = project_properties.ProjectProperties(self)
        if window.exec() == QDialog.Accepted:
            p = prj.Project()
            name_val = (
                window.name_txt.text()
                if hasattr(window.name_txt, "text")
                else window.name_txt.GetValue()
            )
            if name_val != p.name:
                p.name = name_val
                session = ses.Session()
                session.ChangeProject()
                self._SetProjectName(p.name)

    def ShowBitmapImporter(self):
        Publisher.sendMessage("Show bitmap dialog")

    def FlipVolume(self, axis):
        Publisher.sendMessage("Flip volume", axis=axis)
        Publisher.sendMessage("Reload actual slice")

    def SwapAxes(self, axes):
        Publisher.sendMessage("Swap volume axes", axes=axes)
        Publisher.sendMessage("Update scroll")
        Publisher.sendMessage("Reload actual slice")

    def OnUndo(self, evt=None):
        Publisher.sendMessage("Undo edition")

    def OnRedo(self, evt=None):
        Publisher.sendMessage("Redo edition")

    def OnGotoSlice(self):
        gt_dialog = dlg.GoToDialog(init_orientation=self._last_viewer_orientation_focus)
        if hasattr(gt_dialog, "CenterOnParent"):
            gt_dialog.CenterOnParent()
        gt_dialog.exec()
        self.update()

    def GoToDialogScannerCoord(self):
        gts_dialog = dlg.GoToDialogScannerCoord()
        if hasattr(gts_dialog, "CenterOnParent"):
            gts_dialog.CenterOnParent()
        gts_dialog.exec()
        self.update()

    def OnMaskBoolean(self):
        Publisher.sendMessage("Show boolean dialog")

    def OnCleanMask(self):
        Publisher.sendMessage("Clean current mask")
        Publisher.sendMessage("Reload actual slice")

    def OnReorientImg(self):
        Publisher.sendMessage("Enable style", style=const.SLICE_STATE_REORIENT)
        rdlg = dlg.ReorientImageDialog()
        rdlg.show()

    def OnFillHolesManually(self):
        Publisher.sendMessage("Enable style", style=const.SLICE_STATE_MASK_FFILL)

    def OnFillHolesAutomatically(self):
        fdlg = dlg.FillHolesAutoDialog(_("Fill holes automatically"))
        fdlg.show()

    def OnRemoveMaskParts(self):
        Publisher.sendMessage("Enable style", style=const.SLICE_STATE_REMOVE_MASK_PARTS)

    def OnSelectMaskParts(self):
        Publisher.sendMessage("Enable style", style=const.SLICE_STATE_SELECT_MASK_PARTS)

    def OnFFillSegmentation(self):
        Publisher.sendMessage("Enable style", style=const.SLICE_STATE_FFILL_SEGMENTATION)

    def OnBrainSegmentation(self):
        from invesalius.gui import deep_learning_seg_dialog

        if deep_learning_seg_dialog.HAS_TORCH or deep_learning_seg_dialog.HAS_TINYGRAD:
            d = deep_learning_seg_dialog.BrainSegmenterDialog(self)
            d.show()
        else:
            QMessageBox.information(
                self,
                "InVesalius 3 - Brain segmenter",
                _(
                    "It's not possible to run brain segmenter because your system doesn't have the following modules installed:"
                )
                + " Torch",
            )

    def OnSubpartSegmentation(self):
        from invesalius.gui import deep_learning_seg_dialog

        if deep_learning_seg_dialog.HAS_TORCH or deep_learning_seg_dialog.HAS_TINYGRAD:
            d = deep_learning_seg_dialog.SubpartSegmenterDialog(self)
            d.show()
        else:
            QMessageBox.information(
                self,
                "InVesalius 3 - Brain subpart Segmentation",
                _(
                    "It's not possible to run subpart segmentation because your system doesn't have the following modules installed:"
                )
                + " Torch",
            )

    def OnTracheSegmentation(self):
        from invesalius.gui import deep_learning_seg_dialog

        if deep_learning_seg_dialog.HAS_TORCH:
            d = deep_learning_seg_dialog.TracheaSegmenterDialog(self)
            d.show()
        else:
            QMessageBox.information(
                self,
                "InVesalius 3 - Trachea segmenter",
                _(
                    "It's not possible to run trachea segmenter because your system doesn't have the following modules installed:"
                )
                + " Torch",
            )

    def OnImplantCTSegmentation(self):
        from invesalius.gui import deep_learning_seg_dialog

        if deep_learning_seg_dialog.HAS_TORCH:
            d = deep_learning_seg_dialog.ImplantSegmenterDialog(self)
            d.show()
        else:
            QMessageBox.information(
                self,
                "InVesalius 3 - Implant prediction",
                _(
                    "It's not possible to run implant prediction because your system doesn't have the following modules installed:"
                )
                + " Torch",
            )

    def OnMandibleCTSegmentation(self):
        from invesalius.gui import deep_learning_seg_dialog

        if deep_learning_seg_dialog.HAS_TORCH:
            d = deep_learning_seg_dialog.MandibleSegmenterDialog(self)
            d.show()
        else:
            QMessageBox.information(
                self,
                "InVesalius 3 - Trachea segmenter",
                _(
                    "It's not possible to run mandible segmenter because your system doesn't have the following modules installed:"
                )
                + " Torch",
            )

    def OnInterpolatedSlices(self, status):
        Publisher.sendMessage("Set interpolated slices", flag=status)

    def OnCropMask(self):
        Publisher.sendMessage("Enable style", style=const.SLICE_STATE_CROP_MASK)

    def OnEnableMask3DPreview(self, value):
        if value:
            Publisher.sendMessage("Enable mask 3D preview")
        else:
            Publisher.sendMessage("Disable mask 3D preview")

    def OnUpdateMaskPreview(self):
        Publisher.sendMessage("Update mask 3D preview")

    def ShowPluginsFolder(self):
        inv_paths.create_conf_folders()
        path = str(inv_paths.USER_PLUGINS_DIRECTORY)
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])

    def OnShowLogViewer(self):
        """Show the log viewer."""
        try:
            from invesalius import enhanced_logging

            enhanced_logging.show_log_viewer(self)
        except Exception as e:
            print(f"Error showing log viewer: {e}")
            import traceback

            traceback.print_exc()
            QMessageBox.critical(self, "Error", f"Error showing log viewer: {e}")

    def OnInteractiveShell(self):
        """Show the interactive Python shell."""
        import numpy as np

        app_context = {
            "project": prj.Project(),
            "slice": Slice(),
            "session": ses.Session(),
            "frame": self,
            "Publisher": Publisher,
            "np": np,
            "app": QApplication.instance(),
        }

        intro_text = (
            "InVesalius Interactive Python Shell\n"
            "===========================\n"
            "Available objects:\n"
            "  app             - Main application instance\n"
            "  frame           - Main frame window\n"
            "  project         - Current project data\n"
            "  slice           - Slice singleton for image data\n"
            "  Publisher       - PubSub publisher for messaging\n"
            "  np              - NumPy module\n"
            "\nIf Navigation mode is active the following objects are also available:\n"
            "  markers         - MarkersControl instance for navigation markers\n"
            "  navigation      - Navigation instance for controlling navigation\n"
            "  robot           - Robot instance for robotic control\n"
            "  tracker         - Tracker instance for tracking data\n"
            "\nExample usage:\n"
            "  >>> frame.windowTitle()\n"
            "  >>> project.name\n"
            "  >>> slice.current_mask\n"
            "  >>> Publisher.sendMessage('Set threshold values', threshold_range=(100, 500))\n"
            "\n"
        )
        if not hasattr(self, "_shell_window") or not self._shell_window:
            self._shell_window = InteractiveShellFrame(self, app_context, introText=intro_text)

        self._shell_window.show()
        self._shell_window.raise_()

        Publisher.sendMessage("Add navigation context to interactive shell")


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


class MenuBar(QMenuBar):
    """
    MenuBar which contains menus used to control project, tools and
    help.
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.parent = parent
        self._plugins_menu_ids = {}
        self._action_map = {}

        self.enable_items = [
            const.ID_PROJECT_SAVE,
            const.ID_PROJECT_SAVE_AS,
            const.ID_EXPORT_SLICE,
            const.ID_PROJECT_PROPERTIES,
            const.ID_PROJECT_CLOSE,
            const.ID_REORIENT_IMG,
            const.ID_FLOODFILL_MASK,
            const.ID_FILL_HOLE_AUTO,
            const.ID_REMOVE_MASK_PART,
            const.ID_SELECT_MASK_PART,
            const.ID_FLOODFILL_SEGMENTATION,
            const.ID_FLIP_X,
            const.ID_FLIP_Y,
            const.ID_FLIP_Z,
            const.ID_SWAP_XY,
            const.ID_SWAP_XZ,
            const.ID_SWAP_YZ,
            const.ID_THRESHOLD_SEGMENTATION,
            const.ID_MANUAL_SEGMENTATION,
            const.ID_WATERSHED_SEGMENTATION,
            const.ID_THRESHOLD_SEGMENTATION,
            const.ID_FLOODFILL_SEGMENTATION,
            const.ID_SEGMENTATION_BRAIN,
            const.ID_SEGMENTATION_SUBPART,
            const.ID_SEGMENTATION_TRACHEA,
            const.ID_SEGMENTATION_MANDIBLE_CT,
            const.ID_PLANNING_CRANIOPLASTY,
            const.ID_MASK_DENSITY_MEASURE,
            const.ID_CREATE_SURFACE,
            const.ID_CREATE_MASK,
            const.ID_GOTO_SLICE,
            const.ID_MANUAL_WWWL,
        ]
        self.__init_items()
        self.__bind_events()
        self.SetStateProjectClose()

    def _add_action(self, menu, id, text, checkable=False, enabled=True, shortcut=None):
        action = QAction(text, self)
        action.setData(id)
        if checkable:
            action.setCheckable(True)
        action.setEnabled(enabled)
        if shortcut:
            action.setShortcut(shortcut)
        action.triggered.connect(lambda checked, a=action: self.parent.OnMenuClick(a))
        menu.addAction(action)
        self._action_map[id] = action
        return action

    def __bind_events(self):
        sub = Publisher.subscribe
        sub(self.OnEnableState, "Enable state project")
        sub(self.OnEnableUndo, "Enable undo")
        sub(self.OnEnableRedo, "Enable redo")
        sub(self.OnEnableGotoCoord, "Enable Go-to-Coord")
        sub(self.OnEnableNavigation, "Navigation status")

        sub(self.OnAddMask, "Add mask")
        sub(self.OnRemoveMasks, "Remove masks")
        sub(self.OnShowMask, "Show mask")
        sub(self.OnUpdateSliceInterpolation, "Update Slice Interpolation MenuBar")
        sub(self.OnUpdateNavigationMode, "Update Navigation Mode MenuBar")

        sub(self.AddPluginsItems, "Add plugins menu items")

        self.num_masks = 0

    def __init_items(self):
        others_file_menu = QMenu(_("Import other files..."), self)
        self._add_action(others_file_menu, const.ID_ANALYZE_IMPORT, _("Analyze 7.5"))
        self._add_action(others_file_menu, const.ID_NIFTI_IMPORT, _("NIfTI 1"))
        self._add_action(others_file_menu, const.ID_PARREC_IMPORT, _("PAR/REC"))
        self._add_action(
            others_file_menu,
            const.ID_TIFF_JPG_PNG,
            "TIFF,BMP,JPG or PNG (\xb5CT)",
        )

        file_menu = self.addMenu(_("File"))
        self._add_action(file_menu, const.ID_DICOM_IMPORT, _("Import DICOM..."), shortcut="Ctrl+I")
        file_menu.addMenu(others_file_menu)
        self._add_action(file_menu, const.ID_PROJECT_OPEN, _("Open project..."), shortcut="Ctrl+O")
        self._add_action(file_menu, const.ID_PROJECT_SAVE, _("Save project"), shortcut="Ctrl+S")
        self._add_action(
            file_menu,
            const.ID_PROJECT_SAVE_AS,
            _("Save project as..."),
            shortcut="Ctrl+Shift+S",
        )
        self._add_action(file_menu, const.ID_EXPORT_SLICE, _("Export project"))
        self._add_action(file_menu, const.ID_PROJECT_PROPERTIES, _("Project properties"))
        self._add_action(file_menu, const.ID_PROJECT_CLOSE, _("Close project"))
        file_menu.addSeparator()
        self._add_action(file_menu, const.ID_EXIT, _("Exit"), shortcut="Ctrl+Q")

        file_edit = self.addMenu(_("Edit"))
        self._undo_action = self._add_action(
            file_edit, const.ID_EDIT_UNDO, _("Undo"), enabled=False, shortcut="Ctrl+Z"
        )
        self._redo_action = self._add_action(
            file_edit, const.ID_EDIT_REDO, _("Redo"), enabled=False, shortcut="Ctrl+Y"
        )
        self._add_action(file_edit, const.ID_GOTO_SLICE, _("Go to slice ..."), shortcut="Ctrl+G")
        self._goto_coord_action = self._add_action(
            file_edit, const.ID_GOTO_COORD, _("Go to scanner coord ..."), enabled=False
        )

        # Tool menu
        tools_menu = QMenu(_("Tools"), self)

        # Mask Menu
        mask_menu = QMenu(_("Mask"), self)
        self.new_mask_action = self._add_action(
            mask_menu, const.ID_CREATE_MASK, _("New"), enabled=False, shortcut="Ctrl+Shift+M"
        )
        self.bool_op_action = self._add_action(
            mask_menu,
            const.ID_BOOLEAN_MASK,
            _("Boolean operations"),
            enabled=False,
            shortcut="Ctrl+Shift+B",
        )
        self.clean_mask_action = self._add_action(
            mask_menu, const.ID_CLEAN_MASK, _("Clean Mask"), enabled=False, shortcut="Ctrl+Shift+A"
        )
        mask_menu.addSeparator()
        self.fill_hole_mask_action = self._add_action(
            mask_menu,
            const.ID_FLOODFILL_MASK,
            _("Fill holes manually"),
            enabled=False,
            shortcut="Ctrl+Shift+H",
        )
        self.fill_hole_auto_action = self._add_action(
            mask_menu,
            const.ID_FILL_HOLE_AUTO,
            _("Fill holes automatically"),
            enabled=False,
            shortcut="Ctrl+Shift+J",
        )
        mask_menu.addSeparator()
        self.remove_mask_part_action = self._add_action(
            mask_menu,
            const.ID_REMOVE_MASK_PART,
            _("Remove parts"),
            enabled=False,
            shortcut="Ctrl+Shift+K",
        )
        self.select_mask_part_action = self._add_action(
            mask_menu,
            const.ID_SELECT_MASK_PART,
            _("Select parts"),
            enabled=False,
            shortcut="Ctrl+Shift+L",
        )
        mask_menu.addSeparator()
        self.crop_mask_action = self._add_action(
            mask_menu, const.ID_CROP_MASK, _("Crop"), enabled=False
        )
        mask_menu.addSeparator()

        mask_preview_menu = QMenu(_("Mask 3D Preview"), self)
        self.mask_preview_action = self._add_action(
            mask_preview_menu,
            const.ID_MASK_3D_PREVIEW,
            _("Enable"),
            checkable=True,
            enabled=False,
            shortcut="Ctrl+Shift+P",
        )

        session = ses.Session()
        auto_reload_preview = session.GetConfig("auto_reload_preview")

        self.mask_auto_reload_action = self._add_action(
            mask_preview_menu,
            const.ID_MASK_3D_AUTO_RELOAD,
            _("Auto reload"),
            checkable=True,
            enabled=False,
            shortcut="Ctrl+Shift+D",
        )
        self.mask_auto_reload_action.setChecked(bool(auto_reload_preview))

        self.mask_preview_reload_action = self._add_action(
            mask_preview_menu,
            const.ID_MASK_3D_RELOAD,
            _("Reload"),
            enabled=False,
            shortcut="Ctrl+Shift+R",
        )
        mask_menu.addMenu(mask_preview_menu)

        # Segmentation Menu
        segmentation_menu = QMenu(_("Segmentation"), self)
        self.threshold_segmentation_action = self._add_action(
            segmentation_menu,
            const.ID_THRESHOLD_SEGMENTATION,
            _("Threshold"),
            shortcut="Ctrl+Shift+T",
        )
        self.manual_segmentation_action = self._add_action(
            segmentation_menu,
            const.ID_MANUAL_SEGMENTATION,
            _("Manual segmentation"),
            shortcut="Ctrl+Shift+E",
        )
        self.watershed_segmentation_action = self._add_action(
            segmentation_menu,
            const.ID_WATERSHED_SEGMENTATION,
            _("Watershed"),
            shortcut="Ctrl+Shift+W",
        )
        self.ffill_segmentation_action = self._add_action(
            segmentation_menu,
            const.ID_FLOODFILL_SEGMENTATION,
            _("Region growing"),
            enabled=False,
            shortcut="Ctrl+Shift+G",
        )
        segmentation_menu.addSeparator()
        self._add_action(
            segmentation_menu,
            const.ID_SEGMENTATION_BRAIN,
            _("Brain segmentation (MRI T1)"),
        )
        self._add_action(
            segmentation_menu,
            const.ID_SEGMENTATION_SUBPART,
            _("Brain subpart segmentation (MRI T1)"),
        )
        self._add_action(
            segmentation_menu,
            const.ID_SEGMENTATION_TRACHEA,
            _("Trachea segmentation (CT)"),
        )
        self._add_action(
            segmentation_menu,
            const.ID_SEGMENTATION_MANDIBLE_CT,
            _("Mandible segmentation (CT)"),
        )

        # Surface Menu
        surface_menu = QMenu(_("Surface"), self)
        self.create_surface_action = self._add_action(
            surface_menu,
            const.ID_CREATE_SURFACE,
            "New",
            enabled=False,
            shortcut="Ctrl+Shift+C",
        )
        self.remove_non_visible_action = self._add_action(
            surface_menu,
            const.ID_REMOVE_NON_VISIBLE_FACES,
            _("Remove non-visible faces"),
        )

        # Image menu
        image_menu = QMenu(_("Image"), self)

        flip_menu = QMenu(_("Flip"), self)
        self._add_action(flip_menu, const.ID_FLIP_X, _("Right - Left"), enabled=False)
        self._add_action(flip_menu, const.ID_FLIP_Y, _("Anterior - Posterior"), enabled=False)
        self._add_action(flip_menu, const.ID_FLIP_Z, _("Top - Bottom"), enabled=False)

        swap_axes_menu = QMenu(_("Swap axes"), self)
        self._add_action(
            swap_axes_menu,
            const.ID_SWAP_XY,
            _("From Right-Left to Anterior-Posterior"),
            enabled=False,
        )
        self._add_action(
            swap_axes_menu,
            const.ID_SWAP_XZ,
            _("From Right-Left to Top-Bottom"),
            enabled=False,
        )
        self._add_action(
            swap_axes_menu,
            const.ID_SWAP_YZ,
            _("From Anterior-Posterior to Top-Bottom"),
            enabled=False,
        )

        image_menu.addMenu(flip_menu)
        image_menu.addMenu(swap_axes_menu)
        self.reorient_action = self._add_action(
            image_menu,
            const.ID_REORIENT_IMG,
            _("Reorient image"),
            enabled=False,
            shortcut="Ctrl+Shift+O",
        )
        self._add_action(image_menu, const.ID_MANUAL_WWWL, _("Set WW&&WL manually"))

        planning_menu = QMenu(_("Planning"), self)
        self._add_action(planning_menu, const.ID_PLANNING_CRANIOPLASTY, _("Cranioplasty"))

        analysis_menu = QMenu(_("Analysis"), self)
        self._add_action(analysis_menu, const.ID_MASK_DENSITY_MEASURE, _("Mask density measure"))

        tools_menu.addMenu(analysis_menu)
        tools_menu.addMenu(image_menu)
        tools_menu.addMenu(mask_menu)
        tools_menu.addMenu(planning_menu)
        tools_menu.addMenu(segmentation_menu)
        tools_menu.addMenu(surface_menu)

        tools_menu.addSeparator()
        self._add_action(tools_menu, ID_SHOW_LOG_VIEWER, _("Show Log Viewer"))
        self._add_action(tools_menu, ID_INTERACTIVE_SHELL, _("Interactive Shell"))

        self.tools_menu = tools_menu

        # View
        self.view_menu = view_menu = self.addMenu(_("View"))
        self._interpolated_action = self._add_action(
            view_menu,
            const.ID_VIEW_INTERPOLATED,
            _("Interpolated slices"),
            checkable=True,
        )
        v = self.SliceInterpolationStatus()
        self._interpolated_action.setChecked(bool(v))

        self.addMenu(tools_menu)

        plugins_menu = self.addMenu(_("Plugins"))
        self._add_action(plugins_menu, const.ID_PLUGINS_SHOW_PATH, _("Open Plugins folder"))
        self.plugins_menu = plugins_menu

        options_menu = self.addMenu(_("Options"))
        self._add_action(options_menu, const.ID_PREFERENCES, _("Preferences"))

        # Mode
        self.mode_menu = mode_menu = self.addMenu(_("Mode"))
        nav_menu = QMenu(_("Navigation Mode"), self)
        self._nav_action = self._add_action(
            nav_menu,
            const.ID_MODE_NAVIGATION,
            _("Transcranial Magnetic Stimulation Mode"),
            checkable=True,
            shortcut="Ctrl+T",
        )
        self.mode_dbs_action = self._add_action(
            nav_menu,
            const.ID_MODE_DBS,
            _("Deep Brain Stimulation Mode"),
            checkable=True,
            enabled=False,
            shortcut="Ctrl+B",
        )
        mode_menu.addMenu(nav_menu)

        v = self.NavigationModeStatus()
        self._nav_action.setChecked(bool(v))

        # Help
        help_menu = self.addMenu(_("Help"))
        self._add_action(help_menu, const.ID_START, _("Getting started..."))
        help_menu.addSeparator()
        self._add_action(help_menu, const.ID_ABOUT, _("About..."))

    def OnPluginMenu(self, action):
        id = action.data()
        if id is not None and id != const.ID_PLUGINS_SHOW_PATH:
            try:
                plugin_name = self._plugins_menu_ids[id]["name"]
                print("Loading plugin:", plugin_name)
                Publisher.sendMessage("Load plugin", plugin_name=plugin_name)
            except KeyError:
                print("Invalid plugin")

    def SliceInterpolationStatus(self):
        session = ses.Session()
        slice_interpolation = session.GetConfig("slice_interpolation")
        return slice_interpolation

    def NavigationModeStatus(self):
        session = ses.Session()
        mode = session.GetConfig("mode")
        return mode == 1

    def OnUpdateSliceInterpolation(self):
        v = self.SliceInterpolationStatus()
        action = self._action_map.get(const.ID_VIEW_INTERPOLATED)
        if action:
            action.setChecked(bool(v))

    def OnUpdateNavigationMode(self):
        v = self.NavigationModeStatus()
        action = self._action_map.get(const.ID_MODE_NAVIGATION)
        if action:
            action.setChecked(bool(v))

    def AddPluginsItems(self, items):
        for action in list(self.plugins_menu.actions()):
            if action.data() != const.ID_PLUGINS_SHOW_PATH:
                self.plugins_menu.removeAction(action)

        for item in items:
            _new_id = const._new_id_ref()
            self._plugins_menu_ids[_new_id] = items[item]
            act = self._add_action(
                self.plugins_menu,
                _new_id,
                item,
                enabled=items[item]["enable_startup"],
            )

    def OnEnableState(self, state):
        if state:
            self.SetStateProjectOpen()
        else:
            self.SetStateProjectClose()

    def SetStateProjectClose(self):
        for item_id in self.enable_items:
            action = self._action_map.get(item_id)
            if action:
                action.setEnabled(False)

        for item_id in self._plugins_menu_ids:
            if not self._plugins_menu_ids[item_id]["enable_startup"]:
                action = self._action_map.get(item_id)
                if action:
                    action.setEnabled(False)

    def SetStateProjectOpen(self):
        for item_id in self.enable_items:
            action = self._action_map.get(item_id)
            if action:
                action.setEnabled(True)

        for item_id in self._plugins_menu_ids:
            if not self._plugins_menu_ids[item_id]["enable_startup"]:
                action = self._action_map.get(item_id)
                if action:
                    action.setEnabled(True)

    def OnEnableUndo(self, value):
        action = self._action_map.get(const.ID_EDIT_UNDO)
        if action:
            action.setEnabled(bool(value))

    def OnEnableRedo(self, value):
        action = self._action_map.get(const.ID_EDIT_REDO)
        if action:
            action.setEnabled(bool(value))

    def OnEnableGotoCoord(self, status=True):
        action = self._action_map.get(const.ID_GOTO_COORD)
        if action:
            action.setEnabled(status)

    def OnEnableNavigation(self, nav_status, vis_status):
        action = self._action_map.get(const.ID_MODE_NAVIGATION)
        if action:
            action.setEnabled(not nav_status)

    def OnAddMask(self, mask):
        self.num_masks += 1
        self.bool_op_action.setEnabled(self.num_masks >= 2)
        self.mask_preview_action.setEnabled(True)
        self.mask_auto_reload_action.setEnabled(True)
        self.mask_preview_reload_action.setEnabled(True)

    def OnRemoveMasks(self, mask_indexes):
        self.num_masks -= len(mask_indexes)
        self.bool_op_action.setEnabled(self.num_masks >= 2)

    def OnShowMask(self, index, value):
        self.clean_mask_action.setEnabled(value)
        self.crop_mask_action.setEnabled(value)
        self.mask_preview_action.setEnabled(value)
        self.mask_auto_reload_action.setEnabled(value)
        self.mask_preview_reload_action.setEnabled(value)


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


class StatusBar(QStatusBar):
    """
    Control general status (both text and gauge)
    """

    def __init__(self, parent):
        super().__init__(parent)

        self.showMessage(_("Ready"))

        self.image_info_label = QLabel("")
        self.addPermanentWidget(self.image_info_label)

        self.__bind_events()

    def __bind_events(self):
        sub = Publisher.subscribe
        sub(self._SetProgressLabel, "Update status text in GUI")
        sub(self._SetImageInfo, "Update statusbar image info")
        sub(self._ClearImageInfo, "Clear statusbar image info")

    def _SetProgressLabel(self, label):
        self.showMessage(label)

    def _SetImageInfo(self, info):
        self.image_info_label.setText(info)

    def _ClearImageInfo(self):
        self.image_info_label.setText("")


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


class TaskBarIcon(QSystemTrayIcon):
    """
    TaskBarIcon has different behaviours according to the platform:
        - win32:  Show icon on "Notification Area" (near clock)
        - darwin: Show icon on Dock
        - linux2: Show icon on "Notification Area" (near clock)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame = parent

        icon = QIcon(str(os.path.join(inv_paths.ICON_DIR, "invesalius.ico")))
        self.setIcon(icon)
        self.setToolTip("InVesalius")

        self.activated.connect(self.OnTaskBarActivate)

    def OnTaskBarActivate(self, reason):
        pass


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


class ProjectToolBar(QToolBar):
    """
    Toolbar related to general project operations.
    """

    def __init__(self, parent):
        super().__init__("General Features Toolbar", parent)
        self.setIconSize(self.iconSize())
        self.setMovable(False)

        self.parent_frame = parent

        self.enable_items = [const.ID_PROJECT_SAVE]
        self._action_map = {}

        self.__init_items()
        self.__bind_events()
        self.SetStateProjectClose()

    def __bind_events(self):
        sub = Publisher.subscribe
        sub(self._EnableState, "Enable state project")

    def _add_tool(self, id, icon_path, tooltip, checkable=False):
        icon = QIcon(QPixmap(str(icon_path)))
        action = self.addAction(icon, "")
        action.setToolTip(tooltip)
        action.setData(id)
        action.setCheckable(checkable)
        action.triggered.connect(lambda checked, a=action: self.parent_frame.OnMenuClick(a))
        self._action_map[id] = action
        return action

    def __init_items(self):
        d = inv_paths.ICON_DIR

        self._add_tool(
            const.ID_DICOM_IMPORT,
            d.joinpath("file_import_original.png"),
            _("Import DICOM files...\tCtrl+I"),
        )
        self._add_tool(
            const.ID_PROJECT_OPEN,
            d.joinpath("file_open_original.png"),
            _("Open InVesalius project..."),
        )
        self._add_tool(
            const.ID_PROJECT_SAVE,
            d.joinpath("file_save_original.png"),
            _("Save InVesalius project"),
        )
        self._add_tool(
            const.ID_PREFERENCES_TOOLBAR,
            d.joinpath("preferences.png"),
            _("Preferences"),
        )

    def _EnableState(self, state):
        if state:
            self.SetStateProjectOpen()
        else:
            self.SetStateProjectClose()

    def SetStateProjectClose(self):
        for tool_id in self.enable_items:
            action = self._action_map.get(tool_id)
            if action:
                action.setEnabled(False)

    def SetStateProjectOpen(self):
        for tool_id in self.enable_items:
            action = self._action_map.get(tool_id)
            if action:
                action.setEnabled(True)


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


class ObjectToolBar(QToolBar):
    """
    Toolbar related to general object operations.
    """

    def __init__(self, parent):
        super().__init__("Object Toolbar", parent)
        self.setIconSize(self.iconSize())
        self.setMovable(False)

        self.parent_frame = parent
        self.enable_items = [
            const.STATE_WL,
            const.STATE_PAN,
            const.STATE_SPIN,
            const.STATE_ZOOM_SL,
            const.STATE_ZOOM,
            const.STATE_MEASURE_DISTANCE,
            const.STATE_MEASURE_ANGLE,
            const.STATE_MEASURE_DENSITY_ELLIPSE,
            const.STATE_MEASURE_DENSITY_POLYGON,
        ]
        self._action_map = {}
        self.__init_items()
        self.__bind_events()
        self.SetStateProjectClose()

    def __bind_events(self):
        sub = Publisher.subscribe
        sub(self._EnableState, "Enable state project")
        sub(self._UntoggleAllItems, "Untoggle object toolbar items")
        sub(self._ToggleLinearMeasure, "Set tool linear measure")
        sub(self._ToggleAngularMeasure, "Set tool angular measure")
        sub(self.ToggleItem, "Toggle toolbar item")

    def _add_tool(self, id, icon_path, tooltip, checkable=False):
        icon = QIcon(QPixmap(str(icon_path)))
        action = self.addAction(icon, "")
        action.setToolTip(tooltip)
        action.setData(id)
        action.setCheckable(checkable)
        action.triggered.connect(lambda checked, a=action: self._OnToggle(a))
        self._action_map[id] = action
        return action

    def __init_items(self):
        d = inv_paths.ICON_DIR

        self._add_tool(
            const.STATE_ZOOM,
            os.path.join(d, "tool_zoom_original.png"),
            _("Zoom"),
            checkable=True,
        )
        self._add_tool(
            const.STATE_ZOOM_SL,
            os.path.join(d, "tool_zoom_select_original.png"),
            _("Zoom based on selection"),
            checkable=True,
        )
        self._add_tool(
            const.STATE_SPIN,
            os.path.join(d, "tool_rotate_original.png"),
            _("Rotate"),
            checkable=True,
        )
        self._add_tool(
            const.STATE_PAN,
            os.path.join(d, "tool_translate_original.png"),
            _("Move"),
            checkable=True,
        )
        self._add_tool(
            const.STATE_WL,
            os.path.join(d, "tool_contrast_original.png"),
            _("Contrast"),
            checkable=True,
        )
        self._add_tool(
            const.STATE_MEASURE_DISTANCE,
            os.path.join(d, "measure_line_original.png"),
            _("Measure distance"),
            checkable=True,
        )
        self._add_tool(
            const.STATE_MEASURE_ANGLE,
            os.path.join(d, "measure_angle_original.png"),
            _("Measure angle"),
            checkable=True,
        )
        self._add_tool(
            const.STATE_MEASURE_DENSITY_ELLIPSE,
            os.path.join(d, "measure_density_ellipse32px.png"),
            _("Measure density ellipse"),
            checkable=True,
        )
        self._add_tool(
            const.STATE_MEASURE_DENSITY_POLYGON,
            os.path.join(d, "measure_density_polygon32px.png"),
            _("Measure density polygon"),
            checkable=True,
        )

    def _EnableState(self, state):
        if state:
            self.SetStateProjectOpen()
        else:
            self.SetStateProjectClose()

    def _UntoggleAllItems(self):
        for id in const.TOOL_STATES:
            action = self._action_map.get(id)
            if action and action.isChecked():
                action.setChecked(False)

    def _ToggleLinearMeasure(self):
        id = const.STATE_MEASURE_DISTANCE
        action = self._action_map.get(id)
        if action:
            action.setChecked(True)
        Publisher.sendMessage("Enable style", style=id)
        Publisher.sendMessage("Untoggle slice toolbar items")
        for item in const.TOOL_STATES:
            a = self._action_map.get(item)
            if a and a.isChecked() and item != id:
                a.setChecked(False)

    def _ToggleAngularMeasure(self):
        id = const.STATE_MEASURE_ANGLE
        action = self._action_map.get(id)
        if action:
            action.setChecked(True)
        Publisher.sendMessage("Enable style", style=id)
        Publisher.sendMessage("Untoggle slice toolbar items")
        for item in const.TOOL_STATES:
            a = self._action_map.get(item)
            if a and a.isChecked() and item != id:
                a.setChecked(False)

    def _OnToggle(self, action):
        id = action.data()
        state = action.isChecked()

        if state and (id == const.STATE_MEASURE_DISTANCE or id == const.STATE_MEASURE_ANGLE):
            Publisher.sendMessage("Fold measure task")

        if state:
            Publisher.sendMessage("Enable style", style=id)
            Publisher.sendMessage("Untoggle slice toolbar items")
        else:
            Publisher.sendMessage("Disable style", style=id)

        for item in const.TOOL_STATES:
            a = self._action_map.get(item)
            if a and a.isChecked() and item != id:
                a.setChecked(False)

    def ToggleItem(self, _id, value):
        if _id in self.enable_items:
            action = self._action_map.get(_id)
            if action:
                action.setChecked(value)

    def SetStateProjectClose(self):
        for tool_id in self.enable_items:
            action = self._action_map.get(tool_id)
            if action:
                action.setEnabled(False)
        self._UntoggleAllItems()

    def SetStateProjectOpen(self):
        for tool_id in self.enable_items:
            action = self._action_map.get(tool_id)
            if action:
                action.setEnabled(True)


# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------


class SliceToolBar(QToolBar):
    """
    Toolbar related to 2D slice specific operations.
    """

    def __init__(self, parent):
        super().__init__("Slice Toolbar", parent)
        self.setIconSize(self.iconSize())
        self.setMovable(False)

        self.parent_frame = parent
        self.enable_items = [
            const.SLICE_STATE_SCROLL,
            const.SLICE_STATE_CROSS,
        ]
        self._action_map = {}
        self.__init_items()
        self.__bind_events()
        self.SetStateProjectClose()

    def __init_items(self):
        d = inv_paths.ICON_DIR

        icon_slice = QIcon(QPixmap(str(os.path.join(d, "slice_original.png"))))
        self.sst = self.addAction(icon_slice, "")
        self.sst.setToolTip(_("Scroll slices"))
        self.sst.setData(const.SLICE_STATE_SCROLL)
        self.sst.setCheckable(True)
        self.sst.triggered.connect(lambda checked, a=self.sst: self._OnToggle(a))
        self._action_map[const.SLICE_STATE_SCROLL] = self.sst

        icon_cross = QIcon(QPixmap(str(os.path.join(d, "cross_original.png"))))
        self.sct = self.addAction(icon_cross, "")
        self.sct.setToolTip(_("Slices' cross intersection"))
        self.sct.setData(const.SLICE_STATE_CROSS)
        self.sct.setCheckable(True)
        self.sct.triggered.connect(lambda checked, a=self.sct: self._OnToggle(a))
        self._action_map[const.SLICE_STATE_CROSS] = self.sct

    def __bind_events(self):
        sub = Publisher.subscribe
        sub(self._EnableState, "Enable state project")
        sub(self._UntoggleAllItems, "Untoggle slice toolbar items")
        sub(self.OnToggle, "Toggle toolbar button")
        sub(self.ToggleItem, "Toggle toolbar item")

    def _EnableState(self, state):
        if state:
            self.SetStateProjectOpen()
        else:
            self.SetStateProjectClose()
            self._UntoggleAllItems()

    def _UntoggleAllItems(self):
        for id in const.TOOL_SLICE_STATES:
            action = self._action_map.get(id)
            if action and action.isChecked():
                action.setChecked(False)
                if id == const.SLICE_STATE_CROSS:
                    Publisher.sendMessage("Disable style", style=const.SLICE_STATE_CROSS)

    def _OnToggle(self, action):
        id = action.data()
        state = action.isChecked()

        if state:
            Publisher.sendMessage("Enable style", style=id)
            Publisher.sendMessage("Untoggle object toolbar items")
        else:
            Publisher.sendMessage("Disable style", style=id)

        if id == const.SLICE_STATE_CROSS and not state:
            Publisher.sendMessage("Stop image registration")

        for item_id in self.enable_items:
            a = self._action_map.get(item_id)
            if a and a.isChecked() and item_id != id:
                a.setChecked(False)

    def OnToggle(self, evt=None, id=None):
        if id is not None:
            action = self._action_map.get(id)
            if action and not action.isChecked():
                action.setChecked(True)
            if action:
                self._OnToggle(action)

    def ToggleItem(self, _id, value):
        if _id in self.enable_items:
            action = self._action_map.get(_id)
            if action:
                action.setChecked(value)

    def SetStateProjectClose(self):
        for tool_id in self.enable_items:
            action = self._action_map.get(tool_id)
            if action:
                action.setEnabled(False)

    def SetStateProjectOpen(self):
        for tool_id in self.enable_items:
            action = self._action_map.get(tool_id)
            if action:
                action.setEnabled(True)


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------


class LayoutToolBar(QToolBar):
    """
    Toolbar related to general layout: show/hide task panel, text, rulers.
    """

    def __init__(self, parent):
        super().__init__("Layout Toolbar", parent)
        self.setIconSize(self.iconSize())
        self.setMovable(False)

        self.parent_frame = parent
        self.enable_items = [ID_LAYOUT, ID_TEXT, ID_RULER]
        self._action_map = {}

        self.ontool_layout = False
        self.ontool_text = True
        self.ontool_ruler = True

        self.__init_items()
        self.__bind_events()
        self.SetStateProjectClose()

    def __bind_events(self):
        sub = Publisher.subscribe
        sub(self._EnableState, "Enable state project")
        sub(self._SetLayoutWithTask, "Set layout button data only")
        sub(self._SetLayoutWithoutTask, "Set layout button full")
        sub(self._SendRulerVisibilityStatus, "Send ruler visibility status")

    def __init_items(self):
        d = inv_paths.ICON_DIR

        p = os.path.join(d, "layout_data_only_original.png")
        self.ICON_WITH_MENU = QIcon(QPixmap(str(p)))

        p = os.path.join(d, "layout_full_original.png")
        self.ICON_WITHOUT_MENU = QIcon(QPixmap(str(p)))

        p = os.path.join(d, "text_inverted_original.png")
        self.ICON_WITHOUT_TEXT = QIcon(QPixmap(str(p)))

        p = os.path.join(d, "text_original.png")
        self.ICON_WITH_TEXT = QIcon(QPixmap(str(p)))

        p = os.path.join(d, "ruler_original_disabled.png")
        self.ICON_WITHOUT_RULER = QIcon(QPixmap(str(p)))

        p = os.path.join(d, "ruler_original_enabled.png")
        self.ICON_WITH_RULER = QIcon(QPixmap(str(p)))

        self._layout_action = self.addAction(self.ICON_WITHOUT_MENU, "")
        self._layout_action.setToolTip(_("Hide task panel"))
        self._layout_action.setData(ID_LAYOUT)
        self._layout_action.triggered.connect(lambda: self.ToggleLayout())
        self._action_map[ID_LAYOUT] = self._layout_action

        self._text_action = self.addAction(self.ICON_WITH_TEXT, "")
        self._text_action.setToolTip(_("Hide text"))
        self._text_action.setData(ID_TEXT)
        self._text_action.triggered.connect(lambda: self.ToggleText())
        self._action_map[ID_TEXT] = self._text_action

        self._ruler_action = self.addAction(self.ICON_WITH_RULER, "")
        self._ruler_action.setToolTip(_("Hide ruler"))
        self._ruler_action.setData(ID_RULER)
        self._ruler_action.triggered.connect(lambda: self.ToggleRulers())
        self._action_map[ID_RULER] = self._ruler_action

    def _EnableState(self, state):
        if state:
            self.SetStateProjectOpen()
        else:
            self.SetStateProjectClose()

    def _SendRulerVisibilityStatus(self):
        Publisher.sendMessage("Receive ruler visibility status", status=self.ontool_ruler)

    def _SetLayoutWithoutTask(self):
        self._layout_action.setIcon(self.ICON_WITHOUT_MENU)

    def _SetLayoutWithTask(self):
        self._layout_action.setIcon(self.ICON_WITH_MENU)

    def ToggleLayout(self):
        if self.ontool_layout:
            self._layout_action.setIcon(self.ICON_WITHOUT_MENU)
            self.parent_frame._ShowTask()
            self._layout_action.setToolTip(_("Hide task panel"))
            self.ontool_layout = False
        else:
            self._layout_action.setIcon(self.ICON_WITH_MENU)
            self.parent_frame._HideTask()
            self._layout_action.setToolTip(_("Show task panel"))
            self.ontool_layout = True

    def ToggleText(self):
        if self.ontool_text:
            self._text_action.setIcon(self.ICON_WITH_TEXT)
            Publisher.sendMessage("Hide text actors on viewers")
            self._text_action.setToolTip(_("Show text"))
            Publisher.sendMessage("Update AUI")
            self.ontool_text = False
        else:
            self._text_action.setIcon(self.ICON_WITHOUT_TEXT)
            Publisher.sendMessage("Show text actors on viewers")
            self._text_action.setToolTip(_("Hide text"))
            Publisher.sendMessage("Update AUI")
            self.ontool_text = True

    def ShowRulers(self):
        self._ruler_action.setIcon(self.ICON_WITH_RULER)
        Publisher.sendMessage("Show rulers on viewers")
        self._ruler_action.setToolTip(_("Hide rulers"))
        Publisher.sendMessage("Update AUI")
        self.ontool_ruler = True

    def HideRulers(self):
        self._ruler_action.setIcon(self.ICON_WITHOUT_RULER)
        Publisher.sendMessage("Hide rulers on viewers")
        self._ruler_action.setToolTip(_("Show rulers"))
        Publisher.sendMessage("Update AUI")
        self.ontool_ruler = False

    def ToggleRulers(self):
        if self.ontool_ruler:
            self.HideRulers()
        else:
            self.ShowRulers()

    def SetStateProjectClose(self):
        self.ontool_text = True
        self.ontool_ruler = True
        self.ToggleText()
        self.HideRulers()
        for tool_id in self.enable_items:
            action = self._action_map.get(tool_id)
            if action:
                action.setEnabled(False)

    def SetStateProjectOpen(self):
        self.ontool_text = False
        self.ontool_ruler = True
        self.ToggleText()
        self.HideRulers()
        for tool_id in self.enable_items:
            action = self._action_map.get(tool_id)
            if action:
                action.setEnabled(True)


# --------------------------------------------------------------------
# --------------------------------------------------------------------
# --------------------------------------------------------------------


class HistoryToolBar(QToolBar):
    """
    Toolbar related to general undo and redo operations.
    """

    def __init__(self, parent):
        super().__init__("History Toolbar", parent)
        self.setIconSize(self.iconSize())
        self.setMovable(False)

        self.parent_frame = parent
        self._action_map = {}
        self.__init_items()
        self.__bind_events()

    def __bind_events(self):
        sub = Publisher.subscribe
        sub(self.OnEnableUndo, "Enable undo")
        sub(self.OnEnableRedo, "Enable redo")

    def __init_items(self):
        d = inv_paths.ICON_DIR

        icon_undo = QIcon(QPixmap(str(os.path.join(d, "undo_original.png"))))
        self._undo_action = self.addAction(icon_undo, "")
        self._undo_action.setToolTip(_("Undo"))
        self._undo_action.setEnabled(False)
        self._undo_action.triggered.connect(self.OnUndo)

        icon_redo = QIcon(QPixmap(str(os.path.join(d, "redo_original.png"))))
        self._redo_action = self.addAction(icon_redo, "")
        self._redo_action.setToolTip(_("Redo"))
        self._redo_action.setEnabled(False)
        self._redo_action.triggered.connect(self.OnRedo)

    def OnUndo(self):
        Publisher.sendMessage("Undo edition")

    def OnRedo(self):
        Publisher.sendMessage("Redo edition")

    def OnEnableUndo(self, value):
        self._undo_action.setEnabled(bool(value))

    def OnEnableRedo(self, value):
        self._redo_action.setEnabled(bool(value))
