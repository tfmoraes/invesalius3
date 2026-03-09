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
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.gui.dicom_preview_panel as dpp
import invesalius.reader.dicom_grouper as dcm
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher


class Panel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        sizer = QVBoxLayout(self)
        sizer.setContentsMargins(5, 5, 5, 5)
        sizer.addWidget(InnerPanel(self), 1)


class InnerPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.patients = []
        self.first_image_selection = None
        self.last_image_selection = None
        self._init_ui()
        self._bind_events()
        self._bind_pubsubevt()

    def _init_ui(self):
        self.splitter = QSplitter(Qt.Vertical, self)

        self.text_panel = TextPanel(self.splitter)
        self.image_panel = ImagePanel(self.splitter)

        self.splitter.addWidget(self.text_panel)
        self.splitter.addWidget(self.image_panel)
        self.splitter.setSizes([250, 250])

        panel = QWidget(self)
        self.btn_cancel = QPushButton(_("Cancel"), panel)
        self.btn_ok = QPushButton(_("Import"), panel)

        self.combo_interval = QComboBox(panel)
        self.combo_interval.addItems(const.IMPORT_INTERVAL)
        self.combo_interval.setCurrentIndex(0)

        inner_sizer = QHBoxLayout(panel)
        inner_sizer.setContentsMargins(5, 5, 5, 5)
        inner_sizer.addWidget(self.btn_ok)
        inner_sizer.addWidget(self.btn_cancel)
        inner_sizer.addWidget(self.combo_interval)

        sizer = QVBoxLayout(self)
        sizer.addWidget(self.splitter, 20)
        sizer.addWidget(panel, 0)

    def _bind_pubsubevt(self):
        Publisher.subscribe(self.ShowDicomPreview, "Load import panel")
        Publisher.subscribe(self.GetSelectedImages, "Selected Import Images")

    def GetSelectedImages(self, selection):
        self.first_image_selection = selection[0]
        self.last_image_selection = selection[1]

    def _bind_events(self):
        self.text_panel.serie_selected.connect(self.OnSelectSerie)
        self.text_panel.serie_double_clicked.connect(self.OnDblClickTextPanel)
        self.btn_ok.clicked.connect(self.OnClickOk)
        self.btn_cancel.clicked.connect(self.OnClickCancel)

    def ShowDicomPreview(self, dicom_groups):
        self.patients.extend(dicom_groups)
        self.text_panel.Populate(dicom_groups)

    def OnSelectSerie(self, patient_id, serie_number):
        self.text_panel.SelectSerie((patient_id, serie_number))
        for patient in self.patients:
            if patient_id == patient.GetDicomSample().patient.id:
                for group in patient.GetGroups():
                    if serie_number == group.GetDicomSample().acquisition.serie_number:
                        self.image_panel.SetSerie(group)

    def OnDblClickTextPanel(self, group):
        if group:
            self.LoadDicom(group)

    def OnClickOk(self):
        group = self.text_panel.GetSelection()
        if group:
            self.LoadDicom(group)

    def OnClickCancel(self):
        Publisher.sendMessage("Cancel DICOM load")

    def LoadDicom(self, group):
        if not group:
            return

        interval = self.combo_interval.currentIndex()
        if not isinstance(group, dcm.DicomGroup):
            if hasattr(group, "GetGroups") and callable(group.GetGroups):
                groups = group.GetGroups()
                if groups:
                    group = max(groups, key=lambda g: g.nslices)
                else:
                    return
            else:
                return

        slice_amont = group.nslices
        if (self.first_image_selection is not None) and (
            self.first_image_selection != self.last_image_selection
        ):
            slice_amont = (self.last_image_selection) - self.first_image_selection
            slice_amont += 1
            if slice_amont == 0:
                slice_amont = group.nslices

        Publisher.sendMessage(
            "Open DICOM group",
            group=group,
            interval=interval,
            file_range=(self.first_image_selection, self.last_image_selection),
        )


class TextPanel(QWidget):
    serie_selected = Signal(object, object)
    serie_double_clicked = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)

        self._selected_by_user = True
        self.idserie_treeitem = {}
        self.treeitem_idpatient = {}

        self.__init_gui()
        self.__bind_events_wx()
        self.__bind_pubsub_evt()

    def __bind_pubsub_evt(self):
        Publisher.subscribe(self.SelectSeries, "Select series in import panel")

    def __bind_events_wx(self):
        pass

    def __init_gui(self):
        self.tree = QTreeWidget(self)
        self.tree.setHeaderLabels(
            [
                _("Patient name"),
                _("Patient ID"),
                _("Age"),
                _("Gender"),
                _("Study description"),
                _("Modality"),
                _("Date acquired"),
                _("# Images"),
                _("Institution"),
                _("Date of birth"),
                _("Accession Number"),
                _("Referring physician"),
            ]
        )
        self.tree.setColumnWidth(0, 280)
        self.tree.setColumnWidth(1, 110)
        self.tree.setColumnWidth(2, 40)
        self.tree.setColumnWidth(3, 60)
        self.tree.setColumnWidth(4, 160)
        self.tree.setColumnWidth(5, 70)
        self.tree.setColumnWidth(6, 200)
        self.tree.setColumnWidth(7, 70)
        self.tree.setColumnWidth(8, 130)
        self.tree.setColumnWidth(9, 100)
        self.tree.setColumnWidth(10, 140)
        self.tree.setColumnWidth(11, 160)
        self.tree.setRootIsDecorated(True)
        self.tree.setSelectionMode(QTreeWidget.SingleSelection)

        sizer = QVBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(self.tree)

    def SelectSeries(self, group_index):
        pass

    def Populate(self, patient_list):
        tree = self.tree
        parent_select = None

        first = 0
        for patient in patient_list:
            if not isinstance(patient, dcm.PatientGroup):
                return None
            ngroups = patient.ngroups
            dicom = patient.GetDicomSample()
            title = dicom.patient.name + " (%d series)" % (ngroups)
            date_time = f"{dicom.acquisition.date} {dicom.acquisition.time}"

            parent = QTreeWidgetItem(tree)
            parent.setText(0, title)
            parent.setData(0, Qt.UserRole, patient)

            if not first:
                parent_select = parent
                first += 1

            parent.setText(1, f"{dicom.patient.id}")
            parent.setText(2, f"{dicom.patient.age}")
            parent.setText(3, f"{dicom.patient.gender}")
            parent.setText(4, f"{dicom.acquisition.study_description}")
            parent.setText(5, f"{dicom.acquisition.modality}")
            parent.setText(6, f"{date_time}")
            parent.setText(7, f"{patient.nslices}")
            parent.setText(8, f"{dicom.acquisition.institution}")
            parent.setText(9, f"{dicom.patient.birthdate}")
            parent.setText(10, f"{dicom.acquisition.accession_number}")
            parent.setText(11, f"{dicom.patient.physician}")

            group_list = patient.GetGroups()
            for n, group in enumerate(group_list):
                dicom = group.GetDicomSample()

                child = QTreeWidgetItem(parent)
                child.setText(0, f"{group.title}")
                child.setData(0, Qt.UserRole, group)
                child.setText(4, f"{dicom.acquisition.protocol_name}")
                child.setText(5, f"{dicom.acquisition.modality}")
                child.setText(6, f"{date_time}")
                child.setText(7, f"{group.nslices}")

                self.idserie_treeitem[(dicom.patient.id, dicom.acquisition.serie_number)] = child

        if parent_select:
            tree.expandItem(parent_select)
            tree.setCurrentItem(parent_select)

        tree.itemActivated.connect(self.OnActivate)
        tree.currentItemChanged.connect(self.OnSelChanged)

    def OnSelChanged(self, current, previous):
        if current is None:
            return
        if self._selected_by_user:
            group = current.data(0, Qt.UserRole)
            if isinstance(group, dcm.DicomGroup):
                Publisher.sendMessage("Load group into import panel", group=group)
            elif isinstance(group, dcm.PatientGroup):
                Publisher.sendMessage("Load patient into import panel", patient=group)
        else:
            parent_item = current.parent()
            if parent_item:
                parent_item.setExpanded(True)

    def OnActivate(self, item, column):
        group = item.data(0, Qt.UserRole)
        self.serie_double_clicked.emit(group)

    def SelectSerie(self, serie):
        self._selected_by_user = False
        item = self.idserie_treeitem.get(serie)
        if item:
            self.tree.setCurrentItem(item)
        self._selected_by_user = True

    def GetSelection(self):
        """Get selected item"""
        items = self.tree.selectedItems()
        if items:
            return items[0].data(0, Qt.UserRole)
        return None


class ImagePanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._init_ui()
        self._bind_events()

    def _init_ui(self):
        self.splitter = QSplitter(Qt.Horizontal, self)

        self.text_panel = SeriesPanel(self.splitter)
        self.image_panel = SlicePanel(self.splitter)

        self.splitter.addWidget(self.text_panel)
        self.splitter.addWidget(self.image_panel)
        self.splitter.setSizes([600, 250])

        sizer = QHBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(self.splitter, 1)

    def _bind_events(self):
        self.text_panel.serie_selected.connect(self._on_select_serie)
        self.text_panel.slice_selected.connect(self._on_select_slice)

    def _on_select_serie(self, select_id, item_data):
        pass

    def _on_select_slice(self, select_id, item_data):
        self.image_panel.dicom_preview.ShowSlice(select_id)

    def SetSerie(self, serie):
        self.image_panel.dicom_preview.SetDicomGroup(serie)


class SeriesPanel(QWidget):
    serie_selected = Signal(object, object)
    slice_selected = Signal(object, object)

    def __init__(self, parent):
        super().__init__(parent)

        self.serie_preview = dpp.DicomPreviewSeries(self)
        self.dicom_preview = dpp.DicomPreviewSlice(self)
        self.dicom_preview.hide()

        self.sizer = QHBoxLayout(self)
        self.sizer.setContentsMargins(5, 5, 5, 5)
        self.sizer.addWidget(self.serie_preview, 1)
        self.sizer.addWidget(self.dicom_preview, 1)

        self.__bind_evt()
        self._bind_gui_evt()

    def __bind_evt(self):
        Publisher.subscribe(self.ShowDicomSeries, "Load dicom preview")
        Publisher.subscribe(self.SetDicomSeries, "Load group into import panel")
        Publisher.subscribe(self.SetPatientSeries, "Load patient into import panel")

    def _bind_gui_evt(self):
        self.serie_preview.serie_clicked.connect(self._on_serie_click)
        self.dicom_preview.slice_clicked.connect(self._on_slice_click)

    def SetDicomSeries(self, group):
        self.dicom_preview.SetDicomGroup(group)
        self.dicom_preview.show()
        self.serie_preview.hide()

    def GetSelectedImagesRange(self):
        return [self.dicom_preview.first_selected, self.dicom_preview_last_selection]

    def SetPatientSeries(self, patient):
        self.dicom_preview.hide()
        self.serie_preview.show()

        self.serie_preview.SetPatientGroups(patient)
        self.dicom_preview.SetPatientGroups(patient)

    def _on_serie_click(self, select_id, item_data):
        self.serie_selected.emit(select_id, item_data)

    def _on_slice_click(self, select_id, item_data):
        self.slice_selected.emit(select_id, item_data)

    def ShowDicomSeries(self, patient):
        if isinstance(patient, dcm.PatientGroup):
            self.serie_preview.SetPatientGroups(patient)
            self.dicom_preview.SetPatientGroups(patient)


class SlicePanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.__init_gui()
        self.__bind_evt()

    def __bind_evt(self):
        Publisher.subscribe(self.ShowDicomSeries, "Load dicom preview")
        Publisher.subscribe(self.SetDicomSeries, "Load group into import panel")
        Publisher.subscribe(self.SetPatientSeries, "Load patient into import panel")

    def __init_gui(self):
        pal = self.palette()
        from PySide6.QtGui import QColor, QPalette

        pal.setColor(QPalette.Window, QColor(255, 255, 255))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

        self.dicom_preview = dpp.SingleImagePreview(self)

        sizer = QVBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(self.dicom_preview, 1)

    def SetPatientSeries(self, patient):
        group = patient.GetGroups()[0]
        self.dicom_preview.SetDicomGroup(group)

    def SetDicomSeries(self, group):
        self.dicom_preview.SetDicomGroup(group)

    def ShowDicomSeries(self, patient):
        group = patient.GetGroups()[0]
        self.dicom_preview.SetDicomGroup(group)
