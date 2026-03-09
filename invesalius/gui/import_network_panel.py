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
import sys

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.gui.dialogs as dlg
import invesalius.net.dicom as dcm_net
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

        self.image_panel = HostFindPanel(self.splitter)
        self.text_panel = TextPanel(self.splitter)

        self.splitter.addWidget(self.image_panel)
        self.splitter.addWidget(self.text_panel)
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
        sizer.addWidget(panel, 1)

    def _bind_pubsubevt(self):
        pass

    def GetSelectedImages(self, pubsub_evt):
        self.first_image_selection = pubsub_evt.data[0]
        self.last_image_selection = pubsub_evt.data[1]

    def _bind_events(self):
        self.text_panel.serie_selected.connect(self.OnSelectSerie)
        self.text_panel.serie_double_clicked.connect(self.OnDblClickTextPanel)
        self.btn_ok.clicked.connect(self.OnClickOk)
        self.btn_cancel.clicked.connect(self.OnClickCancel)

    def ShowDicomPreview(self, pubsub_evt):
        dicom_groups = pubsub_evt.data
        self.patients.extend(dicom_groups)
        self.text_panel.Populate(dicom_groups)

    def OnSelectSerie(self, patient_id, serie_number):
        self.text_panel.SelectSerie((patient_id, serie_number))
        for patient in self.patients:
            if patient_id == patient.GetDicomSample().patient.id:
                for group in patient.GetGroups():
                    if serie_number == group.GetDicomSample().acquisition.serie_number:
                        self.image_panel.SetSerie(group)

    def OnSelectSlice(self):
        pass

    def OnSelectPatient(self):
        pass

    def OnDblClickTextPanel(self, group):
        self.LoadDicom(group)

    def OnClickOk(self):
        group = self.text_panel.GetSelection()
        if group:
            self.LoadDicom(group)

    def OnClickCancel(self):
        pass

    def LoadDicom(self, group):
        interval = self.combo_interval.currentIndex()

        if not isinstance(group, dcm.DicomGroup):
            group = max(group.GetGroups(), key=lambda g: g.nslices)

        slice_amont = group.nslices
        if (self.first_image_selection is not None) and (
            self.first_image_selection != self.last_image_selection
        ):
            slice_amont = (self.last_image_selection) - self.first_image_selection
            slice_amont += 1
            if slice_amont == 0:
                slice_amont = group.nslices

        nslices_result = slice_amont / (interval + 1)
        if nslices_result > 1:
            pass
        else:
            dlg.MissingFilesForReconstruction()


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
        Publisher.subscribe(self.Populate, "Populate tree")
        Publisher.subscribe(self.SetHostsList, "Set FindPanel hosts list")

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

    def Populate(self, pubsub_evt):
        tree = self.tree
        patients = pubsub_evt.data

        self.idserie_treeitem = {}

        for patient in patients.keys():
            first_serie = patients[patient].keys()[0]
            title = patients[patient][first_serie]["name"]
            p = patients[patient][first_serie]

            p_id = patient
            age = p["age"]
            gender = p["gender"]
            study_description = p["study_description"]
            modality = p["modality"]
            date = p["acquisition_date"]
            time_ = p["acquisition_time"]
            institution = p["institution"]
            birthdate = p["date_of_birth"]
            acession_number = p["acession_number"]
            physician = p["ref_physician"]

            parent = QTreeWidgetItem(tree)
            parent.setText(0, title)

            n_amount_images = 0
            for se in patients[patient]:
                n_amount_images = n_amount_images + patients[patient][se]["n_images"]

            parent.setData(0, Qt.UserRole, patient)
            parent.setText(1, f"{p_id}")
            parent.setText(2, f"{age}")
            parent.setText(3, f"{gender}")
            parent.setText(4, f"{study_description}")
            parent.setText(5, "")
            parent.setText(6, f"{date} {time_}")
            parent.setText(7, f"{str(n_amount_images)}")
            parent.setText(8, f"{institution}")
            parent.setText(9, f"{birthdate}")
            parent.setText(10, f"{acession_number}")
            parent.setText(11, f"{physician}")

            for series in patients[patient].keys():
                serie_description = patients[patient][series]["serie_description"]
                n_images = patients[patient][series]["n_images"]
                date = patients[patient][series]["acquisition_date"]
                time_ = patients[patient][series]["acquisition_time"]
                modality = patients[patient][series]["modality"]

                child = QTreeWidgetItem(parent)
                child.setText(0, f"{serie_description}")
                child.setData(0, Qt.UserRole, series)
                child.setText(5, f"{modality}")
                child.setText(6, f"{date} {time_}")
                child.setText(7, f"{n_images}")

                self.idserie_treeitem[(patient, series)] = child

        tree.expandAll()
        tree.itemActivated.connect(self.OnActivate)

    def SetHostsList(self, evt_pub):
        self.hosts = evt_pub.data

    def GetHostList(self):
        Publisher.sendMessage("Get NodesPanel host list")
        return self.hosts

    def OnSelChanged(self, current, previous):
        if current is None:
            return
        if self._selected_by_user:
            group = current.data(0, Qt.UserRole)
            if isinstance(group, dcm.DicomGroup):
                pass
            elif isinstance(group, dcm.PatientGroup):
                id_ = group.GetDicomSample().patient.id
                self.serie_selected.emit(id_, None)
        else:
            parent_item = current.parent()
            if parent_item:
                parent_item.setExpanded(True)

    def OnActivate(self, item, column):
        parent_item = item.parent()
        if parent_item is None:
            return

        patient_id = parent_item.data(0, Qt.UserRole)
        serie_id = item.data(0, Qt.UserRole)

        hosts = self.GetHostList()

        for key in hosts.keys():
            if key != 0:
                dn = dcm_net.DicomNet()
                dn.SetHost(self.hosts[key][1])
                dn.SetPort(self.hosts[key][2])
                dn.SetAETitleCall(self.hosts[key][3])
                dn.SetAETitle(self.hosts[0][3])
                dn.RunCMove((patient_id, serie_id))

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


class FindPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.sizer = QVBoxLayout(self)

        find_label = QLabel(_("Word"), self)

        sizer_word_label = QHBoxLayout()
        sizer_word_label.addSpacing(5)
        sizer_word_label.addWidget(find_label)

        self.find_txt = QLineEdit(self)
        self.find_txt.setMinimumWidth(225)
        self.btn_find = QPushButton(_("Search"), self)

        sizer_txt_find = QHBoxLayout()
        sizer_txt_find.addSpacing(5)
        sizer_txt_find.addWidget(self.find_txt)
        sizer_txt_find.addWidget(self.btn_find)

        self.sizer.addSpacing(5)
        self.sizer.addLayout(sizer_word_label)
        self.sizer.addLayout(sizer_txt_find)

        self.__bind_evt()
        self._bind_gui_evt()

    def __bind_evt(self):
        Publisher.subscribe(self.SetHostsList, "Set FindPanel hosts list")

    def _bind_gui_evt(self):
        self.btn_find.clicked.connect(self.OnButtonFind)

    def OnButtonFind(self):
        hosts = self.GetHostList()

        for key in hosts.keys():
            if key != 0:
                dn = dcm_net.DicomNet()
                dn.SetHost(self.hosts[key][1])
                dn.SetPort(self.hosts[key][2])
                dn.SetAETitleCall(self.hosts[key][3])
                dn.SetAETitle(self.hosts[0][3])
                dn.SetSearchWord(self.find_txt.text())

                Publisher.sendMessage("Populate tree", dn.RunCFind())

    def SetHostsList(self, evt_pub):
        self.hosts = evt_pub.data

    def GetHostList(self):
        Publisher.sendMessage("Get NodesPanel host list")
        return self.hosts


class HostFindPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        self.splitter = QSplitter(Qt.Horizontal, self)

        self.image_panel = NodesPanel(self.splitter)
        self.text_panel = FindPanel(self.splitter)

        self.splitter.addWidget(self.image_panel)
        self.splitter.addWidget(self.text_panel)
        self.splitter.setSizes([500, 750])

        sizer = QHBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(self.splitter, 1)

    def SetSerie(self, serie):
        pass


class NodesPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.selected_item = None
        self.hosts = {}
        self.__init_gui()
        self.__bind_evt()

    def __bind_evt(self):
        self.tree_node.itemChanged.connect(self.OnItemChanged)
        self.tree_node.itemSelectionChanged.connect(self.OnItemSelectionChanged)
        self.btn_add.clicked.connect(self.OnButtonAdd)
        self.btn_remove.clicked.connect(self.OnButtonRemove)
        self.btn_check.clicked.connect(self.OnButtonCheck)

        Publisher.subscribe(self.GetHostsList, "Get NodesPanel host list")

    def __init_gui(self):
        self.tree_node = QTableWidget(0, 5, self)
        self.tree_node.setHorizontalHeaderLabels(
            [
                _("Active"),
                _("Host"),
                _("Port"),
                _("AETitle"),
                _("Status"),
            ]
        )
        self.tree_node.setColumnWidth(0, 50)
        self.tree_node.setColumnWidth(1, 150)
        self.tree_node.setColumnWidth(2, 50)
        self.tree_node.setColumnWidth(3, 150)
        self.tree_node.setColumnWidth(4, 80)
        self.tree_node.setSelectionBehavior(QTableWidget.SelectRows)
        self.tree_node.setSelectionMode(QTableWidget.SingleSelection)

        self.hosts[0] = [True, "localhost", "", "invesalius"]
        row = self.tree_node.rowCount()
        self.tree_node.insertRow(row)

        check_item = QTableWidgetItem()
        check_item.setFlags(check_item.flags() | Qt.ItemIsUserCheckable)
        check_item.setCheckState(Qt.Checked)
        self.tree_node.setItem(row, 0, check_item)
        self.tree_node.setItem(row, 1, QTableWidgetItem("localhost"))
        self.tree_node.setItem(row, 2, QTableWidgetItem(""))
        self.tree_node.setItem(row, 3, QTableWidgetItem("invesalius"))
        self.tree_node.setItem(row, 4, QTableWidgetItem("ok"))
        self.tree_node.item(row, 0).setBackground(QColor(245, 245, 245))

        self.btn_add = QPushButton(_("Add"), self)
        self.btn_remove = QPushButton(_("Remove"), self)
        self.btn_check = QPushButton(_("Check status"), self)

        sizer_btn = QHBoxLayout()
        sizer_btn.addSpacing(90)
        sizer_btn.addWidget(self.btn_add, 10)
        sizer_btn.addWidget(self.btn_remove, 10)
        sizer_btn.addWidget(self.btn_check, 0)

        sizer = QVBoxLayout(self)
        sizer.addWidget(self.tree_node, 85)
        sizer.addLayout(sizer_btn, 15)

    def GetHostsList(self, pub_evt):
        Publisher.sendMessage("Set FindPanel hosts list", self.hosts)

    def OnItemChanged(self, item):
        row = item.row()
        col = item.column()
        if col == 0:
            flag = item.checkState() == Qt.Checked
            if row != 0:
                self.hosts[row][0] = flag
            else:
                item.setCheckState(Qt.Checked)
        elif 1 <= col <= 3:
            txt = item.text()
            if row in self.hosts:
                self.hosts[row][col] = str(txt)

    def OnItemSelectionChanged(self):
        selected = self.tree_node.selectedItems()
        if selected:
            self.selected_item = selected[0].row()
        else:
            self.selected_item = None

    def OnButtonAdd(self):
        row = self.tree_node.rowCount()
        self.tree_node.insertRow(row)

        self.hosts[row] = [True, "localhost", "80", ""]

        check_item = QTableWidgetItem()
        check_item.setFlags(check_item.flags() | Qt.ItemIsUserCheckable)
        check_item.setCheckState(Qt.Checked)
        self.tree_node.setItem(row, 0, check_item)
        self.tree_node.setItem(row, 1, QTableWidgetItem("localhost"))
        self.tree_node.setItem(row, 2, QTableWidgetItem("80"))
        self.tree_node.setItem(row, 3, QTableWidgetItem(""))

    def OnButtonRemove(self):
        if self.selected_item is not None and self.selected_item != 0:
            self.tree_node.removeRow(self.selected_item)
            self.hosts.pop(self.selected_item, None)
            self.selected_item = None

            k = list(self.hosts.keys())
            tmp_cont = 0
            tmp_host = {}
            for x in k:
                tmp_host[tmp_cont] = self.hosts[x]
                tmp_cont += 1
            self.hosts = tmp_host

    def OnButtonCheck(self):
        for key in self.hosts.keys():
            if key != 0:
                dn = dcm_net.DicomNet()
                dn.SetHost(self.hosts[key][1])
                dn.SetPort(self.hosts[key][2])
                dn.SetAETitleCall(self.hosts[key][3])
                dn.SetAETitle(self.hosts[0][3])

                status_item = self.tree_node.item(key, 4)
                if status_item is None:
                    status_item = QTableWidgetItem()
                    self.tree_node.setItem(key, 4, status_item)

                if dn.RunCEcho():
                    status_item.setText(_("ok"))
                else:
                    status_item.setText(_("error"))
