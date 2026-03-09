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
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.gui.bitmap_preview_panel as bpp
import invesalius.gui.dialogs as dlg
import invesalius.reader.bitmap_reader as bpr
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
        Publisher.subscribe(self.ShowBitmapPreview, "Load import bitmap panel")
        Publisher.subscribe(self.GetSelectedImages, "Selected Import Images")

    def ShowBitmapPreview(self, data):
        self.text_panel.Populate(data)

    def GetSelectedImages(self, selection):
        self.first_image_selection = selection[0]
        self.last_image_selection = selection[1]

    def _bind_events(self):
        self.btn_ok.clicked.connect(self.OnClickOk)
        self.btn_cancel.clicked.connect(self.OnClickCancel)

    def OnClickOk(self):
        parm = dlg.ImportBitmapParameters()
        parm.SetInterval(self.combo_interval.currentIndex())
        parm.ShowModal()

    def OnClickCancel(self):
        Publisher.sendMessage("Cancel DICOM load")


class TextPanel(QWidget):
    serie_double_clicked = Signal(object)

    def __init__(self, parent):
        super().__init__(parent)

        self.parent_widget = parent

        self._selected_by_user = True
        self.idserie_treeitem = {}
        self.treeitem_idpatient = {}

        self.selected_items = None

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
                _("Path"),
                _("Type"),
                _("Width x Height"),
            ]
        )
        self.tree.setColumnWidth(0, 880)
        self.tree.setColumnWidth(1, 60)
        self.tree.setColumnWidth(2, 130)
        self.tree.setRootIsDecorated(False)
        self.tree.setSelectionMode(QTreeWidget.ExtendedSelection)

        sizer = QVBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(self.tree)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Delete:
            selected_items = self.tree.selectedItems()
            for selected_item in selected_items:
                text_item = selected_item.text(0)
                index = bpr.BitmapData().GetIndexByPath(text_item)
                bpr.BitmapData().RemoveFileByPath(text_item)
                data_size = len(bpr.BitmapData().GetData())

                if index >= 0 and index < data_size:
                    Publisher.sendMessage("Set bitmap in preview panel", pos=index)
                elif index == data_size and data_size > 0:
                    Publisher.sendMessage("Set bitmap in preview panel", pos=index - 1)
                elif data_size == 1:
                    Publisher.sendMessage("Set bitmap in preview panel", pos=0)
                else:
                    Publisher.sendMessage("Show black slice in single preview image")

                idx = self.tree.indexOfTopLevelItem(selected_item)
                if idx >= 0:
                    self.tree.takeTopLevelItem(idx)
                Publisher.sendMessage("Remove preview panel", data=text_item)
        super().keyPressEvent(event)

    def SelectSeries(self, group_index):
        pass

    def Populate(self, data):
        tree = self.tree
        for value in data:
            item = QTreeWidgetItem(tree)
            item.setText(0, value[0])
            item.setText(1, value[2])
            item.setText(2, value[5])

        tree.itemActivated.connect(self.OnActivate)
        tree.currentItemChanged.connect(self.OnSelChanged)

        Publisher.sendMessage("Load bitmap into import panel", data=data)

    def OnSelChanged(self, current, previous):
        if current is None:
            return
        if self._selected_by_user:
            text_item = current.text(0)
            index = bpr.BitmapData().GetIndexByPath(text_item)
            Publisher.sendMessage("Set bitmap in preview panel", pos=index)

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
        self.text_panel.slice_selected.connect(self._on_select_slice)

    def _on_select_slice(self, select_id, item_data):
        self.image_panel.bitmap_preview.ShowSlice(select_id)

    def SetSerie(self, serie):
        self.image_panel.bitmap_preview.SetDicomGroup(serie)


class SeriesPanel(QWidget):
    slice_selected = Signal(object, object)

    def __init__(self, parent):
        super().__init__(parent)

        self.thumbnail_preview = bpp.BitmapPreviewSeries(self)

        self.sizer = QHBoxLayout(self)
        self.sizer.setContentsMargins(5, 5, 5, 5)
        self.sizer.addWidget(self.thumbnail_preview, 1)

        self.__bind_evt()
        self._bind_gui_evt()

    def __bind_evt(self):
        Publisher.subscribe(self.SetBitmapFiles, "Load bitmap into import panel")

    def _bind_gui_evt(self):
        self.thumbnail_preview.serie_clicked.connect(self._on_serie_click)

    def GetSelectedImagesRange(self):
        return [self.bitmap_preview.first_selected, self.dicom_preview_last_selection]

    def SetBitmapFiles(self, data):
        bitmap = data
        self.thumbnail_preview.show()
        self.thumbnail_preview.SetBitmapFiles(bitmap)

    def _on_serie_click(self, select_id, item_data):
        self.slice_selected.emit(select_id, item_data)


class SlicePanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.__init_gui()
        self.__bind_evt()

    def __bind_evt(self):
        Publisher.subscribe(self.SetBitmapFiles, "Load bitmap into import panel")

    def __init_gui(self):
        from PySide6.QtGui import QColor, QPalette

        pal = self.palette()
        pal.setColor(QPalette.Window, QColor(255, 255, 255))
        self.setPalette(pal)
        self.setAutoFillBackground(True)

        self.bitmap_preview = bpp.SingleImagePreview(self)

        sizer = QVBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(self.bitmap_preview, 1)

    def SetBitmapFiles(self, data):
        self.bitmap_preview.SetBitmapFiles(data)
