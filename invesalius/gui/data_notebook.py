# -*- coding: UTF-8 -*-
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

try:
    import Image
except ImportError:
    from PIL import Image

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QCursor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.data.slice_ as slice_
import invesalius.gui.dialogs as dlg
from invesalius import inv_paths, project
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher

TYPE = {
    const.LINEAR: _("Linear"),
    const.ANGULAR: _("Angular"),
    const.DENSITY_ELLIPSE: _("Density Ellipse"),
    const.DENSITY_POLYGON: _("Density Polygon"),
}

LOCATION = {
    const.SURFACE: _("3D"),
    const.AXIAL: _("Axial"),
    const.CORONAL: _("Coronal"),
    const.SAGITAL: _("Sagittal"),
}


class NotebookPanel(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        book = QTabWidget(self)

        book.addTab(MaskPage(book), _("Masks"))
        book.addTab(SurfacePage(book), _("3D surfaces"))
        book.addTab(MeasurePage(book), _("Measures"))
        # book.addTab(AnnotationsListCtrlPanel(book), _("Notes"))

        book.setCurrentIndex(0)

        sizer = QVBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(book)

        self.book = book

        self.__bind_events()

    def __bind_events(self):
        Publisher.subscribe(self._FoldSurface, "Fold surface task")
        Publisher.subscribe(self._FoldSurface, "Fold surface page")
        Publisher.subscribe(self._FoldMeasure, "Fold measure task")
        Publisher.subscribe(self._FoldMask, "Fold mask task")
        Publisher.subscribe(self._FoldMask, "Fold mask page")

    def _FoldSurface(self):
        """
        Fold surface notebook page.
        """
        self.book.setCurrentIndex(1)

    def _FoldMeasure(self):
        """
        Fold measure notebook page.
        """
        self.book.setCurrentIndex(2)

    def _FoldMask(self):
        """
        Fold mask notebook page.
        """
        self.book.setCurrentIndex(0)


class MeasurePage(QWidget):
    """
    Page related to mask items.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.__init_gui()

    def __init_gui(self):
        self.listctrl = MeasuresListCtrlPanel(self)
        self.buttonctrl = MeasureButtonControlPanel(self)

        sizer = QVBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(self.listctrl, 1)
        sizer.addWidget(self.buttonctrl, 0)


class MeasureButtonControlPanel(QWidget):
    """
    Button control panel that includes data notebook operations.
    TODO: Enhace interface with parent class - it is really messed up
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.__init_gui()

    def __init_gui(self):
        icon_new = QIcon(os.path.join(inv_paths.ICON_DIR, "data_new.png"))
        icon_remove = QIcon(os.path.join(inv_paths.ICON_DIR, "data_remove.png"))
        icon_duplicate = QIcon(os.path.join(inv_paths.ICON_DIR, "data_duplicate.png"))

        button_new = QPushButton(icon_new, "", self)
        button_new.setFixedSize(QSize(24, 20))
        button_new.setToolTip(_("Create a new measure"))
        button_new.clicked.connect(self.OnNew)

        button_remove = QPushButton(icon_remove, "", self)
        button_remove.setFixedSize(QSize(24, 20))
        button_remove.setToolTip(_("Remove measure"))
        button_remove.clicked.connect(self.OnRemove)

        button_duplicate = QPushButton(icon_duplicate, "", self)
        button_duplicate.setFixedSize(QSize(24, 20))
        button_duplicate.setToolTip(_("Duplicate measure"))
        button_duplicate.setEnabled(False)
        button_duplicate.clicked.connect(self.OnDuplicate)

        sizer = QHBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(button_new)
        sizer.addWidget(button_remove)
        sizer.addWidget(button_duplicate)
        sizer.addStretch()

        self.menu = QMenu(self)
        action_linear = self.menu.addAction(_("Measure distance"))
        action_angular = self.menu.addAction(_("Measure angle"))
        action_linear.triggered.connect(lambda: Publisher.sendMessage("Set tool linear measure"))
        action_angular.triggered.connect(lambda: Publisher.sendMessage("Set tool angular measure"))

    def OnNew(self):
        self.menu.exec(QCursor.pos())

    def OnMenu(self, evt):
        id = evt.GetId()
        if id == const.MEASURE_LINEAR:
            Publisher.sendMessage("Set tool linear measure")
        else:
            Publisher.sendMessage("Set tool angular measure")

    def OnRemove(self):
        self.parent.listctrl.RemoveMeasurements()

    def OnDuplicate(self):
        selected_items = self.parent.listctrl.GetSelected()
        if selected_items:
            Publisher.sendMessage("Duplicate measurement", selected_items)
        else:
            dlg.MaskSelectionRequiredForDuplication()


class MaskPage(QWidget):
    """
    Page related to mask items.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.categories = {}
        self.sizer = QVBoxLayout(self)
        self.sizer.setContentsMargins(0, 0, 0, 0)
        self.__init_gui()

        Publisher.subscribe(self.AddMask, "Add mask")
        Publisher.subscribe(self.RefreshMasks, "Refresh Masks")
        Publisher.subscribe(self.OnCloseProject, "Close project data")
        Publisher.subscribe(self.EditMaskThreshold, "Set mask threshold in notebook")
        Publisher.subscribe(self.EditMaskColour, "Change mask colour in notebook")
        Publisher.subscribe(self.OnChangeCurrentMask, "Change mask selected")
        Publisher.subscribe(self.hide_current_mask, "Hide current mask")
        Publisher.subscribe(self.show_current_mask, "Show current mask")
        Publisher.subscribe(self.update_current_colour, "Set GUI items colour")
        Publisher.subscribe(self.update_selection_state, "Update mask selection state")

    def __init_gui(self):
        self.buttonctrl = ButtonControlPanel(self)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_widget)

        self.sizer.addWidget(self.scroll_area, 1)
        self.sizer.addWidget(self.buttonctrl, 0)

        self.create_category("General")

    def create_category_header(self, parent, category):
        """Create header panel with category controls"""
        header_panel = QWidget(parent)
        header_layout = QHBoxLayout(header_panel)
        header_layout.setContentsMargins(2, 0, 0, 0)

        expand_btn = QPushButton("\u25bc", header_panel)
        expand_btn.setFixedSize(20, 20)
        expand_btn.clicked.connect(
            lambda checked, cat=category: self.toggle_category_expansion(cat)
        )

        category_label = QLabel(category, header_panel)
        font = category_label.font()
        font.setBold(True)
        category_label.setFont(font)

        header_layout.addWidget(expand_btn)
        header_layout.addWidget(category_label, 1)

        return header_panel, expand_btn

    def create_category(self, category):
        header_panel, expand_btn = self.create_category_header(self.scroll_widget, category)

        content_panel = QWidget(self.scroll_widget)
        listctrl = MasksListCtrlPanel(content_panel)
        listctrl.category = category
        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(listctrl)

        self.categories[category] = {
            "header": header_panel,
            "content": content_panel,
            "expand_btn": expand_btn,
            "list": listctrl,
            "expanded": True,
        }

        stretch_item = self.scroll_layout.takeAt(self.scroll_layout.count() - 1)
        self.scroll_layout.addWidget(header_panel)
        self.scroll_layout.addWidget(content_panel)
        self.scroll_layout.addStretch()

        return listctrl

    def toggle_category_expansion(self, category):
        if category not in self.categories:
            return

        category_info = self.categories[category]
        content_panel = category_info["content"]
        expand_btn = category_info["expand_btn"]
        is_expanded = category_info["expanded"]

        if is_expanded:
            content_panel.hide()
            expand_btn.setText("\u25b6")
            self.categories[category]["expanded"] = False
        else:
            content_panel.show()
            expand_btn.setText("\u25bc")
            self.categories[category]["expanded"] = True

        self.update_scroll_layout()

    def update_selection_state(self, category=None):
        """Limit selection to a single category and notify other components."""
        if not category or category not in self.categories:
            Publisher.sendMessage("Update selected masks list", indices=[])
            Publisher.sendMessage("Select all masks changed", select_all_active=False)
            return

        for cat, info in self.categories.items():
            if cat != category:
                lst = info["list"]
                if hasattr(lst, "ClearSelection"):
                    lst.ClearSelection()
        listctrl = self.categories[category]["list"]
        selected_indices = list(listctrl.GetSelected())
        Publisher.sendMessage("Update selected masks list", indices=selected_indices)
        is_batch_mode = len(selected_indices) > 1
        Publisher.sendMessage("Select all masks changed", select_all_active=is_batch_mode)

    def AddMask(self, mask):
        category = getattr(mask, "category", "General")
        if category not in self.categories:
            self.create_category(category)

        self.categories[category]["list"].AddMask(mask)
        self.update_scroll_layout()

        self.update_selection_state(category)

    def RefreshMasks(self, clear_project=False):
        """Destroy all components and clear sizer"""
        old_widget = self.scroll_area.takeWidget()
        if old_widget:
            old_widget.deleteLater()

        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_widget)

        self.categories.clear()

        self.create_category("General")

        if not clear_project:
            mask_dict = project.Project().mask_dict
            for i in sorted(mask_dict.keys()):
                mask = mask_dict[i]
                self.AddMask(mask)
        self.update_scroll_layout()

    def OnPaneChanged(self, evt):
        self.update_scroll_layout()

    def update_scroll_layout(self):
        """Update scroll panel layout"""
        self.scroll_widget.updateGeometry()
        self.scroll_widget.adjustSize()

    def update_current_colour(self, colour):
        """Handle updating the current mask colour in the respective category list"""
        for category_info in self.categories.values():
            listctrl = category_info["list"]
            if hasattr(listctrl, "current_index") and listctrl.current_index >= 0:
                listctrl.update_current_colour(colour)

    def hide_current_mask(self):
        """Handle hiding the current mask in the respective category list"""
        for category_info in self.categories.values():
            listctrl = category_info["list"]
            if hasattr(listctrl, "current_index") and listctrl.current_index >= 0:
                listctrl.hide_current_mask()

    def show_current_mask(self):
        """Handle showing the current mask in the respective category list"""
        for category_info in self.categories.values():
            listctrl = category_info["list"]
            if hasattr(listctrl, "current_index") and listctrl.current_index >= 0:
                listctrl.show_current_mask()

    def OnChangeCurrentMask(self, index):
        """Handle mask selection change in the appropriate category list"""
        selected_listctrl = None
        local_idx_to_select = -1
        selected_category = None

        for category, category_info in self.categories.items():
            listctrl = category_info["list"]
            if index in listctrl.mask_list_index:
                selected_listctrl = listctrl
                selected_category = category
                local_idx_to_select = listctrl.mask_list_index[index]
                break

        if selected_listctrl and local_idx_to_select != -1:
            for category_info in self.categories.values():
                listctrl = category_info["list"]
                if listctrl is not selected_listctrl:
                    for local_idx in listctrl.mask_list_index.values():
                        listctrl.SetItemImage(local_idx, 0)

            selected_listctrl.OnChangeCurrentMask(local_idx_to_select)

            if selected_category:
                self.update_selection_state(selected_category)

    def EditMaskThreshold(self, index, threshold_range):
        """Edit mask threshold in the appropriate category list"""
        for category_info in self.categories.values():
            listctrl = category_info["list"]
            if index in listctrl.mask_list_index:
                listctrl.EditMaskThreshold(index, threshold_range)
                return

    def EditMaskColour(self, index, colour):
        """Edit mask colour in the appropriate category list"""
        for category_info in self.categories.values():
            listctrl = category_info["list"]
            if index in listctrl.mask_list_index:
                listctrl.EditMaskColour(index, colour)
                return

    def OnCloseProject(self):
        QTimer.singleShot(0, lambda: self.RefreshMasks(True))


class ButtonControlPanel(QWidget):
    """
    Button control panel that includes data notebook operations.
    TODO: Enhace interface with parent class - it is really messed up
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.__init_gui()

    def __init_gui(self):
        icon_new = QIcon(os.path.join(inv_paths.ICON_DIR, "data_new.png"))
        icon_remove = QIcon(os.path.join(inv_paths.ICON_DIR, "data_remove.png"))
        icon_duplicate = QIcon(os.path.join(inv_paths.ICON_DIR, "data_duplicate.png"))
        icon_import = QIcon(os.path.join(inv_paths.ICON_DIR, "object_add.png"))
        icon_export = QIcon(os.path.join(inv_paths.ICON_DIR, "surface_export.png"))

        button_new = QPushButton(icon_new, "", self)
        button_new.setFixedSize(QSize(24, 20))
        button_new.setToolTip(_("Create a new mask"))
        button_new.clicked.connect(self.OnNew)

        button_remove = QPushButton(icon_remove, "", self)
        button_remove.setFixedSize(QSize(24, 20))
        button_remove.setToolTip(_("Remove mask"))
        button_remove.clicked.connect(self.OnRemove)

        button_duplicate = QPushButton(icon_duplicate, "", self)
        button_duplicate.setFixedSize(QSize(24, 20))
        button_duplicate.setToolTip(_("Duplicate mask"))
        button_duplicate.clicked.connect(self.OnDuplicate)

        button_import = QPushButton(icon_import, "", self)
        button_import.setFixedSize(QSize(24, 20))
        button_import.setToolTip(_("Import mask"))
        button_import.clicked.connect(self.OnImportMask)

        button_export = QPushButton(icon_export, "", self)
        button_export.setFixedSize(QSize(24, 20))
        button_export.setIconSize(QSize(22, 22))
        button_export.setToolTip(_("Export mask"))
        button_export.clicked.connect(self.OnExportMask)

        sizer = QHBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(button_new)
        sizer.addWidget(button_remove)
        sizer.addWidget(button_duplicate)
        sizer.addWidget(button_import)
        sizer.addWidget(button_export)
        sizer.addStretch()

    def OnNew(self):
        dialog = dlg.NewMask()

        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                ok = 1
            else:
                ok = 0
        except Exception:
            ok = 1

        if ok:
            mask_name, thresh, colour = dialog.GetValue()
            if mask_name:
                Publisher.sendMessage(
                    "Create new mask",
                    mask_name=mask_name,
                    thresh=thresh,
                    colour=colour,
                )

    def OnImportMask(self):
        Publisher.sendMessage("Show import mask dialog")

    def OnExportMask(self):
        all_selected_indices = []
        for category_info in self.parent.categories.values():
            listctrl = category_info["list"]
            selected = listctrl.GetSelected()
            all_selected_indices.extend(selected)

        if all_selected_indices:
            Publisher.sendMessage("Show export mask dialog", mask_indexes=all_selected_indices)
        else:
            dlg.MaskSelectionRequiredForDuplication()

    def OnRemove(self):
        all_selected_indices = []
        categories_snapshot = list(self.parent.categories.values())
        selections = []
        for category_info in categories_snapshot:
            listctrl = category_info["list"]
            selected = list(listctrl.GetSelected())
            all_selected_indices.extend(selected)
            if selected:
                selections.append((listctrl, selected))
        for listctrl, selected in selections:
            listctrl.RemoveMasks(sorted(set(selected), reverse=True))

    def OnDuplicate(self):
        all_selected_indices = []
        for category_info in self.parent.categories.values():
            listctrl = category_info["list"]
            selected = listctrl.GetSelected()
            all_selected_indices.extend(selected)

        if all_selected_indices:
            Publisher.sendMessage("Duplicate masks", mask_indexes=all_selected_indices)
        else:
            dlg.MaskSelectionRequiredForDuplication()


class InvListCtrl(QTreeWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setRootIsDecorated(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAllColumnsShowFocus(True)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

        self._programmatic_update = False
        self.icon_invisible = None
        self.icon_visible = None
        self.image_gray = None

        self.itemClicked.connect(self._on_item_clicked)
        self.itemDoubleClicked.connect(self._on_item_double_clicked)

    def _load_icons(self):
        """Load visibility icons and grayscale template."""
        self.icon_invisible = QIcon(os.path.join(inv_paths.ICON_DIR, "object_invisible.png"))
        self.icon_visible = QIcon(os.path.join(inv_paths.ICON_DIR, "object_visible.png"))
        self.image_gray = Image.open(os.path.join(inv_paths.ICON_DIR, "object_colour.png"))

    def CreateColourBitmap(self, colour):
        """
        Create a QIcon with a mask colour.
        colour: colour in rgb format(0 - 1)
        """
        image = self.image_gray
        new_image = Image.new("RGB", image.size)
        for x in range(image.size[0]):
            for y in range(image.size[1]):
                pixel_colour = [int(i * image.getpixel((x, y))) for i in colour]
                new_image.putpixel((x, y), tuple(pixel_colour))

        new_image = new_image.resize((16, 16), Image.LANCZOS)
        data = new_image.tobytes()
        qimage = QImage(data, 16, 16, 16 * 3, QImage.Format.Format_RGB888)
        return QIcon(QPixmap.fromImage(qimage.copy()))

    def _on_item_clicked(self, item, column):
        index = self.indexOfTopLevelItem(item)
        if index < 0:
            return
        if column == 0:
            visible = not bool(item.data(0, Qt.ItemDataRole.UserRole))
            self.SetItemImage(index, int(visible))
            self.OnCheckItem(index, visible)
        elif column == 1:
            self.OnChangeColor(index)
        elif column == 5:
            self.OnChangeTransparency(index)

    def _on_item_double_clicked(self, item, column):
        if column == 2:
            self.editItem(item, 2)

    def OnChangeColor(self, item_idx):
        pass

    def OnChangeTransparency(self, item_idx):
        pass

    def OnCheckItem(self, index, flag):
        pass

    def SetItemImage(self, index, flag):
        """Set the visibility icon on column 0."""
        item = self.topLevelItem(index)
        if item:
            item.setIcon(0, self.icon_visible if flag else self.icon_invisible)
            item.setData(0, Qt.ItemDataRole.UserRole, bool(flag))

    def GetItemImage(self, index):
        """Get visibility state: 1=visible, 0=invisible."""
        item = self.topLevelItem(index)
        if item:
            return 1 if item.data(0, Qt.ItemDataRole.UserRole) else 0
        return 0


class MasksListCtrlPanel(InvListCtrl):
    def __init__(self, parent, **kwargs):
        super().__init__(parent)
        self._click_check = False
        self.mask_list_index = {}
        self.current_index = 0
        self.current_color = [255, 255, 255]
        self.__init_columns()
        self.__init_image_list()
        self.__bind_events_wx()
        self.__bind_events()

    def __bind_events_wx(self):
        self.itemChanged.connect(self._on_item_changed)
        self.itemSelectionChanged.connect(self.on_selection_changed)

    def __bind_events(self):
        Publisher.subscribe(self.OnCloseProject, "Close project data")

    def keyPressEvent(self, event):
        key = event.key()
        if (sys.platform == "darwin") and (key == Qt.Key.Key_Backspace):
            self.RemoveMasks()
        elif key == Qt.Key.Key_Delete:
            self.RemoveMasks()
        else:
            super().keyPressEvent(event)

    def _on_item_changed(self, item, column):
        if self._programmatic_update:
            return
        if column == 2:
            index = self.indexOfTopLevelItem(item)
            Publisher.sendMessage("Change mask name", index=index, name=item.text(2))

    def on_selection_changed(self):
        """Handle selection changes in the mask list"""
        if hasattr(self, "category"):
            Publisher.sendMessage("Update mask selection state", category=self.category)
        else:
            print("Selection changed but 'category' attribute not found on self.")

    def on_mouse_right_click(self, pos):
        item = self.itemAt(pos)
        if not item:
            return
        focused_item_idx = self.indexOfTopLevelItem(item)

        menu = QMenu(self)

        action_colour = menu.addAction(_("Change color"))
        action_colour.triggered.connect(lambda: self.change_mask_color(None))

        action_duplicate = menu.addAction(_("Duplicate"))
        action_duplicate.triggered.connect(lambda: self.duplicate_masks(None))

        menu.addSeparator()

        action_delete = menu.addAction(_("Delete"))
        action_delete.triggered.connect(lambda: self.delete_mask(None))

        menu.addSeparator()

        action_export = menu.addAction(_("Export as NIfTI"))
        action_export.triggered.connect(lambda: self.export_mask_nifti(None))

        Publisher.sendMessage("Change mask selected", index=focused_item_idx)
        Publisher.sendMessage("Show mask", index=focused_item_idx, value=True)

        menu.exec(self.viewport().mapToGlobal(pos))

    def update_current_colour(self, colour):
        self.current_colour = colour

    def OnChangeColor(self, item_idx):
        """Open color picker for the clicked mask"""
        global_mask_id = None
        for mask_id, local_pos in self.mask_list_index.items():
            if local_pos == item_idx:
                global_mask_id = mask_id
                break

        if global_mask_id is None:
            return

        Publisher.sendMessage("Change mask selected", index=global_mask_id)
        self.change_mask_color(None)

    def change_mask_color(self, event):
        current_color = self.current_color

        new_color = dlg.ShowColorDialog(color_current=current_color)

        if not new_color:
            return

        Publisher.sendMessage("Change mask colour", colour=new_color)

    def duplicate_masks(self, event):
        selected_items = self.GetSelected()
        if selected_items:
            Publisher.sendMessage("Duplicate masks", mask_indexes=selected_items)
        else:
            dlg.MaskSelectionRequiredForDuplication()

    def delete_mask(self, event):
        result = dlg.ShowConfirmationDialog(msg=_("Delete mask?"))
        if result != QDialog.DialogCode.Accepted:
            return
        self.RemoveMasks()

    def export_mask_nifti(self, event):
        selected_items = self.GetSelected()
        if selected_items:
            Publisher.sendMessage("Export masks to nifti", mask_indexes=selected_items)

    def RemoveMasks(self, selected_items=None):
        """
        Remove selected items.
        """
        if not selected_items:
            selected_items = self.GetSelected()

        if selected_items:
            Publisher.sendMessage("Remove masks", mask_indexes=selected_items)
            QTimer.singleShot(0, lambda: Publisher.sendMessage("Refresh Masks"))
        else:
            dlg.MaskSelectionRequiredForRemoval()

    def OnCloseProject(self):
        self.clear()
        self.mask_list_index = {}

    def OnChangeCurrentMask(self, index):
        try:
            self.SetItemImage(index, 1)
            self.current_index = index
        except Exception:
            pass
        for local_idx in self.mask_list_index.values():
            if local_idx != index:
                self.SetItemImage(local_idx, 0)

    def hide_current_mask(self):
        if self.mask_list_index:
            self.SetItemImage(self.current_index, 0)

    def show_current_mask(self):
        if self.mask_list_index:
            self.SetItemImage(self.current_index, 1)

    def __init_columns(self):
        self.setHeaderLabels(["", "", _("Name"), _("Threshold")])
        self.setColumnWidth(0, 25)
        self.setColumnWidth(1, 25)
        self.setColumnWidth(2, 95)
        self.setColumnWidth(3, 90)
        self.setToolTip(_("Change mask color"))

    def __init_image_list(self):
        self._load_icons()

    def ClearSelection(self):
        """Unselect all items in this list control."""
        self.clearSelection()

    def OnCheckItem(self, index, flag):
        global_idx = -1
        for g_id, l_id in self.mask_list_index.items():
            if l_id == index:
                global_idx = g_id
                break

        if global_idx == -1:
            print(f" OnCheckItem: global_idx not found for local index {index}")
            return

        print(f" OnCheckItem: global_idx = {global_idx}")

        if flag:
            Publisher.sendMessage("Change mask selected", index=global_idx)
            self.current_index = index

        Publisher.sendMessage("Show mask", index=global_idx, value=flag)

        self.on_selection_changed()

    def InsertNewItem(self, index=0, label=_("Mask"), threshold="(1000, 4500)", colour=None):
        colour_icon = self.CreateColourBitmap(colour)

        item = QTreeWidgetItem()
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        item.setIcon(0, self.icon_invisible)
        item.setData(0, Qt.ItemDataRole.UserRole, False)
        item.setIcon(1, colour_icon)
        item.setText(2, label)
        item.setText(3, threshold)
        item.setTextAlignment(3, Qt.AlignmentFlag.AlignRight)

        self._programmatic_update = True
        self.insertTopLevelItem(index, item)
        self._programmatic_update = False

    def AddMask(self, mask):
        if mask.index not in self.mask_list_index:
            local_position = len(self.mask_list_index)
            self.mask_list_index[mask.index] = local_position
            self.InsertNewItem(
                local_position,
                mask.name,
                str(mask.threshold_range),
                mask.colour,
            )

    def EditMaskThreshold(self, global_mask_id, threshold_range):
        if global_mask_id in self.mask_list_index:
            local_pos = self.mask_list_index[global_mask_id]
            try:
                if 0 <= local_pos < self.topLevelItemCount():
                    item = self.topLevelItem(local_pos)
                    if item:
                        self._programmatic_update = True
                        item.setText(3, str(threshold_range))
                        self._programmatic_update = False
            except Exception:
                pass

    def EditMaskColour(self, global_mask_id, colour):
        if global_mask_id in self.mask_list_index:
            local_pos = self.mask_list_index[global_mask_id]
            try:
                if 0 <= local_pos < self.topLevelItemCount():
                    item = self.topLevelItem(local_pos)
                    if item:
                        item.setIcon(1, self.CreateColourBitmap(colour))
            except Exception:
                pass

    def GetSelected(self):
        """
        Return all items selected (highlighted).
        """
        selected = []
        for global_mask_id, local_pos in self.mask_list_index.items():
            item = self.topLevelItem(local_pos)
            if item and item.isSelected():
                selected.append(global_mask_id)
        selected.sort(reverse=True)
        return selected


# -------------------------------------------------
# -------------------------------------------------
class SurfacePage(QWidget):
    """
    Page related to surface items.
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.categories = {}
        self.sizer = QVBoxLayout(self)
        self.sizer.setContentsMargins(0, 0, 0, 0)

        self.__init_gui()

        Publisher.subscribe(self.AddSurface, "Update surface info in GUI")
        Publisher.subscribe(self.RepopulateSurfaces, "Repopulate surfaces")
        Publisher.subscribe(self.OnCloseProject, "Close project data")
        Publisher.subscribe(self.EditSurfaceTransparency, "Set surface transparency")
        Publisher.subscribe(self.EditSurfaceColour, "Set surface colour")
        Publisher.subscribe(self.OnShowSingle, "Show single surface")
        Publisher.subscribe(self.OnShowMultiple, "Show multiple surfaces")
        Publisher.subscribe(self.update_current_surface_data, "Update surface info in GUI")
        Publisher.subscribe(self.update_select_all_checkbox, "Update surface select all checkbox")

    def __init_gui(self):
        self.buttonctrl = SurfaceButtonControlPanel(self)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_widget)

        self.sizer.addWidget(self.scroll_area, 1)
        self.sizer.addWidget(self.buttonctrl, 0)

        self.create_category("General")

    def create_category_header(self, parent, category):
        """Create header panel with category controls"""
        header_panel = QWidget(parent)
        header_layout = QHBoxLayout(header_panel)
        header_layout.setContentsMargins(2, 0, 2, 0)

        expand_btn = QPushButton("\u25bc", header_panel)
        expand_btn.setFixedSize(20, 20)
        expand_btn.clicked.connect(
            lambda checked, cat=category: self.toggle_category_expansion(cat)
        )

        category_label = QLabel(category, header_panel)
        font = category_label.font()
        font.setBold(True)
        category_label.setFont(font)

        invisible_icon = QIcon(os.path.join(inv_paths.ICON_DIR, "object_invisible.png"))
        visible_icon = QIcon(os.path.join(inv_paths.ICON_DIR, "object_visible.png"))

        visibility_btn = QPushButton(header_panel)
        visibility_btn.setFixedSize(24, 24)
        visibility_btn.setIcon(visible_icon)
        visibility_btn.setToolTip("Toggle visibility for all surfaces in this category")
        visibility_btn.clicked.connect(
            lambda checked, cat=category: self.on_category_visibility_toggle(cat)
        )

        select_all_cb = QCheckBox("", header_panel)
        select_all_cb.setTristate(True)
        select_all_cb.setToolTip("Select/Unselect all surfaces in this category")
        select_all_cb.clicked.connect(
            lambda checked, cat=category: self.on_category_select_all(cat, checked)
        )

        header_layout.addWidget(expand_btn)
        header_layout.addWidget(category_label, 1)
        header_layout.addWidget(visibility_btn)
        header_layout.addWidget(select_all_cb)

        return (
            header_panel,
            expand_btn,
            visibility_btn,
            select_all_cb,
            invisible_icon,
            visible_icon,
        )

    def create_category(self, category):
        (
            header_panel,
            expand_btn,
            visibility_btn,
            select_all_cb,
            invisible_icon,
            visible_icon,
        ) = self.create_category_header(self.scroll_widget, category)

        content_panel = QWidget(self.scroll_widget)
        listctrl = SurfacesListCtrlPanel(content_panel, category=category)
        content_layout = QVBoxLayout(content_panel)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(listctrl)

        self.categories[category] = {
            "header": header_panel,
            "content": content_panel,
            "expand_btn": expand_btn,
            "visibility_btn": visibility_btn,
            "select_all_cb": select_all_cb,
            "list": listctrl,
            "invisible_icon": invisible_icon,
            "visible_icon": visible_icon,
            "expanded": True,
        }

        stretch_item = self.scroll_layout.takeAt(self.scroll_layout.count() - 1)
        self.scroll_layout.addWidget(header_panel)
        self.scroll_layout.addWidget(content_panel)
        self.scroll_layout.addStretch()

        return listctrl

    def toggle_category_expansion(self, category):
        if category not in self.categories:
            return

        category_info = self.categories[category]
        content_panel = category_info["content"]
        expand_btn = category_info["expand_btn"]
        is_expanded = category_info["expanded"]

        if is_expanded:
            content_panel.hide()
            expand_btn.setText("\u25b6")
            self.categories[category]["expanded"] = False
        else:
            content_panel.show()
            expand_btn.setText("\u25bc")
            self.categories[category]["expanded"] = True

        self.update_scroll_layout()

    def on_category_visibility_toggle(self, category):
        """Toggle visibility for all surfaces in the given category"""
        if category not in self.categories:
            return

        listctrl = self.categories[category]["list"]
        visibility_btn = self.categories[category]["visibility_btn"]
        invisible_icon = self.categories[category]["invisible_icon"]
        visible_icon = self.categories[category]["visible_icon"]

        is_visible = False
        for local_pos in listctrl.surface_list_index.values():
            item = listctrl.topLevelItem(local_pos)
            if item and item.data(0, Qt.ItemDataRole.UserRole):
                is_visible = True
                break

        new_visibility = not is_visible
        for global_surface_id, local_pos in listctrl.surface_list_index.items():
            listctrl.SetItemImage(local_pos, int(new_visibility))
            Publisher.sendMessage(
                "Show surface",
                index=global_surface_id,
                visibility=new_visibility,
            )

        if new_visibility:
            visibility_btn.setIcon(visible_icon)
        else:
            visibility_btn.setIcon(invisible_icon)

    def on_category_select_all(self, category, select_all):
        """Select or unselect all surfaces in the given category"""
        if category not in self.categories:
            return

        listctrl = self.categories[category]["list"]

        for local_pos in listctrl.surface_list_index.values():
            item = listctrl.topLevelItem(local_pos)
            if item:
                item.setSelected(select_all)

    def update_select_all_checkbox(self, category):
        """Update the select all checkbox state based on current selection"""
        if category not in self.categories:
            return

        listctrl = self.categories[category]["list"]
        select_all_cb = self.categories[category]["select_all_cb"]

        total_items = len(listctrl.surface_list_index)
        if total_items == 0:
            select_all_cb.setCheckState(Qt.CheckState.Unchecked)
            return

        selected_items = len(listctrl.GetSelected())

        if selected_items == 0:
            select_all_cb.setCheckState(Qt.CheckState.Unchecked)
        elif selected_items == total_items:
            select_all_cb.setCheckState(Qt.CheckState.Checked)
        else:
            select_all_cb.setCheckState(Qt.CheckState.PartiallyChecked)

    def _update_visibility_button_icon(self, category):
        """Update the visibility button icon based on current visibility state"""
        if category not in self.categories:
            return

        listctrl = self.categories[category]["list"]
        visibility_btn = self.categories[category]["visibility_btn"]
        invisible_icon = self.categories[category]["invisible_icon"]
        visible_icon = self.categories[category]["visible_icon"]

        any_visible = False
        for local_pos in listctrl.surface_list_index.values():
            item = listctrl.topLevelItem(local_pos)
            if item and item.data(0, Qt.ItemDataRole.UserRole):
                any_visible = True
                break

        if any_visible:
            visibility_btn.setIcon(visible_icon)
        else:
            visibility_btn.setIcon(invisible_icon)

    def AddSurface(self, surface):
        category = getattr(surface, "category", "General")
        if category not in self.categories:
            self.create_category(category)

        self.categories[category]["list"].InsertSurfaceItem(surface)
        self.update_scroll_layout()
        self.update_select_all_checkbox(category)

        self._update_visibility_button_icon(category)

    def RepopulateSurfaces(self, clear_project=False):
        old_widget = self.scroll_area.takeWidget()
        if old_widget:
            old_widget.deleteLater()

        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setContentsMargins(2, 2, 2, 2)
        self.scroll_layout.addStretch()
        self.scroll_area.setWidget(self.scroll_widget)

        self.categories.clear()

        if not clear_project:
            self.create_category("General")
            surface_dict = project.Project().surface_dict
            for i in sorted(surface_dict.keys()):
                surface = surface_dict[i]
                self.AddSurface(surface)
        else:
            self.create_category("General")

        self.update_scroll_layout()

    def OnPaneChanged(self, evt):
        self.update_scroll_layout()

    def update_scroll_layout(self):
        """Update scroll panel layout"""
        self.scroll_widget.updateGeometry()
        self.scroll_widget.adjustSize()

    def OnCloseProject(self):
        self.RepopulateSurfaces(clear_project=True)

    def EditSurfaceColour(self, surface_index, colour):
        """Edit surface colour in the appropriate category list"""
        for category_info in self.categories.values():
            listctrl = category_info["list"]
            if surface_index in listctrl.surface_list_index:
                listctrl.EditSurfaceColour(surface_index, colour)
                return

    def EditSurfaceTransparency(self, surface_index, transparency):
        for category_info in self.categories.values():
            listctrl = category_info["list"]
            if surface_index in listctrl.surface_list_index:
                listctrl.EditSurfaceTransparency(surface_index, transparency)
                return

    def update_current_surface_data(self, surface):
        for category_info in self.categories.values():
            listctrl = category_info["list"]
            listctrl.update_current_surface_data(surface)

    def OnShowSingle(self, index, visibility):
        for category_info in self.categories.values():
            listctrl = category_info["list"]
            for key in list(listctrl.surface_list_index.keys()):
                show = (key == index) and visibility
                local_idx = listctrl.surface_list_index[key]
                listctrl.SetItemImage(local_idx, int(show))
                Publisher.sendMessage("Show surface", index=key, visibility=show)

    def OnShowMultiple(self, index_list, visibility):
        for category_info in self.categories.values():
            listctrl = category_info["list"]
            for key in list(listctrl.surface_list_index.keys()):
                show = (key in index_list) and visibility
                local_idx = listctrl.surface_list_index[key]
                listctrl.SetItemImage(local_idx, int(show))
                if listctrl.GetItemImage(local_idx) != int(show):
                    Publisher.sendMessage("Show surface", index=key, visibility=show)


class SurfaceButtonControlPanel(QWidget):
    """
    Button control panel that includes data notebook operations.
    TODO: Enhace interface with parent class - it is really messed up
    """

    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.__init_gui()

    def __init_gui(self):
        icon_new = QIcon(os.path.join(inv_paths.ICON_DIR, "data_new.png"))
        icon_remove = QIcon(os.path.join(inv_paths.ICON_DIR, "data_remove.png"))
        icon_duplicate = QIcon(os.path.join(inv_paths.ICON_DIR, "data_duplicate.png"))
        icon_open = QIcon(os.path.join(inv_paths.ICON_DIR, "load_mesh.png"))

        button_new = QPushButton(icon_new, "", self)
        button_new.setFixedSize(QSize(24, 20))
        button_new.setToolTip(_("Create a new surface"))
        button_new.clicked.connect(self.OnNew)

        button_remove = QPushButton(icon_remove, "", self)
        button_remove.setFixedSize(QSize(24, 20))
        button_remove.setToolTip(_("Remove surface"))
        button_remove.clicked.connect(self.OnRemove)

        button_duplicate = QPushButton(icon_duplicate, "", self)
        button_duplicate.setFixedSize(QSize(24, 20))
        button_duplicate.setToolTip(_("Duplicate surface"))
        button_duplicate.clicked.connect(self.OnDuplicate)

        button_open = QPushButton(icon_open, "", self)
        button_open.setFixedSize(QSize(24, 20))
        button_open.setToolTip(_("Import a surface file into InVesalius"))
        button_open.clicked.connect(self.OnOpenMesh)

        sizer = QHBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(button_new)
        sizer.addWidget(button_remove)
        sizer.addWidget(button_duplicate)
        sizer.addWidget(button_open)
        sizer.addStretch()

    def OnNew(self):
        sl = slice_.Slice()
        dialog = dlg.SurfaceCreationDialog(
            None,
            -1,
            _("New surface"),
            mask_edited=sl.current_mask.was_edited,
        )
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                ok = 1
            else:
                ok = 0
        except Exception:
            ok = 1

        if ok:
            surface_options = dialog.GetValue()
            Publisher.sendMessage(
                "Create surface from index",
                surface_parameters=surface_options,
            )

    def OnRemove(self):
        all_selected_indices = []
        for category_info in self.parent.categories.values():
            listctrl = category_info["list"]
            selected = listctrl.GetSelected()
            all_selected_indices.extend(selected)
            if selected:
                listctrl.RemoveSurfaces(selected)

    def OnDuplicate(self):
        all_selected_indices = []
        for category_info in self.parent.categories.values():
            listctrl = category_info["list"]
            selected = listctrl.GetSelected()
            all_selected_indices.extend(selected)

        if all_selected_indices:
            Publisher.sendMessage("Duplicate surfaces", surface_indexes=all_selected_indices)
        else:
            dlg.SurfaceSelectionRequiredForDuplication()

    def OnOpenMesh(self):
        filename = dlg.ShowImportMeshFilesDialog()
        if filename:
            Publisher.sendMessage("Import surface file", filename=filename)


class SurfacesListCtrlPanel(InvListCtrl):
    def __init__(self, parent, category="General", **kwargs):
        super().__init__(parent)
        self._click_check = False
        self.category = category
        self.__init_columns()
        self.__init_image_list()
        self.__init_evt()
        self.__bind_events_wx()

        self.current_color = [255, 255, 255]
        self.current_transparency = 0
        self.surface_list_index = {}
        self.surface_bmp_idx_to_name = {}

    def __init_evt(self):
        pass

    def __bind_events_wx(self):
        self.itemChanged.connect(self._on_item_changed)
        self.itemSelectionChanged.connect(self.on_selection_changed)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.on_mouse_right_click)

    def keyPressEvent(self, event):
        key = event.key()
        if (sys.platform == "darwin") and (key == Qt.Key.Key_Backspace):
            self.RemoveSurfaces()
        elif key == Qt.Key.Key_Delete:
            self.RemoveSurfaces()
        else:
            super().keyPressEvent(event)

    def _on_item_changed(self, item, column):
        if self._programmatic_update:
            return
        if column == 2:
            index = self.indexOfTopLevelItem(item)
            Publisher.sendMessage("Change surface name", index=index, name=item.text(2))

    def on_selection_changed(self):
        Publisher.sendMessage("Update surface select all checkbox", category=self.category)

    def on_mouse_right_click(self, pos):
        item = self.itemAt(pos)
        if not item:
            return
        focused_idx = self.indexOfTopLevelItem(item)

        Publisher.sendMessage("Change surface selected", surface_index=focused_idx)

        menu = QMenu(self)

        action_colour = menu.addAction(_("Change color"))
        action_colour.triggered.connect(lambda: self.change_surface_color(None))

        action_transparency = menu.addAction(_("Change transparency"))
        action_transparency.triggered.connect(lambda: self.change_transparency(None))

        action_duplicate = menu.addAction(_("Duplicate"))
        action_duplicate.triggered.connect(lambda: self.duplicate_surface(None))

        menu.addSeparator()

        action_delete = menu.addAction(_("Delete"))
        action_delete.triggered.connect(lambda: self.delete_surface(None))

        menu.exec(self.viewport().mapToGlobal(pos))

    def update_current_surface_data(self, surface):
        self.current_color = [int(255 * c) for c in surface.colour][:3]
        self.current_transparency = int(100 * surface.transparency)

    def OnChangeColor(self, item_idx):
        global_surface_id = None
        for surface_id, local_pos in self.surface_list_index.items():
            if local_pos == item_idx:
                global_surface_id = surface_id
                break

        if global_surface_id is None:
            return

        Publisher.sendMessage("Change surface selected", surface_index=global_surface_id)
        self.change_surface_color(None)

    def change_surface_color(self, event):
        current_item = self.currentItem()
        focused_idx = self.indexOfTopLevelItem(current_item) if current_item else -1
        current_color = self.current_color

        new_color = dlg.ShowColorDialog(color_current=current_color)

        if not new_color:
            return

        new_vtk_color = [c / 255.0 for c in new_color]

        Publisher.sendMessage(
            "Set surface colour",
            surface_index=focused_idx,
            colour=new_vtk_color,
        )

        Publisher.sendMessage("Change surface selected", surface_index=focused_idx)

    def OnChangeTransparency(self, item_idx):
        global_surface_id = None
        for surface_id, local_pos in self.surface_list_index.items():
            if local_pos == item_idx:
                global_surface_id = surface_id
                break

        if global_surface_id is None:
            return

        Publisher.sendMessage("Change surface selected", surface_index=global_surface_id)
        self.change_transparency(None)

    def change_transparency(self, event):
        current_item = self.currentItem()
        focused_idx = self.indexOfTopLevelItem(current_item) if current_item else -1
        initial_value = self.current_transparency

        transparency_dialog = dlg.SurfaceTransparencyDialog(
            self,
            surface_index=focused_idx,
            transparency=initial_value,
        )

        if transparency_dialog.exec() == QDialog.DialogCode.Accepted:
            new_value = transparency_dialog.get_value()
        else:
            new_value = initial_value

        Publisher.sendMessage(
            "Set surface transparency",
            surface_index=focused_idx,
            transparency=new_value / 100.0,
        )

        Publisher.sendMessage("Change surface selected", surface_index=focused_idx)

    def duplicate_surface(self, event):
        selected_items = self.GetSelected()
        if selected_items:
            Publisher.sendMessage("Duplicate surfaces", surface_indexes=selected_items)
        else:
            dlg.SurfaceSelectionRequiredForDuplication()

    def delete_surface(self, event):
        result = dlg.ShowConfirmationDialog(msg=_("Delete surface?"))
        if result != QDialog.DialogCode.Accepted:
            return
        self.RemoveSurfaces()

    def RemoveSurfaces(self, selected_items=None):
        """
        Remove item given its index.
        """
        if not selected_items:
            selected_items = self.GetSelected()

        if selected_items:
            Publisher.sendMessage("Remove surfaces", surface_indexes=selected_items)
            Publisher.sendMessage("Repopulate surfaces")
        else:
            dlg.SurfaceSelectionRequiredForRemoval()

    def OnCloseProject(self):
        self.clear()
        self.surface_list_index = {}
        self.surface_bmp_idx_to_name = {}

    def OnItemSelected_(self, evt):
        pass

    def GetSelected(self):
        """
        Return all items selected (highlighted).
        """
        selected = []
        for global_surface_id, local_pos in self.surface_list_index.items():
            item = self.topLevelItem(local_pos)
            if item and item.isSelected():
                selected.append(global_surface_id)
        selected.sort(reverse=True)
        return selected

    def __init_columns(self):
        self.setHeaderLabels(
            [
                "",
                "",
                _("Name"),
                _("Volume (mm\u00b3)"),
                _("Area (mm\u00b2)"),
                _("Transparency"),
            ]
        )
        self.setColumnWidth(0, 25)
        self.setColumnWidth(1, 25)
        self.setColumnWidth(2, 85)
        self.setColumnWidth(3, 85)
        self.setColumnWidth(4, 85)
        self.setColumnWidth(5, 80)

    def __init_image_list(self):
        self._load_icons()

    def OnCheckItem(self, index, flag):
        global_idx = -1
        for g_id, l_id in self.surface_list_index.items():
            if l_id == index:
                global_idx = g_id
                break

        if global_idx == -1:
            return

        Publisher.sendMessage("Show surface", index=global_idx, visibility=flag)

    def InsertSurfaceItem(self, surface):
        index = surface.index
        name = surface.name
        colour = surface.colour
        volume = f"{surface.volume:.3f}"
        area = f"{surface.area:.3f}"
        transparency = f"{int(100 * surface.transparency)}%"

        if index not in self.surface_list_index:
            colour_icon = self.CreateColourBitmap(colour)

            local_position = self.topLevelItemCount()
            self.surface_list_index[index] = local_position

            self.InsertNewItem(
                local_position,
                name,
                volume,
                area,
                transparency,
                colour,
                colour_icon,
            )
        else:
            local_position = self.surface_list_index[index]
            self.UpdateItemInfo(local_position, name, volume, area, transparency, colour)

    def InsertNewItem(
        self,
        index=0,
        label="Surface 1",
        volume="0 mm3",
        area="0 mm2",
        transparency="0%%",
        colour=None,
        colour_icon=None,
    ):
        item = QTreeWidgetItem()
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        item.setIcon(0, self.icon_visible)
        item.setData(0, Qt.ItemDataRole.UserRole, True)
        if colour_icon:
            item.setIcon(1, colour_icon)
        item.setText(2, label)
        item.setText(3, volume)
        item.setText(4, area)
        item.setText(5, transparency)
        item.setTextAlignment(5, Qt.AlignmentFlag.AlignRight)

        self._programmatic_update = True
        self.insertTopLevelItem(index, item)
        self._programmatic_update = False

    def UpdateItemInfo(
        self,
        index=0,
        label="Surface 1",
        volume="0 mm3",
        area="0 mm2",
        transparency="0%%",
        colour=None,
    ):
        item = self.topLevelItem(index)
        if item:
            self._programmatic_update = True
            item.setText(2, label)
            item.setText(3, volume)
            item.setText(4, area)
            item.setText(5, transparency)
            self._programmatic_update = False
            self.SetItemImage(index, 1)

    def EditSurfaceTransparency(self, surface_index, transparency):
        if surface_index in self.surface_list_index:
            local_pos = self.surface_list_index[surface_index]
            item = self.topLevelItem(local_pos)
            if item:
                self._programmatic_update = True
                item.setText(5, f"{int(transparency * 100)}%")
                self._programmatic_update = False

    def EditSurfaceColour(self, surface_index, colour):
        if surface_index in self.surface_list_index:
            local_pos = self.surface_list_index[surface_index]
            item = self.topLevelItem(local_pos)
            if item:
                item.setIcon(1, self.CreateColourBitmap(colour))


# -------------------------------------------------
# -------------------------------------------------


class MeasuresListCtrlPanel(InvListCtrl):
    def __init__(self, parent, **kwargs):
        super().__init__(parent)
        self._click_check = False
        self.__init_columns()
        self.__init_image_list()
        self.__init_evt()
        self.__bind_events_wx()
        self._list_index = {}
        self._bmp_idx_to_name = {}

    def __init_evt(self):
        Publisher.subscribe(self.AddItem_, "Update measurement info in GUI")
        Publisher.subscribe(self.EditItemColour, "Set measurement colour")
        Publisher.subscribe(self.OnCloseProject, "Close project data")
        Publisher.subscribe(self.OnShowSingle, "Show single measurement")
        Publisher.subscribe(self.OnShowMultiple, "Show multiple measurements")
        Publisher.subscribe(self.OnLoadData, "Load measurement dict")
        Publisher.subscribe(self.OnRemoveGUIMeasure, "Remove GUI measurement")

    def __bind_events_wx(self):
        self.itemChanged.connect(self._on_item_changed)
        self.itemSelectionChanged.connect(self.OnItemSelected_)

    def keyPressEvent(self, event):
        key = event.key()
        if (sys.platform == "darwin") and (key == Qt.Key.Key_Backspace):
            self.RemoveMeasurements()
        elif key == Qt.Key.Key_Delete:
            self.RemoveMeasurements()
        else:
            super().keyPressEvent(event)

    def _on_item_changed(self, item, column):
        if self._programmatic_update:
            return
        if column == 1:
            index = self.indexOfTopLevelItem(item)
            Publisher.sendMessage("Change measurement name", index=index, name=item.text(1))

    def OnRemoveGUIMeasure(self, measure_index):
        if measure_index in self._list_index:
            self.takeTopLevelItem(measure_index)

            old_dict = self._list_index
            new_dict = {}
            j = 0
            for i in old_dict:
                if i != measure_index:
                    new_dict[j] = old_dict[i]
                    j += 1
            self._list_index = new_dict

    def RemoveMeasurements(self):
        """
        Remove items selected.
        """
        selected_items = self.GetSelected()
        selected_items.sort(reverse=True)

        old_dict = self._list_index
        if selected_items:
            for index in selected_items:
                new_dict = {}
                self.takeTopLevelItem(index)
                for i in old_dict:
                    if i < index:
                        new_dict[i] = old_dict[i]
                    if i > index:
                        new_dict[i - 1] = old_dict[i]
                old_dict = new_dict
            self._list_index = new_dict
            Publisher.sendMessage("Remove measurements", indexes=selected_items)
        else:
            dlg.MeasureSelectionRequiredForRemoval()

    def OnCloseProject(self):
        self.clear()
        self._list_index = {}
        self._bmp_idx_to_name = {}

    def OnItemSelected_(self):
        pass

    def GetSelected(self):
        """
        Return all items selected (highlighted).
        """
        selected = []
        for index in self._list_index:
            item = self.topLevelItem(index)
            if item and item.isSelected():
                selected.append(index)
        selected.sort(reverse=True)

        return selected

    def __init_columns(self):
        self.setHeaderLabels(["", _("Name"), _("Location"), _("Type"), _("Value")])
        self.setColumnWidth(0, 25)
        self.setColumnWidth(1, 65)
        self.setColumnWidth(2, 55)
        self.setColumnWidth(3, 50)
        self.setColumnWidth(4, 75)

    def __init_image_list(self):
        self._load_icons()

    def OnCheckItem(self, index, flag):
        Publisher.sendMessage("Show measurement", index=index, visibility=flag)

    def OnShowSingle(self, index, visibility):
        for key in self._list_index.keys():
            if key != index:
                self.SetItemImage(key, not visibility)
                Publisher.sendMessage("Show measurement", index=key, visibility=not visibility)
        self.SetItemImage(index, visibility)
        Publisher.sendMessage("Show measurement", index=index, visibility=visibility)

    def OnShowMultiple(self, index_list, visibility):
        for key in self._list_index.keys():
            if key not in index_list:
                self.SetItemImage(key, not visibility)
                Publisher.sendMessage(
                    "Show measurement",
                    index=key,
                    visibility=not visibility,
                )
        for index in index_list:
            self.SetItemImage(index, visibility)
            Publisher.sendMessage("Show measurement", index=index, visibility=visibility)

    def OnLoadData(self, measurement_dict, spacing=(1.0, 1.0, 1.0)):
        for i in sorted(measurement_dict):
            m = measurement_dict[i]
            colour_icon = self.CreateColourBitmap(m.colour)

            self._list_index[m.index] = colour_icon

            colour = [255 * c for c in m.colour]
            type = TYPE[m.type]
            location = LOCATION[m.location]
            if m.type == const.LINEAR:
                value = f"{m.value:.2f} mm"
            elif m.type == const.ANGULAR:
                value = f"{m.value:.2f}\u00b0"
            else:
                value = f"{m.value:.3f}"
            self.InsertNewItem(m.index, m.name, colour, location, type, value)

            if not m.visible:
                self.SetItemImage(i, False)

    def AddItem_(self, index, name, colour, location, type_, value):
        if index not in self._list_index:
            colour_icon = self.CreateColourBitmap(colour)

            index_list = self._list_index.keys()
            self._list_index[index] = colour_icon

            if (index in index_list) and index_list:
                try:
                    self.UpdateItemInfo(index, name, colour, location, type_, value)
                except Exception:
                    self.InsertNewItem(index, name, colour, location, type_, value)
            else:
                self.InsertNewItem(index, name, colour, location, type_, value)
        else:
            try:
                self.UpdateItemInfo(index, name, colour, location, type_, value)
            except Exception:
                self.InsertNewItem(index, name, colour, location, type_, value)

    def InsertNewItem(
        self,
        index=0,
        label="Measurement 1",
        colour=None,
        location="SURFACE",
        type_="LINEAR",
        value="0 mm",
    ):
        colour_icon = self._list_index.get(index)

        item = QTreeWidgetItem()
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        item.setIcon(0, self.icon_visible)
        item.setData(0, Qt.ItemDataRole.UserRole, True)
        if colour_icon:
            item.setIcon(1, colour_icon)
        item.setText(1, label)
        item.setText(2, location)
        item.setText(3, type_)
        item.setText(4, value)
        item.setTextAlignment(4, Qt.AlignmentFlag.AlignRight)

        self._programmatic_update = True
        self.insertTopLevelItem(index, item)
        self._programmatic_update = False

    def UpdateItemInfo(
        self,
        index=0,
        label="Measurement 1",
        colour=None,
        location="SURFACE",
        type_="LINEAR",
        value="0 mm",
    ):
        item = self.topLevelItem(index)
        if item:
            colour_icon = self._list_index.get(index)
            self._programmatic_update = True
            if colour_icon:
                item.setIcon(1, colour_icon)
            item.setText(1, label)
            item.setText(2, location)
            item.setText(3, type_)
            item.setText(4, value)
            self._programmatic_update = False
            self.SetItemImage(index, 1)

    def EditItemColour(self, measure_index, colour):
        """ """
        new_icon = self.CreateColourBitmap(colour)
        self._list_index[measure_index] = new_icon
        item = self.topLevelItem(measure_index)
        if item:
            item.setIcon(1, new_icon)


# *******************************************************************
# *******************************************************************


class AnnotationsListCtrlPanel(QTreeWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.setRootIsDecorated(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.setAllColumnsShowFocus(True)
        self._click_check = False
        self.__init_columns()
        self.__init_image_list()
        self.__init_evt()

        self.Populate()

    def __init_evt(self):
        self.itemDoubleClicked.connect(self._on_item_activated)

    def __init_columns(self):
        self.setHeaderLabels(["", _("Name"), _("Type"), _("Value")])
        self.setColumnWidth(0, 25)
        self.setColumnWidth(1, 90)
        self.setColumnWidth(2, 50)
        self.setColumnWidth(3, 120)

    def __init_image_list(self):
        self.icon_visible = QIcon(os.path.join(inv_paths.ICON_DIR, "object_visible.png"))
        self.icon_invisible = QIcon(os.path.join(inv_paths.ICON_DIR, "object_invisible.png"))
        self.icon_colour = QIcon(os.path.join(inv_paths.ICON_DIR, "object_colour.png"))

    def _on_item_activated(self, item, column):
        index = self.indexOfTopLevelItem(item)
        visible = not bool(item.data(0, Qt.ItemDataRole.UserRole))
        item.setData(0, Qt.ItemDataRole.UserRole, visible)
        item.setIcon(0, self.icon_visible if visible else self.icon_invisible)
        self.OnCheckItem(index, visible)

    def OnCheckItem(self, index, flag):
        if flag:
            print("checked, ", index)
        else:
            print("unchecked, ", index)

    def InsertNewItem(self, index=0, name="Axial 1", type_="2d", value="bla", colour=None):
        item = QTreeWidgetItem()
        item.setIcon(1, self.icon_colour)
        item.setText(1, name)
        item.setText(2, type_)
        item.setText(3, value)
        self.insertTopLevelItem(index, item)

    def Populate(self):
        data_list = (
            (0, "Axial 1", "2D", "blalbalblabllablalbla"),
            (1, "Coronal 1", "2D", "hello here we are again"),
            (2, "Volume 1", "3D", "hey ho, lets go"),
        )
        for data in data_list:
            self.InsertNewItem(data[0], data[1], data[2], data[3])
