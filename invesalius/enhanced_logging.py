#!/usr/bin/env python3
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
# -------------------------------------------------------------------------

"""
Module for enhanced logging in InVesalius.

This module provides a comprehensive logging system for InVesalius,
including:
- Structured logging with different levels
- Log rotation
- Log filtering
- Log viewing GUI
- Integration with the error handling system
"""

import json
import logging
import logging.config
import logging.handlers
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
from invesalius import inv_paths
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher
from invesalius.utils import deep_merge_dict

# Constants
LOG_CONFIG_PATH = os.path.join(inv_paths.USER_INV_DIR, "log_config.json")
DEFAULT_LOGFILE = os.path.join(
    inv_paths.USER_LOG_DIR, datetime.now().strftime("invlog-%Y_%m_%d-%I_%M_%S_%p.log")
)

# Default logging configuration
DEFAULT_LOG_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "detailed": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
        },
        "simple": {"format": "%(asctime)s - %(levelname)s - %(message)s"},
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "simple",
            "stream": "ext://sys.stdout",
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "detailed",
            "filename": DEFAULT_LOGFILE,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8",
        },
    },
    "loggers": {
        "invesalius": {"level": "DEBUG", "handlers": ["console", "file"], "propagate": False}
    },
    "root": {"level": "INFO", "handlers": ["console", "file"], "propagate": True},
}


class LogRecord:
    """Class to represent a log record for the GUI."""

    def __init__(
        self,
        timestamp: str,
        level: str,
        name: str,
        message: str,
        pathname: Optional[str] = None,
        lineno: Optional[int] = None,
        exc_info: Optional[str] = None,
        args: Optional[tuple] = None,
        funcName: Optional[str] = None,
        thread: Optional[int] = None,
        threadName: Optional[str] = None,
    ):
        self.timestamp = timestamp
        self.level = level
        self.name = name
        self.message = message
        self.pathname = pathname
        self.lineno = lineno
        self.exc_info = exc_info
        self.args = args
        self.funcName = funcName
        self.thread = thread
        self.threadName = threadName

    @classmethod
    def from_record(cls, record: logging.LogRecord) -> "LogRecord":
        """Create a LogRecord from a logging.LogRecord."""
        exc_info = None
        if record.exc_info:
            import traceback

            exc_info = "".join(traceback.format_exception(*record.exc_info))

        # Format the message with args if any
        message = record.getMessage()

        return cls(
            timestamp=datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S,%f")[:-3],
            level=record.levelname,
            name=record.name,
            message=message,
            pathname=record.pathname,
            lineno=record.lineno,
            exc_info=exc_info,
            args=record.args if hasattr(record, "args") else None,
            funcName=record.funcName if hasattr(record, "funcName") else None,
            thread=record.thread if hasattr(record, "thread") else None,
            threadName=record.threadName if hasattr(record, "threadName") else None,
        )

    def get_full_details(self) -> str:
        """Return a detailed string representation of the record."""
        details = f"Timestamp: {self.timestamp}\n"
        details += f"Level: {self.level}\n"
        details += f"Component: {self.name}\n"
        details += f"Message: {self.message}\n"

        if self.pathname:
            details += f"File: {self.pathname}\n"

        if self.lineno:
            details += f"Line: {self.lineno}\n"

        if self.funcName:
            details += f"Function: {self.funcName}\n"

        if self.thread:
            details += f"Thread: {self.thread}"
            if self.threadName:
                details += f" ({self.threadName})"
            details += "\n"

        if self.exc_info:
            details += f"\nException Information:\n{self.exc_info}\n"

        return details


class InMemoryHandler(logging.Handler):
    """Logging handler that keeps records in memory for the GUI."""

    def __init__(self, capacity: int = 5000):
        super().__init__()
        self.capacity = capacity
        self.records = []
        self.formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(pathname)s:%(lineno)d - %(message)s"
        )
        # Set level to DEBUG to capture all logs
        self.setLevel(logging.DEBUG)

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a record."""
        try:
            log_record = LogRecord.from_record(record)
            self.records.append(log_record)
            if len(self.records) > self.capacity:
                self.records.pop(0)
        except Exception:
            self.handleError(record)

    def get_records(self, level: Optional[str] = None) -> List[LogRecord]:
        """Get records, optionally filtered by level."""
        if level is None:
            return self.records

        return [r for r in self.records if r.level == level]

    def clear(self) -> None:
        """Clear all records."""
        self.records = []


class LogViewerFrame(QMainWindow):
    """Enhanced frame for viewing detailed logs with filtering and searching capabilities."""

    def __init__(self, parent: Optional[QWidget], in_memory_handler: InMemoryHandler):
        """Initialize the enhanced log viewer frame."""
        super().__init__(parent)
        self.setWindowTitle(_("InVesalius Enhanced Log Viewer"))
        self.resize(1200, 800)

        self.in_memory_handler = in_memory_handler
        self.search_results = []
        self.current_search_index = -1
        self.search_history = []

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- Filter controls ---
        filter_layout = QHBoxLayout()

        filter_layout.addWidget(QLabel(_("Level:")))
        self.level_choice = QComboBox()
        self.level_choice.addItems(["ALL", "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
        self.level_choice.currentIndexChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.level_choice)

        filter_layout.addWidget(QLabel(_("Component:")))
        self.component_choice = QComboBox()
        self.component_choice.addItem("ALL")
        self.component_choice.currentIndexChanged.connect(self.on_filter_changed)
        filter_layout.addWidget(self.component_choice)

        filter_layout.addWidget(QLabel(_("Time:")))
        self.time_choice = QComboBox()
        self.time_choice.addItems(["ALL", "Last hour", "Last day", "Last week", "Custom..."])
        self.time_choice.currentIndexChanged.connect(self.on_time_filter_changed)
        filter_layout.addWidget(self.time_choice)

        filter_layout.addStretch()
        main_layout.addLayout(filter_layout)

        # --- Search controls ---
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel(_("Search:")))

        self.search_text = QComboBox()
        self.search_text.setEditable(True)
        self.search_text.lineEdit().returnPressed.connect(self.on_search)
        search_layout.addWidget(self.search_text, 1)

        search_button = QPushButton(_("Search"))
        search_button.clicked.connect(self.on_search)
        search_layout.addWidget(search_button)

        clear_search_button = QPushButton(_("Clear"))
        clear_search_button.clicked.connect(self.on_search_cancel)
        search_layout.addWidget(clear_search_button)

        self.search_prev_btn = QPushButton("\u25c0 Previous")
        self.search_prev_btn.setEnabled(False)
        self.search_prev_btn.clicked.connect(self.on_search_prev)
        search_layout.addWidget(self.search_prev_btn)

        self.search_next_btn = QPushButton("Next \u25b6")
        self.search_next_btn.setEnabled(False)
        self.search_next_btn.clicked.connect(self.on_search_next)
        search_layout.addWidget(self.search_next_btn)

        self.search_count = QLabel("")
        search_layout.addWidget(self.search_count)

        main_layout.addLayout(search_layout)

        # --- Splitter with table and detail view ---
        self.splitter = QSplitter(Qt.Orientation.Vertical)

        self.log_table = QTableWidget(0, 6)
        self.log_table.setHorizontalHeaderLabels(
            [_("Time"), _("Level"), _("Component"), _("Message"), _("File"), _("Line")]
        )
        self.log_table.setColumnWidth(0, 180)
        self.log_table.setColumnWidth(1, 80)
        self.log_table.setColumnWidth(2, 180)
        self.log_table.setColumnWidth(3, 400)
        self.log_table.setColumnWidth(4, 200)
        self.log_table.setColumnWidth(5, 60)
        self.log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.log_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.log_table.cellClicked.connect(self.on_cell_select)
        self.log_table.cellDoubleClicked.connect(self.on_cell_double_click)
        self.splitter.addWidget(self.log_table)

        detail_widget = QWidget()
        detail_layout = QVBoxLayout(detail_widget)
        detail_layout.addWidget(QLabel(_("Details:")))
        self.detail_text = QPlainTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setFont(QFont("Monospace", 10))
        detail_layout.addWidget(self.detail_text)
        self.splitter.addWidget(detail_widget)

        self.splitter.setSizes([500, 300])
        self.splitter.setChildrenCollapsible(False)
        main_layout.addWidget(self.splitter, 1)

        # --- Bottom buttons ---
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self.auto_refresh_cb = QCheckBox(_("Auto refresh"))
        self.auto_refresh_cb.setChecked(True)
        button_layout.addWidget(self.auto_refresh_cb)

        refresh_button = QPushButton(_("Refresh"))
        refresh_button.clicked.connect(self.on_refresh)
        button_layout.addWidget(refresh_button)

        clear_button = QPushButton(_("Clear"))
        clear_button.clicked.connect(self.on_clear)
        button_layout.addWidget(clear_button)

        export_button = QPushButton(_("Export"))
        export_button.clicked.connect(self.on_export)
        button_layout.addWidget(export_button)

        copy_all_button = QPushButton(_("Copy All"))
        copy_all_button.clicked.connect(self.on_copy_all)
        button_layout.addWidget(copy_all_button)

        main_layout.addLayout(button_layout)

        # Status bar
        self.statusBar()

        self.populate_logs()
        self.update_component_list()

        # Auto-refresh timer
        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.on_timer)
        self.refresh_timer.start(2000)

    def populate_logs(self):
        """Populate the table with filtered logs."""
        self.log_table.setRowCount(0)
        records = self.get_filtered_records()

        self.log_table.setRowCount(len(records))
        for i, record in enumerate(records):
            self.log_table.setItem(i, 0, QTableWidgetItem(record.timestamp))
            self.log_table.setItem(i, 1, QTableWidgetItem(record.level))
            self.log_table.setItem(i, 2, QTableWidgetItem(record.name))
            self.log_table.setItem(i, 3, QTableWidgetItem(record.message))
            self.log_table.setItem(
                i, 4, QTableWidgetItem(os.path.basename(record.pathname) if record.pathname else "")
            )
            self.log_table.setItem(
                i, 5, QTableWidgetItem(str(record.lineno) if record.lineno else "")
            )

            bg = fg = bold = None
            if record.level == "CRITICAL":
                bg, fg, bold = QColor(255, 150, 150), QColor(128, 0, 0), True
            elif record.level == "ERROR":
                bg, fg = QColor(255, 200, 200), QColor(139, 0, 0)
            elif record.level == "WARNING":
                bg, fg = QColor(255, 200, 200), QColor(139, 0, 0)
            elif record.level == "INFO":
                bg = QColor(240, 255, 240)
            elif record.level == "DEBUG":
                bg, fg = QColor(240, 240, 240), QColor(100, 100, 100)

            if bg or fg or bold:
                for col in range(6):
                    item = self.log_table.item(i, col)
                    if item:
                        if bg:
                            item.setBackground(bg)
                        if fg:
                            item.setForeground(fg)
                        if bold:
                            f = item.font()
                            f.setBold(True)
                            item.setFont(f)

        if self.search_results:
            self.highlight_search_results()

        total_records = len(self.in_memory_handler.records)
        filtered_records = len(records)
        level_counts: Dict[str, int] = {}
        for record in records:
            level_counts[record.level] = level_counts.get(record.level, 0) + 1
        level_info = ", ".join([f"{lv}: {ct}" for lv, ct in level_counts.items()])
        self.statusBar().showMessage(
            f"Showing {filtered_records} of {total_records} log records ({level_info})"
        )

        if not self.search_results and filtered_records > 0:
            self.log_table.scrollToItem(self.log_table.item(filtered_records - 1, 0))
            self.log_table.selectRow(filtered_records - 1)
            self._show_details_for_row(filtered_records - 1)

    def get_filtered_records(self):
        """Get records filtered by the current filter settings."""
        records = self.in_memory_handler.records.copy()

        level = self.level_choice.currentText()
        if level != "ALL":
            records = [r for r in records if r.level == level]

        component = self.component_choice.currentText()
        if component != "ALL":
            records = [r for r in records if r.name == component]

        time_filter = self.time_choice.currentText()
        if time_filter != "ALL":
            now = datetime.now()
            if time_filter == "Last hour":
                cutoff = now - timedelta(hours=1)
                records = [r for r in records if self._parse_timestamp(r.timestamp) > cutoff]
            elif time_filter == "Last day":
                cutoff = now - timedelta(days=1)
                records = [r for r in records if self._parse_timestamp(r.timestamp) > cutoff]
            elif time_filter == "Last week":
                cutoff = now - timedelta(days=7)
                records = [r for r in records if self._parse_timestamp(r.timestamp) > cutoff]

        if not self.search_results:
            search_text = self.search_text.currentText().lower()
            if search_text:
                records = [
                    r
                    for r in records
                    if (
                        search_text in r.message.lower()
                        or search_text in r.name.lower()
                        or (r.pathname and search_text in r.pathname.lower())
                        or search_text in r.level.lower()
                    )
                ]

        return records

    def _parse_timestamp(self, timestamp_str):
        """Parse a timestamp string into a datetime object."""
        try:
            return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
        except ValueError:
            try:
                return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return datetime(1970, 1, 1)

    def update_component_list(self):
        """Update the component filter list with available components."""
        components = {"ALL"}
        for record in self.in_memory_handler.records:
            if record.name:
                components.add(record.name)

        current = self.component_choice.currentText()
        self.component_choice.blockSignals(True)
        self.component_choice.clear()
        for comp in sorted(components):
            self.component_choice.addItem(comp)

        idx = self.component_choice.findText(current)
        self.component_choice.setCurrentIndex(idx if idx >= 0 else 0)
        self.component_choice.blockSignals(False)

    def on_filter_changed(self):
        self.populate_logs()

    def on_time_filter_changed(self):
        if self.time_choice.currentText() == "Custom...":
            QMessageBox.information(
                self,
                _("Not Implemented"),
                _("Custom time range filtering will be implemented in a future version."),
            )
            self.time_choice.setCurrentIndex(0)
        self.populate_logs()

    def on_search(self):
        search_text = self.search_text.currentText().strip().lower()
        if not search_text:
            self.search_results = []
            self.current_search_index = -1
            self.search_prev_btn.setEnabled(False)
            self.search_next_btn.setEnabled(False)
            self.search_count.setText("")
            self.populate_logs()
            return

        if search_text not in self.search_history:
            self.search_history.append(search_text)
            if len(self.search_history) > 10:
                self.search_history.pop(0)
            self.search_text.clear()
            self.search_text.addItems(self.search_history)
            self.search_text.setEditText(search_text)

        records = self.get_filtered_records()
        self.search_results = []
        for i, record in enumerate(records):
            if (
                search_text in record.message.lower()
                or search_text in record.name.lower()
                or (record.pathname and search_text in record.pathname.lower())
                or search_text in record.level.lower()
            ):
                self.search_results.append(i)

        if self.search_results:
            self.current_search_index = 0
            self.search_prev_btn.setEnabled(True)
            self.search_next_btn.setEnabled(True)
            self.search_count.setText(
                f"Match {self.current_search_index + 1} of {len(self.search_results)}"
            )
            self.highlight_search_results()
            self.navigate_to_current_search()
        else:
            self.current_search_index = -1
            self.search_prev_btn.setEnabled(False)
            self.search_next_btn.setEnabled(False)
            self.search_count.setText("No matches")
            self.populate_logs()

    def on_search_cancel(self):
        self.search_text.setEditText("")
        self.search_results = []
        self.current_search_index = -1
        self.search_prev_btn.setEnabled(False)
        self.search_next_btn.setEnabled(False)
        self.search_count.setText("")
        self.populate_logs()

    def on_search_prev(self):
        if not self.search_results:
            return
        self.current_search_index = (self.current_search_index - 1) % len(self.search_results)
        self.search_count.setText(
            f"Match {self.current_search_index + 1} of {len(self.search_results)}"
        )
        self.navigate_to_current_search()

    def on_search_next(self):
        if not self.search_results:
            return
        self.current_search_index = (self.current_search_index + 1) % len(self.search_results)
        self.search_count.setText(
            f"Match {self.current_search_index + 1} of {len(self.search_results)}"
        )
        self.navigate_to_current_search()

    def highlight_search_results(self):
        search_text = self.search_text.currentText().strip().lower()
        if not search_text:
            return
        for i in self.search_results:
            for col in range(6):
                item = self.log_table.item(i, col)
                if item:
                    bg = item.background().color()
                    item.setBackground(
                        QColor(min(bg.red() + 20, 255), min(bg.green() + 20, 255), bg.blue())
                    )

    def navigate_to_current_search(self):
        if not self.search_results or self.current_search_index < 0:
            return
        row = self.search_results[self.current_search_index]
        self.log_table.clearSelection()
        item = self.log_table.item(row, 0)
        if item:
            self.log_table.scrollToItem(item)
        self.log_table.selectRow(row)
        self._show_details_for_row(row)

    def on_refresh(self):
        self.update_component_list()
        self.populate_logs()

    def on_timer(self):
        if self.auto_refresh_cb.isChecked():
            current_records = len(self.get_filtered_records())
            if current_records != self.log_table.rowCount():
                self.update_component_list()
                self.populate_logs()

    def on_clear(self):
        reply = QMessageBox.question(
            self,
            _("Confirm Clear"),
            _("Are you sure you want to clear all logs?"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.in_memory_handler.clear()
            self.populate_logs()

    def on_export(self):
        path, _ = QFileDialog.getSaveFileName(
            self,
            _("Export logs"),
            "",
            "CSV files (*.csv);;Text files (*.txt)",
        )
        if not path:
            return

        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write("Timestamp,Level,Component,Message,File,Line\n")
                for record in self.get_filtered_records():
                    f.write(
                        f'"{record.timestamp}","{record.level}","{record.name}","{record.message}",'
                    )
                    f.write(f'"{os.path.basename(record.pathname) if record.pathname else ""}",')
                    f.write(f'"{record.lineno if record.lineno else ""}"\n')

            QMessageBox.information(self, _("Export Complete"), _("Logs exported successfully."))
        except Exception as e:
            QMessageBox.critical(
                self, _("Export Error"), _("Error exporting logs: {0}").format(str(e))
            )

    def on_cell_select(self, row, _column):
        self._show_details_for_row(row)

    def on_cell_double_click(self, row, _column):
        records = self.get_filtered_records()
        if row >= len(records):
            return

        record = records[row]
        mono_font = QFont("Monospace", 10)

        dlg = QDialog(self)
        dlg.setWindowTitle(_("Log Record Details"))
        dlg.resize(900, 600)
        dlg_layout = QVBoxLayout(dlg)

        tabs = QTabWidget()

        basic_text = QPlainTextEdit()
        basic_text.setReadOnly(True)
        basic_text.setFont(mono_font)
        basic_text.setPlainText(record.get_full_details())

        if record.exc_info:
            tabs.addTab(basic_text, _("Basic Info"))
            exc_text = QPlainTextEdit()
            exc_text.setReadOnly(True)
            exc_text.setFont(mono_font)
            exc_text.setPlainText(record.exc_info)
            tabs.addTab(exc_text, _("Exception Info"))
        else:
            tabs.addTab(basic_text, _("Details"))

        if record.pathname and record.lineno:
            try:
                import linecache

                context_lines = []
                for i in range(max(1, record.lineno - 5), record.lineno + 6):
                    line = linecache.getline(record.pathname, i)
                    if line:
                        prefix = ">" if i == record.lineno else " "
                        context_lines.append(f"{prefix} {i:4d}: {line}")
                if context_lines:
                    ctx_text = QPlainTextEdit()
                    ctx_text.setReadOnly(True)
                    ctx_text.setFont(mono_font)
                    ctx_text.setPlainText(
                        f"Source file: {record.pathname}\n\n" + "".join(context_lines)
                    )
                    tabs.addTab(ctx_text, _("Source Context"))
            except Exception:
                pass

        dlg_layout.addWidget(tabs)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        copy_btn = QPushButton(_("Copy to Clipboard"))
        copy_btn.clicked.connect(lambda: self.copy_record_to_clipboard(record))
        btn_layout.addWidget(copy_btn)
        close_btn = QPushButton(_("Close"))
        close_btn.clicked.connect(dlg.accept)
        btn_layout.addWidget(close_btn)
        dlg_layout.addLayout(btn_layout)

        dlg.exec()

    def copy_record_to_clipboard(self, record):
        clipboard = QApplication.clipboard()
        clipboard.setText(record.get_full_details())
        QMessageBox.information(self, _("Copy Complete"), _("Record details copied to clipboard."))

    def _show_details_for_row(self, row):
        records = self.get_filtered_records()
        if 0 <= row < len(records):
            self.detail_text.setPlainText(records[row].get_full_details())

    def closeEvent(self, event):
        if self.refresh_timer.isActive():
            self.refresh_timer.stop()
        self.hide()
        logging.getLogger("invesalius.enhanced_logging").info("Log viewer closed")
        event.ignore()

    def on_copy_all(self):
        records = self.get_filtered_records()
        if not records:
            QMessageBox.warning(self, _("Copy Failed"), _("No log records to copy."))
            return

        text = ""
        for record in records:
            text += f"{record.timestamp} - {record.level} - {record.name} - {record.message}\n"

        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        QMessageBox.information(self, _("Copy Complete"), _("Log records copied to clipboard."))


class EnhancedLogger:
    """Enhanced logger for InVesalius."""

    def __init__(self):
        """Initialize the enhanced logger."""
        # Make a deep copy to avoid modifying the original
        self._config = DEFAULT_LOG_CONFIG.copy()
        self._logger = logging.getLogger("invesalius")

        # Create in-memory handler with larger capacity
        self._in_memory_handler = InMemoryHandler(capacity=10000)
        self._in_memory_handler.setLevel(logging.DEBUG)  # Capture all logs

        # Add handler to the root logger to ensure we capture all logs
        root_logger = logging.getLogger()
        root_logger.addHandler(self._in_memory_handler)

        # Also add to our app logger
        self._logger.addHandler(self._in_memory_handler)

        self._log_viewer_frame = None

        # Create the log directory if it doesn't exist
        os.makedirs(inv_paths.USER_LOG_DIR, exist_ok=True)

        # Read the configuration file if it exists
        self._read_config()

        # Configure logging
        self._configure_logging()

        # Log startup message
        self._logger.info("Enhanced logging system initialized")

        # Register cleanup handler for application exit
        import atexit

        atexit.register(self.cleanup)

    def _read_config(self) -> None:
        """Read the logging configuration from the config file."""
        try:
            if os.path.exists(LOG_CONFIG_PATH):
                with open(LOG_CONFIG_PATH, "r") as f:
                    config = json.load(f)
                    self._config = deep_merge_dict(self._config.copy(), config)
        except Exception as e:
            print(f"Error reading log config: {e}")

    def _write_config(self) -> None:
        """Write the logging configuration to the config file."""
        try:
            with open(LOG_CONFIG_PATH, "w") as f:
                json.dump(self._config, f, indent=4)
        except Exception as e:
            print(f"Error writing log config: {e}")

    def _configure_logging(self) -> None:
        """Configure logging based on the configuration."""
        try:
            # Ensure the root logger will capture everything
            root_logger = logging.getLogger()
            root_logger.setLevel(logging.DEBUG)

            # Configure logging
            logging.config.dictConfig(self._config)

            # Get the logger
            self._logger = logging.getLogger("invesalius")

            # Ensure our loggers have the in-memory handler
            if not any(isinstance(h, InMemoryHandler) for h in self._logger.handlers):
                self._logger.addHandler(self._in_memory_handler)

            if not any(isinstance(h, InMemoryHandler) for h in root_logger.handlers):
                root_logger.addHandler(self._in_memory_handler)

            # Log the configuration
            self._logger.info("Logging configured")

            # Log additional configuration information
            handlers_info = {
                "file_logging": any(
                    isinstance(h, logging.FileHandler) for h in self._logger.handlers
                ),
                "console_logging": any(
                    isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
                    for h in self._logger.handlers
                ),
            }

            self._logger.info(
                f"file_logging: {int(handlers_info['file_logging'])}, console_logging: {int(handlers_info['console_logging'])}"
            )
            self._logger.info("configureLogging called ...")
            self._logger.info(
                f"file_logging: {int(handlers_info['file_logging'])}, console_logging: {int(handlers_info['console_logging'])}"
            )

        except Exception as e:
            print(f"Error configuring logging: {e}")
            import traceback

            traceback.print_exc()

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger."""
        if name is None:
            return self._logger

        return logging.getLogger(f"invesalius.{name}")

    def show_log_viewer(self, parent: Optional[QWidget] = None) -> None:
        """Show the log viewer."""
        try:
            if self._log_viewer_frame is None:
                self._log_viewer_frame = LogViewerFrame(parent, self._in_memory_handler)
            else:
                # Restart the timer if it was stopped
                if not self._log_viewer_frame.refresh_timer.isActive():
                    self._log_viewer_frame.refresh_timer.start(2000)

                # Refresh the log viewer with the latest logs
                self._log_viewer_frame.update_component_list()
                self._log_viewer_frame.populate_logs()

            self._log_viewer_frame.show()
            self._log_viewer_frame.raise_()
        except Exception as e:
            import traceback

            traceback.print_exc()
            logging.error(f"Error showing log viewer: {e}")

    def set_level(self, level: Union[str, int]) -> None:
        """Set the logging level."""
        self._logger.setLevel(level)

        # Update the configuration
        self._config["loggers"]["invesalius"]["level"] = (
            level if isinstance(level, str) else logging.getLevelName(level)
        )

        # Write the configuration
        self._write_config()

    def get_level(self) -> int:
        """Get the logging level."""
        return self._logger.level

    def set_file_logging(self, enabled: bool) -> None:
        """Enable or disable file logging."""
        # Update the configuration
        if enabled:
            if "file" not in self._config["handlers"]:
                self._config["handlers"]["file"] = {
                    "class": "logging.handlers.RotatingFileHandler",
                    "level": "DEBUG",
                    "formatter": "detailed",
                    "filename": DEFAULT_LOGFILE,
                    "maxBytes": 10485760,  # 10MB
                    "backupCount": 5,
                    "encoding": "utf8",
                }

            if "file" not in self._config["loggers"]["invesalius"]["handlers"]:
                self._config["loggers"]["invesalius"]["handlers"].append("file")
        else:
            if "file" in self._config["loggers"]["invesalius"]["handlers"]:
                self._config["loggers"]["invesalius"]["handlers"].remove("file")

        # Reconfigure logging
        self._configure_logging()

        # Write the configuration
        self._write_config()

    def set_console_logging(self, enabled: bool) -> None:
        """Enable or disable console logging."""
        # Update the configuration
        if enabled:
            if "console" not in self._config["handlers"]:
                self._config["handlers"]["console"] = {
                    "class": "logging.StreamHandler",
                    "level": "INFO",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout",
                }

            if "console" not in self._config["loggers"]["invesalius"]["handlers"]:
                self._config["loggers"]["invesalius"]["handlers"].append("console")
        else:
            if "console" in self._config["loggers"]["invesalius"]["handlers"]:
                self._config["loggers"]["invesalius"]["handlers"].remove("console")

        # Reconfigure logging
        self._configure_logging()

        # Write the configuration
        self._write_config()

    def set_log_file(self, path: str) -> None:
        """Set the log file path."""
        # Update the configuration
        if "file" in self._config["handlers"]:
            self._config["handlers"]["file"]["filename"] = path

        # Reconfigure logging
        self._configure_logging()

        # Write the configuration
        self._write_config()

    def get_log_file(self) -> str:
        """Get the log file path."""
        if "file" in self._config["handlers"]:
            return self._config["handlers"]["file"]["filename"]

        return DEFAULT_LOGFILE

    def cleanup(self):
        """Clean up resources when the application exits."""
        try:
            if self._log_viewer_frame:
                # Stop the timer
                if self._log_viewer_frame.refresh_timer.isActive():
                    self._log_viewer_frame.refresh_timer.stop()

                # Log the cleanup
                self._logger.info("Cleaning up enhanced logger resources")
        except Exception as e:
            print(f"Error during enhanced logger cleanup: {e}")


# Create the enhanced logger instance
enhanced_logger = EnhancedLogger()


# Function to get the enhanced logger
def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Get a logger."""
    return enhanced_logger.get_logger(name)


# Function to show the log viewer
def show_log_viewer(parent: Optional[QWidget] = None) -> None:
    """Show the log viewer."""
    try:
        enhanced_logger.show_log_viewer(parent)
    except Exception as e:
        import traceback

        traceback.print_exc()
        logging.error(f"Error showing log viewer: {e}")


# Function to set the logging level
def set_level(level: Union[str, int]) -> None:
    """Set the logging level."""
    enhanced_logger.set_level(level)


# Function to get the logging level
def get_level() -> int:
    """Get the logging level."""
    return enhanced_logger.get_level()


# Function to enable or disable file logging
def set_file_logging(enabled: bool) -> None:
    """Enable or disable file logging."""
    enhanced_logger.set_file_logging(enabled)


# Function to enable or disable console logging
def set_console_logging(enabled: bool) -> None:
    """Enable or disable console logging."""
    enhanced_logger.set_console_logging(enabled)


# Function to set the log file path
def set_log_file(path: str) -> None:
    """Set the log file path."""
    enhanced_logger.set_log_file(path)


# Function to get the log file path
def get_log_file() -> str:
    """Get the log file path."""
    return enhanced_logger.get_log_file()


# Register a menu handler for the log viewer
def register_menu_handler() -> None:
    """Register a menu handler for the log viewer."""
    Publisher.subscribe(show_log_viewer, "Show log viewer")
