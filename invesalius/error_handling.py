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
Module for centralized error handling in InVesalius.

This module provides a comprehensive error handling system for InVesalius,
including:
- Custom exception classes for different types of errors
- Error handling decorators for functions and methods
- User-friendly error messages
- Integration with the logging system
- Crash reporting functionality
"""

import functools
import inspect
import logging
import os
import platform
import sys
import traceback
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import psutil
from PySide6.QtWidgets import QApplication, QDialog, QMessageBox

import invesalius.constants as const
from invesalius import inv_paths
from invesalius.i18n import tr as _
from invesalius.pubsub import pub as Publisher


class ErrorCategory(Enum):
    """Enum for categorizing errors in InVesalius."""

    GENERAL = auto()
    IO = auto()
    DICOM = auto()
    SEGMENTATION = auto()
    SURFACE = auto()
    RENDERING = auto()
    NAVIGATION = auto()
    PLUGIN = auto()
    NETWORK = auto()
    CONFIGURATION = auto()
    USER_INTERFACE = auto()
    MEMORY = auto()
    PERFORMANCE = auto()
    HARDWARE = auto()
    EXTERNAL_LIBRARY = auto()


class ErrorSeverity(Enum):
    """Enum for error severity levels in InVesalius."""

    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class InVesaliusException(Exception):
    """Base exception class for all InVesalius exceptions."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.GENERAL,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        self.message = message
        self.category = category
        self.severity = severity
        self.details = details or {}
        self.original_exception = original_exception
        self.timestamp = datetime.now()

        if original_exception:
            self.details["original_traceback"] = "".join(
                traceback.format_exception(
                    type(original_exception), original_exception, original_exception.__traceback__
                )
            )

        super().__init__(message)


class IOError(InVesaliusException):
    """Exception raised for I/O errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.IO,
            severity=ErrorSeverity.ERROR,
            details=details,
            original_exception=original_exception,
        )


class DicomError(InVesaliusException):
    """Exception raised for DICOM-related errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.DICOM,
            severity=ErrorSeverity.ERROR,
            details=details,
            original_exception=original_exception,
        )


class SegmentationError(InVesaliusException):
    """Exception raised for segmentation-related errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.SEGMENTATION,
            severity=ErrorSeverity.ERROR,
            details=details,
            original_exception=original_exception,
        )


class SurfaceError(InVesaliusException):
    """Exception raised for surface-related errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.SURFACE,
            severity=ErrorSeverity.ERROR,
            details=details,
            original_exception=original_exception,
        )


class RenderingError(InVesaliusException):
    """Exception raised for rendering-related errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.RENDERING,
            severity=ErrorSeverity.ERROR,
            details=details,
            original_exception=original_exception,
        )


class NavigationError(InVesaliusException):
    """Exception raised for navigation-related errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.NAVIGATION,
            severity=ErrorSeverity.ERROR,
            details=details,
            original_exception=original_exception,
        )


class PluginError(InVesaliusException):
    """Exception raised for plugin-related errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.PLUGIN,
            severity=ErrorSeverity.ERROR,
            details=details,
            original_exception=original_exception,
        )


class MemoryError(InVesaliusException):
    """Exception raised for memory-related errors."""

    def __init__(
        self,
        message: str,
        details: Optional[Dict[str, Any]] = None,
        original_exception: Optional[Exception] = None,
    ):
        super().__init__(
            message,
            category=ErrorCategory.MEMORY,
            severity=ErrorSeverity.ERROR,
            details=details,
            original_exception=original_exception,
        )


def handle_errors(
    error_message: str = "An error occurred",
    show_dialog: bool = True,
    log_error: bool = True,
    reraise: bool = False,
    expected_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    category: ErrorCategory = ErrorCategory.GENERAL,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
):
    """
    Decorator for handling errors in functions and methods.

    Parameters:
        error_message (str): The error message to display to the user.
        show_dialog (bool): Whether to show an error dialog to the user.
        log_error (bool): Whether to log the error.
        reraise (bool): Whether to reraise the exception after handling.
        expected_exceptions (tuple): The exceptions to catch.
        category (ErrorCategory): The category of the error.
        severity (ErrorSeverity): The severity of the error.

    Returns:
        The decorated function.
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except expected_exceptions as e:
                module_name = func.__module__
                function_name = func.__qualname__

                _, _, tb = sys.exc_info()
                while tb.tb_next:
                    tb = tb.tb_next
                line_number = tb.tb_lineno

                detailed_message = (
                    f"{error_message} in {module_name}.{function_name} (line {line_number})"
                )

                details = {
                    "module": module_name,
                    "function": function_name,
                    "line": line_number,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "traceback": traceback.format_exc(),
                }

                inv_exception = InVesaliusException(
                    detailed_message,
                    category=category,
                    severity=severity,
                    details=details,
                    original_exception=e,
                )

                if log_error:
                    from invesalius.gui import log

                    logger = log.invLogger.getLogger()

                    if severity == ErrorSeverity.DEBUG:
                        logger.debug(detailed_message, exc_info=True)
                    elif severity == ErrorSeverity.INFO:
                        logger.info(detailed_message, exc_info=True)
                    elif severity == ErrorSeverity.WARNING:
                        logger.warning(detailed_message, exc_info=True)
                    elif severity == ErrorSeverity.ERROR:
                        logger.error(detailed_message, exc_info=True)
                    elif severity == ErrorSeverity.CRITICAL:
                        logger.critical(detailed_message, exc_info=True)

                if show_dialog and QApplication.instance() is not None:
                    show_error_dialog(detailed_message, inv_exception)

                Publisher.sendMessage("Error occurred", error=inv_exception)

                if reraise:
                    raise inv_exception from e

                return None

        return wrapper

    return decorator


def show_error_dialog(message: str, exception: Optional[InVesaliusException] = None):
    """
    Show an error dialog to the user.
    """
    if QApplication.instance() is None:
        print(f"ERROR: {message}")
        if exception and exception.details.get("traceback"):
            print(exception.details["traceback"])
        return

    if exception:
        dlg = ErrorDialog(None, message, exception)
    else:
        QMessageBox.critical(None, _("Error"), message)
        return

    dlg.exec()


def get_system_info() -> Dict[str, str]:
    """
    Get system information for error reporting.
    """
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "memory": str(get_system_memory()),
        "invesalius_version": const.INVESALIUS_VERSION,
    }

    try:
        import PySide6

        info["pyside6_version"] = PySide6.__version__
    except ImportError:
        info["pyside6_version"] = "Not available"

    try:
        from vtkmodules.vtkCommonCore import vtkVersion

        info["vtk_version"] = vtkVersion.GetVTKVersion()
    except ImportError:
        info["vtk_version"] = "Not available"

    return info


def get_system_memory() -> int:
    """
    Get the total system memory in GB.
    """
    try:
        return psutil.virtual_memory().total // (1024**3)
    except ImportError:
        return 0


def create_crash_report(exception: InVesaliusException) -> str:
    """
    Create a crash report for an exception.
    """
    crash_dir = os.path.join(inv_paths.USER_LOG_DIR, "crash_reports")
    os.makedirs(crash_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"crash_report_{timestamp}.txt"
    filepath = os.path.join(crash_dir, filename)

    system_info = get_system_info()

    with open(filepath, "w") as f:
        f.write("InVesalius Crash Report\n")
        f.write("======================\n\n")

        f.write(f"Timestamp: {exception.timestamp}\n")
        f.write(f"Error Category: {exception.category.name}\n")
        f.write(f"Error Severity: {exception.severity.name}\n")
        f.write(f"Error Message: {exception.message}\n\n")

        f.write("System Information\n")
        f.write("------------------\n")
        for key, value in system_info.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")

        f.write("Error Details\n")
        f.write("------------\n")
        for key, value in exception.details.items():
            if key != "traceback" and key != "original_traceback":
                f.write(f"{key}: {value}\n")
        f.write("\n")

        if "traceback" in exception.details:
            f.write("Traceback\n")
            f.write("---------\n")
            f.write(exception.details["traceback"])
            f.write("\n\n")

        if "original_traceback" in exception.details:
            f.write("Original Traceback\n")
            f.write("-----------------\n")
            f.write(exception.details["original_traceback"])

    return filepath


class ErrorDialog(QDialog):
    """
    Dialog for displaying detailed error information.
    """

    def __init__(self, parent, message: str, exception: InVesaliusException):
        super().__init__(parent)
        self.setWindowTitle(_("Error"))
        self.resize(600, 400)

        self.exception = exception
        self._create_layout(message)

    def _create_layout(self, message: str):
        from PySide6.QtWidgets import (
            QHBoxLayout,
            QLabel,
            QPushButton,
            QTabWidget,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
        from PySide6.QtWidgets import QStyle

        main_layout = QVBoxLayout(self)

        error_layout = QHBoxLayout()
        icon_label = QLabel()
        icon = self.style().standardIcon(QStyle.StandardPixmap.SP_MessageBoxCritical)
        icon_label.setPixmap(icon.pixmap(32, 32))
        error_layout.addWidget(icon_label)

        error_text = QLabel(message)
        error_text.setWordWrap(True)
        error_layout.addWidget(error_text, 1)
        main_layout.addLayout(error_layout)

        tab_widget = QTabWidget()

        details_text = QTextEdit()
        details_text.setReadOnly(True)
        details = []
        details.append(f"Error Category: {self.exception.category.name}")
        details.append(f"Error Severity: {self.exception.severity.name}")
        details.append(f"Timestamp: {self.exception.timestamp}")
        for key, value in self.exception.details.items():
            if key != "traceback" and key != "original_traceback":
                details.append(f"{key}: {value}")
        details_text.setPlainText("\n".join(details))
        tab_widget.addTab(details_text, _("Details"))

        if "traceback" in self.exception.details:
            traceback_text = QTextEdit()
            traceback_text.setReadOnly(True)
            traceback_text.setPlainText(self.exception.details["traceback"])
            tab_widget.addTab(traceback_text, _("Traceback"))

        system_text = QTextEdit()
        system_text.setReadOnly(True)
        system_info = get_system_info()
        system_details = [f"{key}: {value}" for key, value in system_info.items()]
        system_text.setPlainText("\n".join(system_details))
        tab_widget.addTab(system_text, _("System Info"))

        main_layout.addWidget(tab_widget)

        button_layout = QHBoxLayout()
        crash_report_button = QPushButton(_("Create Crash Report"))
        crash_report_button.clicked.connect(self._on_crash_report)
        button_layout.addWidget(crash_report_button)
        button_layout.addStretch()

        ok_button = QPushButton(_("OK"))
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        main_layout.addLayout(button_layout)

    def _on_crash_report(self):
        crash_report_path = create_crash_report(self.exception)
        QMessageBox.information(
            self,
            _("Crash Report Created"),
            _("Crash report created at:\n\n{0}").format(crash_report_path),
        )


def global_exception_handler(exctype, value, tb):
    """
    Global exception handler for unhandled exceptions.
    """
    logging.critical("Unhandled exception", exc_info=(exctype, value, tb))

    exception = InVesaliusException(
        str(value),
        category=ErrorCategory.GENERAL,
        severity=ErrorSeverity.CRITICAL,
        details={"traceback": "".join(traceback.format_exception(exctype, value, tb))},
        original_exception=value,
    )

    crash_report_path = create_crash_report(exception)

    if QApplication.instance() is not None:
        show_error_dialog(
            _("An unhandled error occurred. A crash report has been created at:\n\n{0}").format(
                crash_report_path
            ),
            exception,
        )
    else:
        print(f"CRITICAL ERROR: {str(value)}")
        print(f"A crash report has been created at: {crash_report_path}")


sys.excepthook = global_exception_handler


def show_message(title, message, icon_type="information", log_level=logging.INFO):
    """
    Show a message to the user and log it.

    Parameters:
    -----------
    title : str
        The title of the message box.
    message : str
        The message to display and log.
    icon_type : str
        The icon type: 'information', 'warning', 'error', 'question'.
    log_level : int
        The logging level to use.

    Returns:
    --------
    The result of the message box.
    """
    frame = inspect.currentframe().f_back
    module_name = frame.f_globals["__name__"]
    logger = logging.getLogger(module_name)

    if log_level == logging.DEBUG:
        logger.debug(f"{title}: {message}")
    elif log_level == logging.INFO:
        logger.info(f"{title}: {message}")
    elif log_level == logging.WARNING:
        logger.warning(f"{title}: {message}")
    elif log_level == logging.ERROR:
        logger.error(f"{title}: {message}")
    elif log_level == logging.CRITICAL:
        logger.critical(f"{title}: {message}")

    if QApplication.instance() is None:
        return None

    if icon_type == "warning":
        return QMessageBox.warning(None, title, message)
    elif icon_type == "error":
        return QMessageBox.critical(None, title, message)
    elif icon_type == "question":
        return QMessageBox.question(None, title, message)
    else:
        return QMessageBox.information(None, title, message)


def show_info(title, message):
    """Show an information message and log it at INFO level."""
    return show_message(title, message, "information", logging.INFO)


def show_warning(title, message):
    """Show a warning message and log it at WARNING level."""
    return show_message(title, message, "warning", logging.WARNING)


def show_error(title, message):
    """Show an error message and log it at ERROR level."""
    return show_message(title, message, "error", logging.ERROR)


def show_question(title, message):
    """Show a question message and log it at INFO level."""
    return show_message(title, message, "question", logging.INFO)
