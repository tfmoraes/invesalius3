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
# --------------------------------------------------------------------------

import code
import sys
from typing import Any, Dict

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QTextCursor
from PySide6.QtWidgets import (
    QMainWindow,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import invesalius.constants as const
import invesalius.session as ses
from invesalius.i18n import tr as _
from invesalius.navigation.markers import MarkersControl
from invesalius.navigation.navigation import Navigation
from invesalius.navigation.robot import Robot
from invesalius.navigation.tracker import Tracker
from invesalius.pubsub import pub as Publisher


class ShellTextEdit(QTextEdit):
    """QTextEdit-based interactive Python console widget."""

    def __init__(self, parent=None, locals_dict=None, intro_text=""):
        super().__init__(parent)
        self.setFont(QFont("Monospace", 10))
        self.setAcceptRichText(False)

        self._locals = locals_dict or {}
        self.interp = code.InteractiveConsole(self._locals)
        self._history = []
        self._history_idx = 0
        self._prompt = ">>> "
        self._cont_prompt = "... "
        self._current_prompt = self._prompt
        self._partial_source = ""

        if intro_text:
            self.insertPlainText(intro_text + "\n")
        self.insertPlainText(self._prompt)

        self._redirect_stdout = _StdoutRedirector(self)

    def keyPressEvent(self, event):
        cursor = self.textCursor()
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            line = self._get_current_line()
            self._history.append(line)
            self._history_idx = len(self._history)
            self.moveCursor(QTextCursor.End)
            self.insertPlainText("\n")
            self._execute(line)
            return
        elif event.key() == Qt.Key_Up:
            if self._history_idx > 0:
                self._history_idx -= 1
                self._replace_current_line(self._history[self._history_idx])
            return
        elif event.key() == Qt.Key_Down:
            if self._history_idx < len(self._history) - 1:
                self._history_idx += 1
                self._replace_current_line(self._history[self._history_idx])
            else:
                self._history_idx = len(self._history)
                self._replace_current_line("")
            return
        elif event.key() == Qt.Key_Backspace:
            if cursor.positionInBlock() <= len(self._current_prompt):
                return
        super().keyPressEvent(event)

    def _get_current_line(self):
        block = self.textCursor().block().text()
        return block[len(self._current_prompt) :]

    def _replace_current_line(self, text):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.StartOfBlock, QTextCursor.MoveAnchor)
        cursor.movePosition(QTextCursor.EndOfBlock, QTextCursor.KeepAnchor)
        cursor.insertText(self._current_prompt + text)
        self.setTextCursor(cursor)

    def _execute(self, line):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = self._redirect_stdout
        sys.stderr = self._redirect_stdout
        try:
            source = self._partial_source + line
            needs_more = self.interp.runsource(source, "<console>")
            if needs_more:
                self._partial_source = source + "\n"
                self._current_prompt = self._cont_prompt
            else:
                self._partial_source = ""
                self._current_prompt = self._prompt
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        self.insertPlainText(self._current_prompt)
        self.moveCursor(QTextCursor.End)


class _StdoutRedirector:
    def __init__(self, text_edit):
        self._text_edit = text_edit

    def write(self, text):
        self._text_edit.moveCursor(QTextCursor.End)
        self._text_edit.insertPlainText(text)

    def flush(self):
        pass


class InteractiveShellPanel(QWidget):
    """
    Interactive Python shell panel for debugging and development.
    """

    def __init__(
        self, parent: QWidget = None, app_context: Dict[str, Any] = {}, introText: str = ""
    ):
        super().__init__(parent)

        self.shell = ShellTextEdit(self, locals_dict=app_context, intro_text=introText)

        sizer = QVBoxLayout(self)
        sizer.setContentsMargins(0, 0, 0, 0)
        sizer.addWidget(self.shell)
        self.setLayout(sizer)

    def update_context(self, new_context):
        """Update the shell's local namespace with new objects."""
        if hasattr(self.shell, "interp"):
            self.shell.interp.locals.update(new_context)


class InteractiveShellFrame(QMainWindow):
    """
    Standalone frame for the interactive shell.
    """

    def __init__(
        self, parent: QWidget = None, app_context: Dict[str, Any] = {}, introText: str = ""
    ):
        super().__init__(parent)
        self.setWindowTitle(_("InVesalius Interactive Python Shell"))
        self.resize(800, 600)
        self.setWindowFlags(self.windowFlags() | Qt.Tool)

        self.shell_panel = InteractiveShellPanel(self, app_context, introText)
        self.setCentralWidget(self.shell_panel)

        Publisher.subscribe(self.update_context, "Update shell context")
        Publisher.subscribe(
            self.add_navigation_context, "Add navigation context to interactive shell"
        )

    def update_context(self, new_context: dict):
        """Update the shell's context."""
        self.shell_panel.update_context(new_context)

    def add_navigation_context(self):
        """Add navigation-related objects to the shell context."""
        mode = ses.Session().GetConfig("mode")
        navigation_context = {}
        if mode == const.MODE_NAVIGATOR:
            navigation_context["markers"] = MarkersControl()
            navigation_context["navigation"] = Navigation()
            navigation_context["robot"] = Robot()
            navigation_context["tracker"] = Tracker()

        Publisher.sendMessage("Update shell context", new_context=navigation_context)
