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

from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QVBoxLayout,
)

import invesalius.i18n as i18n
from invesalius.i18n import tr as _
from invesalius.inv_paths import ICON_DIR

file_path = os.path.split(__file__)[0]

if hasattr(sys, "frozen") and (sys.frozen == "windows_exe" or sys.frozen == "console_exe"):
    abs_file_path = os.path.abspath(
        file_path + os.sep + ".." + os.sep + ".." + os.sep + ".." + os.sep + ".."
    )
    ICON_DIR = os.path.abspath(os.path.join(abs_file_path, "icons"))

if not os.path.exists(ICON_DIR):
    ICON_DIR = os.path.abspath(os.path.join(file_path, "..", "..", "..", "..", "..", "icons"))


class ComboBoxLanguage:
    def __init__(self, parent):
        """Initialize combobox with language icons"""

        dict_locales = i18n.GetLocales()

        self.locales = dict_locales.values()
        self.locales = sorted(self.locales)

        self.locales_key = [dict_locales.get_key(value) for value in self.locales]

        self.os_locale = i18n.GetLocaleOS()

        try:
            os_lang = self.os_locale[0:2]
        except TypeError:
            os_lang = None

        selection = self.locales_key.index("en")

        self.combo = QComboBox(parent)
        for key in self.locales_key:
            filepath = os.path.join(ICON_DIR, f"{key}.png")
            icon = QIcon(QPixmap(str(filepath)))
            self.combo.addItem(icon, dict_locales[key], key)
            if os_lang and key.startswith(os_lang):
                selection = self.locales_key.index(key)

        self.combo.setCurrentIndex(selection)

    def GetComboBox(self):
        return self.combo

    def GetLocalesKey(self):
        return self.locales_key


class LanguageDialog(QDialog):
    """Dialog to select the language for InVesalius UI."""

    def __init__(self, parent=None, startApp=None):
        super().__init__(parent)
        self.__TranslateMessage__()
        self.setWindowTitle(_("Language selection"))
        self.__init_gui()

    def __init_gui(self):
        layout = QVBoxLayout(self)

        self.txtMsg = QLabel(_("Choose user interface language"))
        layout.addWidget(self.txtMsg)

        self.cmb = ComboBoxLanguage(self)
        self.bitmapCmb = self.cmb.GetComboBox()
        layout.addWidget(self.bitmapCmb)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def GetSelectedLanguage(self):
        """Return String with Selected Language"""
        self.locales_key = self.cmb.GetLocalesKey()
        return self.locales_key[self.bitmapCmb.currentIndex()]

    def __TranslateMessage__(self):
        """Translate Messages of the Window"""
        os_language = i18n.GetLocaleOS()

        if os_language[0:2] == "pt":
            _ = i18n.InstallLanguage("pt_BR")
        elif os_language[0:2] == "es":
            _ = i18n.InstallLanguage("es")
        else:
            _ = i18n.InstallLanguage("en")

    def Cancel(self, event):
        """Close LanguageDialog"""
        self.close()
