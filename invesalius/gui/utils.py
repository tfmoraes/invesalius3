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
import logging
from typing import TYPE_CHECKING

from invesalius.error_handling import (
    show_error,
    show_info,
    show_message,
    show_question,
    show_warning,
)

__all__ = ["show_message", "show_info", "show_warning", "show_error", "show_question"]


def message_dialog(message, title="InVesalius 3", icon_type="information"):
    """
    Show a message dialog and log it with the appropriate level.

    Parameters:
    -----------
    message : str
        The message to display and log.
    title : str
        The title of the message box.
    icon_type : str
        The icon type: 'information', 'warning', or 'error'.

    Returns:
    --------
    int
        The result of the message box.
    """
    log_level = logging.INFO
    if icon_type == "warning":
        log_level = logging.WARNING
    elif icon_type == "error":
        log_level = logging.ERROR

    return show_message(title, message, icon_type, log_level)


if TYPE_CHECKING:
    from PySide6.QtWidgets import QWidget


def calc_width_needed(widget: "QWidget", num_chars: int) -> int:
    fm = widget.fontMetrics()
    return fm.horizontalAdvance("M" * num_chars)
