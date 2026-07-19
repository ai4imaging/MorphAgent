"""Visual tokens and Qt stylesheet for MorphAgent."""

from __future__ import annotations

from qtpy.QtGui import QColor, QPalette
from qtpy.QtWidgets import QApplication


COLORS = {
    "ink_950": "#07111F",
    "ink_900": "#0B1626",
    "slate_850": "#111F33",
    "slate_800": "#172940",
    "slate_700": "#29405E",
    "text": "#E6F6FF",
    "muted": "#9CB0C8",
    "aqua": "#22D3EE",
    "aqua_dark": "#0891B2",
    "violet": "#A78BFA",
    "coral": "#FB7185",
    "success": "#34D399",
    "warning": "#FBBF24",
    "error": "#F87171",
}


STYLESHEET = f"""
QWidget {{
    background: {COLORS['ink_950']};
    color: {COLORS['text']};
    font-family: "Fira Sans", "Inter", "Segoe UI", sans-serif;
    font-size: 13px;
}}
QLabel, QCheckBox, QRadioButton {{ background: transparent; }}
QToolTip {{
    background: {COLORS['slate_800']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['slate_700']};
    padding: 6px;
}}
QLabel[role="display"] {{ font-size: 28px; font-weight: 700; }}
QLabel[role="homeDisplay"] {{ font-size: 34px; font-weight: 700; }}
QLabel[role="title"] {{ font-size: 19px; font-weight: 700; }}
QLabel[role="subtitle"] {{ color: {COLORS['muted']}; font-size: 14px; }}
QLabel[role="eyebrow"] {{ color: {COLORS['aqua']}; font-size: 11px; font-weight: 700; }}
QLabel[role="eyebrowMuted"] {{ color: {COLORS['muted']}; font-size: 10px; font-weight: 700; }}
QLabel[role="fieldLabel"] {{ color: {COLORS['text']}; font-size: 16px; font-weight: 700; margin-top: 4px; }}
QLabel[role="filterLabel"] {{ color: {COLORS['muted']}; font-size: 13px; font-weight: 700; padding-right: 2px; }}
QLabel[role="evidencePreviewTitle"] {{ color: {COLORS['text']}; font-size: 14px; font-weight: 700; }}
QLabel[role="evidenceSectionTitle"] {{ color: {COLORS['text']}; font-size: 16px; font-weight: 700; margin-top: 3px; }}
QLabel[role="evidenceFacts"] {{
    color: #C5D5E6;
    background: #182B40;
    border: 1px solid #3C5878;
    border-radius: 5px;
    padding: 6px 9px;
    font-size: 12px;
    font-weight: 600;
}}
QLabel[role="evidenceNote"] {{
    color: {COLORS['muted']};
    background: #101D2D;
    border: 1px solid #405269;
    border-radius: 6px;
    padding: 7px 9px;
    font-size: 12px;
}}
QLabel[role="scaleSummary"] {{
    color: #D7E7F7;
    background: #182B40;
    border: 1px solid #3C5878;
    border-radius: 6px;
    padding: 8px 11px;
    font-size: 13px;
    font-weight: 600;
}}
QLabel[role="muted"] {{ color: {COLORS['muted']}; }}
QLabel[role="success"] {{ color: {COLORS['success']}; font-weight: 600; }}
QLabel[role="warning"] {{ color: {COLORS['warning']}; font-weight: 600; }}
QLabel[role="error"] {{ color: {COLORS['error']}; font-weight: 600; }}
QFrame[card="true"], QGroupBox {{
    background: {COLORS['slate_850']};
    border: 1px solid {COLORS['slate_700']};
    border-radius: 10px;
}}
QFrame[stepCard="true"] {{
    background: #101D2D;
    border: 2px solid #405269;
    border-radius: 11px;
}}
QFrame[evidencePreview="true"] {{
    background: {COLORS['ink_900']};
    border: 1px solid {COLORS['slate_700']};
    border-radius: 8px;
}}
QGroupBox {{
    margin-top: 13px;
    padding: 16px 12px 12px 12px;
    font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: {COLORS['muted']};
}}
QPushButton {{
    background: {COLORS['slate_800']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['slate_700']};
    border-radius: 7px;
    padding: 8px 14px;
    min-height: 18px;
    font-weight: 600;
}}
QPushButton:hover {{ border-color: {COLORS['aqua']}; background: #19324A; }}
QPushButton:focus {{ border: 2px solid {COLORS['aqua']}; padding: 7px 13px; }}
QPushButton:disabled {{ color: #64748B; background: #101B2B; border-color: #21344D; }}
QPushButton[primary="true"] {{
    background: {COLORS['aqua_dark']};
    border-color: {COLORS['aqua']};
    color: #F0FDFF;
}}
QPushButton[primary="true"]:hover {{ background: #0E7490; }}
QPushButton[homePrimary="true"] {{
    min-height: 30px;
    padding: 12px 22px;
    border-radius: 9px;
    font-size: 15px;
    font-weight: 700;
}}
QPushButton[homePrimary="true"]:focus {{ padding: 11px 21px; }}
QPushButton[homeSecondary="true"] {{
    min-height: 28px;
    padding: 10px 20px;
    border-radius: 9px;
    background: #18283A;
    border: 1px solid #5B6B7E;
    font-size: 14px;
    font-weight: 700;
}}
QPushButton[homeSecondary="true"]:hover {{ background: #223449; border-color: {COLORS['aqua']}; }}
QPushButton[choiceAction="true"] {{
    background: #18283A;
    color: {COLORS['text']};
    border: 1px solid #5B6B7E;
    border-radius: 7px;
    min-height: 26px;
    text-align: left;
    padding: 10px 14px;
}}
QPushButton[choiceAction="true"]:hover {{ background: #223449; border-color: #8E9AAA; }}
QPushButton[largePrimary="true"] {{ min-height: 28px; padding: 11px 18px; font-size: 14px; }}
QPushButton[runCta="true"] {{
    min-height: 32px;
    padding: 12px 24px;
    border: 2px solid {COLORS['aqua']};
    border-radius: 9px;
    font-size: 16px;
    font-weight: 800;
}}
QPushButton[runCta="true"]:focus {{ padding: 12px 24px; }}
QPushButton[danger="true"] {{ border-color: {COLORS['error']}; color: {COLORS['error']}; }}
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
    background: {COLORS['ink_900']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['slate_700']};
    border-radius: 6px;
    padding: 7px;
    selection-background-color: {COLORS['aqua_dark']};
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus,
QSpinBox:focus, QDoubleSpinBox:focus {{ border: 2px solid {COLORS['aqua']}; padding: 6px; }}
QPlainTextEdit[evidenceText="true"] {{
    background: #081321;
    border: 0;
    border-radius: 4px;
    color: #D7E7F7;
    font-family: "SFMono-Regular", "Menlo", "Consolas", monospace;
    font-size: 12px;
    padding: 8px;
}}
QComboBox::drop-down {{ border: 0; width: 22px; }}
QComboBox QAbstractItemView {{
    background: {COLORS['slate_850']};
    color: {COLORS['text']};
    selection-background-color: {COLORS['aqua_dark']};
}}
QCheckBox {{ spacing: 7px; }}
QCheckBox::indicator {{ width: 16px; height: 16px; }}
QCheckBox::indicator:unchecked {{ background: {COLORS['ink_900']}; border: 1px solid {COLORS['slate_700']}; border-radius: 3px; }}
QCheckBox::indicator:checked {{ background: {COLORS['aqua_dark']}; border: 1px solid {COLORS['aqua']}; border-radius: 3px; }}
QCheckBox[choiceTile="true"] {{
    background: {COLORS['ink_900']};
    color: #C9D8E8;
    border: 1px solid #3B526F;
    border-radius: 7px;
    padding: 9px 12px;
    spacing: 10px;
    font-size: 14px;
    font-weight: 600;
}}
QCheckBox[choiceTile="true"]:hover {{ color: {COLORS['text']}; border-color: #8291A3; }}
QCheckBox[choiceTile="true"]:checked {{
    background: #103448;
    color: {COLORS['text']};
    border: 2px solid {COLORS['aqua']};
    padding: 8px 11px;
}}
QCheckBox[choiceTile="true"]::indicator:checked {{ background: {COLORS['aqua_dark']}; border: 1px solid {COLORS['aqua']}; }}
QRadioButton[choiceTile="true"] {{
    background: {COLORS['ink_900']};
    color: {COLORS['muted']};
    border: 1px solid {COLORS['slate_700']};
    border-radius: 7px;
    padding: 9px 12px;
    spacing: 8px;
    font-size: 14px;
    font-weight: 600;
}}
QRadioButton[choiceTile="true"]:hover {{ color: {COLORS['text']}; border-color: #8291A3; }}
QRadioButton[choiceTile="true"]:checked {{
    background: #102C3E;
    color: {COLORS['text']};
    border: 2px solid {COLORS['aqua']};
    padding: 8px 11px;
}}
QRadioButton[choiceTile="true"]::indicator {{ width: 14px; height: 14px; }}
QRadioButton[choiceTile="true"]::indicator:unchecked {{
    background: {COLORS['ink_950']};
    border: 1px solid {COLORS['slate_700']};
    border-radius: 7px;
}}
QRadioButton[choiceTile="true"]::indicator:checked {{
    background: {COLORS['aqua_dark']};
    border: 2px solid {COLORS['aqua']};
    border-radius: 7px;
}}
QRadioButton[filterChoice="true"] {{
    background: {COLORS['ink_900']};
    color: {COLORS['muted']};
    border: 1px solid {COLORS['slate_700']};
    border-radius: 7px;
    padding: 7px 12px;
    spacing: 7px;
    font-size: 13px;
    font-weight: 650;
}}
QRadioButton[filterChoice="true"]:hover {{
    color: {COLORS['text']};
    background: #18283A;
    border-color: #8291A3;
}}
QRadioButton[filterChoice="true"]:checked {{
    background: #102C3E;
    color: {COLORS['text']};
    border: 2px solid {COLORS['aqua']};
    padding: 6px 11px;
}}
QRadioButton[filterChoice="true"]::indicator {{ width: 12px; height: 12px; }}
QRadioButton[filterChoice="true"]::indicator:unchecked {{
    background: {COLORS['ink_950']};
    border: 1px solid {COLORS['slate_700']};
    border-radius: 6px;
}}
QRadioButton[filterChoice="true"]::indicator:checked {{
    background: {COLORS['aqua_dark']};
    border: 2px solid {COLORS['aqua']};
    border-radius: 6px;
}}
QListWidget#Navigation {{
    background: {COLORS['ink_900']};
    border: 0;
    border-right: 1px solid {COLORS['slate_700']};
    outline: none;
    padding: 8px;
}}
QListWidget#Navigation::item {{
    border-radius: 7px;
    padding: 10px 10px;
    margin: 2px 0;
    color: {COLORS['muted']};
}}
QListWidget#Navigation::item:selected {{
    color: {COLORS['text']};
    background: #15334A;
    border-left: 3px solid {COLORS['aqua']};
}}
QListWidget#Navigation::item:hover {{ background: {COLORS['slate_850']}; color: {COLORS['text']}; }}
QListWidget#Readiness, QListWidget#Artifacts {{
    background: transparent;
    border: 0;
    outline: none;
}}
QListWidget#Readiness::item, QListWidget#Artifacts::item {{ padding: 4px 2px; }}
QScrollArea#EvidenceFeatureScroll, QWidget#EvidenceFeatureGrid {{
    background: transparent;
    border: 0;
}}
QPushButton[featureChoice="true"] {{
    background: {COLORS['ink_900']};
    color: {COLORS['muted']};
    border: 1px solid {COLORS['slate_700']};
    border-radius: 7px;
    padding: 8px 9px;
    text-align: left;
    font-size: 12px;
    font-weight: 650;
}}
QPushButton[featureChoice="true"]:hover {{
    background: #162A3E;
    color: {COLORS['text']};
    border-color: #4B6687;
}}
QPushButton[featureChoice="true"]:focus {{
    padding: 7px 8px;
    border: 2px solid {COLORS['aqua']};
}}
QPushButton[featureChoice="true"]:checked {{
    background: #164E63;
    color: {COLORS['text']};
    border: 2px solid {COLORS['aqua']};
    padding: 7px 8px;
}}
QPushButton[featureChoice="true"]:checked:hover {{ background: #185C70; }}
QProgressBar {{
    background: {COLORS['ink_900']};
    border: 1px solid {COLORS['slate_700']};
    border-radius: 6px;
    text-align: center;
    min-height: 14px;
}}
QProgressBar::chunk {{ background: {COLORS['aqua_dark']}; border-radius: 5px; }}
QFrame[stageState="pending"] {{ border: 1px solid {COLORS['slate_700']}; background: {COLORS['ink_900']}; border-radius: 8px; }}
QFrame[stageState="active"] {{ border: 2px solid {COLORS['aqua']}; background: #102C3E; border-radius: 8px; }}
QFrame[stageState="done"] {{ border: 1px solid {COLORS['success']}; background: #0E2A2A; border-radius: 8px; }}
QTableWidget, QTableView, QTreeWidget {{
    background: {COLORS['ink_900']};
    alternate-background-color: {COLORS['slate_850']};
    color: {COLORS['text']};
    border: 1px solid {COLORS['slate_700']};
    gridline-color: {COLORS['slate_700']};
    selection-background-color: #164E63;
}}
QTreeWidget#EvidenceTree {{ outline: none; }}
QTreeWidget#EvidenceTree::item {{ min-height: 24px; padding: 3px 4px; }}
QTreeWidget#EvidenceTree::item:hover {{ background: #162A3E; }}
QTreeWidget#EvidenceTree::item:selected {{ background: #164E63; color: {COLORS['text']}; }}
QHeaderView::section {{
    background: {COLORS['slate_800']};
    color: {COLORS['muted']};
    border: 0;
    border-right: 1px solid {COLORS['slate_700']};
    border-bottom: 1px solid {COLORS['slate_700']};
    padding: 7px;
    font-weight: 700;
}}
QTabWidget::pane {{ border: 1px solid {COLORS['slate_700']}; border-radius: 7px; }}
QTabBar::tab {{ background: {COLORS['ink_900']}; color: {COLORS['muted']}; padding: 8px 14px; }}
QTabBar::tab:selected {{ color: {COLORS['aqua']}; border-bottom: 2px solid {COLORS['aqua']}; }}
QScrollBar:vertical {{ background: {COLORS['ink_900']}; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {COLORS['slate_700']}; border-radius: 5px; min-height: 28px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ background: {COLORS['ink_900']}; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {COLORS['slate_700']}; border-radius: 5px; min-width: 28px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
"""


def apply_theme(application: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["ink_950"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Base, QColor(COLORS["ink_900"]))
    palette.setColor(QPalette.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.Button, QColor(COLORS["slate_800"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Highlight, QColor(COLORS["aqua_dark"]))
    palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    application.setPalette(palette)
    application.setStyleSheet(STYLESHEET)
