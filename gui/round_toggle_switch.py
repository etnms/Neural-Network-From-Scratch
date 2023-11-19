from PyQt6.QtWidgets import QCheckBox

from PyQt6.QtGui import QPainter, QColor, QBrush

class CustomRoundToggleSwitch(QCheckBox):
    def __init__(self, text):
        super().__init__(text)
        self.setFixedSize(60, 34)  # Set the fixed size to match the HTML/CSS example
        self.setChecked(False)

    def paintEvent(self, event):
        # Custom painting logic
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Define switch colors
        background_color = QColor(204, 204, 204)  # #ccc
        slider_color = QColor(255, 255, 255)  # white
        checked_color = QColor(114, 137, 218)  # #7289da

        # Draw the background
        #painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(background_color))
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 17, 17)

        # Draw the slider
        slider_position = self.width() - 34 if self.isChecked() else 0
        painter.setBrush(QBrush(checked_color if self.isChecked() else slider_color))
        painter.drawRoundedRect(slider_position, 0, 34, self.height(), 17, 17)

    def mousePressEvent(self, event):
        # Toggle the checkbox state when clicked
        self.setChecked(not self.isChecked())