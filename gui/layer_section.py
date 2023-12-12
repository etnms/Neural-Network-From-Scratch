
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QSpacerItem, QSizePolicy
from PyQt6.QtCore import pyqtSignal

'''
UI class to create fields for layer creations to add to the model
'''


class DynamicSection(QWidget):
    remove_section_signal = pyqtSignal(QWidget)

    def __init__(self, main_window):
        super().__init__()
        self.main_window = main_window
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Input fields for each layer
        self.label_x = QLabel('Number of classes:')
        self.edit_x = QLineEdit()

        self.label_y = QLabel('Number of neurons:')
        self.edit_y = QLineEdit()

        # Remove button
        self.remove_button = QPushButton('Remove')
        self.remove_button.clicked.connect(self.remove_section)

        # Layout for input fields
        input_layout = QVBoxLayout()
        input_layout.addWidget(self.label_x)
        input_layout.addWidget(self.edit_x)
        input_layout.addWidget(self.label_y)
        input_layout.addWidget(self.edit_y)

        # Add input layout and remove button to the main layout
        layout.addLayout(input_layout)
        layout.addWidget(self.remove_button)

        # Add a spacer to push the remove button to the right
        spacer_item = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        layout.addItem(spacer_item)

        self.setLayout(layout)

    def remove_section(self):
        # Emit a signal to notify the parent widget to remove this section
        self.remove_section_signal.emit(self)