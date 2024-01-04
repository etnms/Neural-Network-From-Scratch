from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFrame


class Button(QFrame):
    def __init__(self, name: str, function):
        super().__init__()

        button_frame_layout = QVBoxLayout(self)
        
        # Create a QPushButton to trigger the function
        self.btn = QPushButton(name, self)
        self.btn.setStyleSheet('background-color: #489BE8; color: #000; border-radius: 10px; padding: 10px;')
        self.btn.clicked.connect(function)
        
        button_frame_layout.addWidget(self.btn)
        button_frame_layout.setContentsMargins(500, 10, 500, 10)