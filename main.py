from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow, QTextEdit, QPushButton, QLabel, QFrame
from PyQt6.QtGui import QTextCursor
from gui.round_toggle_switch import CustomRoundToggleSwitch
from gui.modular_slider import ModularSlider


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        self.setWindowTitle('Neural Network Training')
        self.setMinimumSize(QSize(500, 500))
        self.setStyleSheet('background-color: #282b30')
        
        parameters_layout = QVBoxLayout()

        # Create instances of ModularSlider with different values
        slider1 = ModularSlider('Learning rate', 0, 1000, float)
        parameters_layout.addWidget(slider1)

        slider2 = ModularSlider('Batch size', 0, 1024, int)
        parameters_layout.addWidget(slider2)

        slider3 = ModularSlider('Number of epochs', 0, 1000, int)
        parameters_layout.addWidget(slider3)

        # Toggle switch (checkbok)
        label_early_stopping = QLabel('Early stopping')
        label_early_stopping.setStyleSheet('color: #fff; font-size: 18px')
        parameters_layout.addWidget(label_early_stopping)

        # Actual checkbox with styling
        custom_round_toggle_switch = CustomRoundToggleSwitch('Early stopping toggle')
        parameters_layout.addWidget(custom_round_toggle_switch)

        slider4 = ModularSlider('Early stopping patience', 0, 10, int)
        parameters_layout.addWidget(slider4)

        parameters_layout.setContentsMargins(200,20,200,20)    
        layout.addLayout(parameters_layout)

        

        # Create a QTextEdit widget for displaying text
        self.text_edit = QTextEdit(self)
        self.text_edit.setStyleSheet('color: #fff')
        self.text_edit.setReadOnly(True)  # Set read-only mode
        layout.addWidget(self.text_edit)

        # Create frame for buttom
        button_frame = QFrame(self)
        layout.addWidget(button_frame)
        button_frame_layout = QVBoxLayout(button_frame)
        # Create a QPushButton to trigger the function
        self.btn = QPushButton('Run Function', self)
        self.btn.setStyleSheet('background-color: #489BE8; color: #000; border-radius: 10px; padding: 10px;')
        self.btn.clicked.connect(self.runFunction)
        
        button_frame_layout.addWidget(self.btn)
        button_frame_layout.setContentsMargins(500, 10, 500, 10)

        self.setCentralWidget(central_widget)


    def runFunction(self):
        # Example function that adds text to the QTextEdit widget
        for i in range(30):
            text_to_add = f'Step {i + 1}: This is some text.\n'
            self.text_edit.append(text_to_add)
            # Scroll to the bottom after adding text
            cursor = self.text_edit.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.text_edit.setTextCursor(cursor)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
