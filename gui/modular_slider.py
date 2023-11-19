from PyQt6.QtWidgets import QSlider, QVBoxLayout, QWidget, QLabel, QLineEdit
from PyQt6.QtCore import Qt, pyqtSlot
import os


class ModularSlider(QWidget):
    def __init__(self, name, min_value, max_value, value_type=float, parent=None):
        super().__init__(parent)


        # Set the current working directory to the script's directory
        script_dir = os.path.dirname(os.path.realpath(__file__))
        os.chdir(script_dir)

        with open('slider_styles.qss', 'r') as file:
                stylesheet = file.read()
        self.setStyleSheet(stylesheet)
        
        layout = QVBoxLayout(self)

        self.slider = QSlider()
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.slider.setMinimum(min_value)
        self.slider.setMaximum(max_value)
        self.slider.setSingleStep(1)

        self.label = QLabel(f'{name} Value: 0')
        layout.addWidget(self.label)

        self.slider.valueChanged.connect(lambda value: self.update_value_label(value))

        layout.addWidget(self.slider)

        self.input_box = QLineEdit(self)
        self.input_box.setPlaceholderText(f'Enter {name} value')
        self.input_box.returnPressed.connect(self.update_slider_from_input)
        layout.addWidget(self.input_box)

        self.value_type = value_type



    @pyqtSlot()
    def update_slider_from_input(self):
        try:
            input_value = self.input_box.text()

            if self.value_type == float:
                value = float(input_value)
                normalized_value = min(max(0.0, value), 1.0) # Ensure the value is between 0 and 1
                slider_value = int(normalized_value * self.slider.maximum())
            elif self.value_type == int:
                slider_value = int(input_value)
                slider_value = max(self.slider.minimum(), min(slider_value, self.slider.maximum()))
            else:
                return  # Unsupported value_type, do nothing

            self.slider.setValue(slider_value)
            self.label.setText(f'{self.label.text().split(":")[0]}: {slider_value}')
        
        except ValueError:
            # Handle the case where the input is not a valid float or int
            pass
       
    def update_value_label(self, value):
        value = value / 1000
        self.label.setText(f'{self.label.text().split(":")[0]}: {value}')