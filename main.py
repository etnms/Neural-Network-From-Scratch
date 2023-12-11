from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QMainWindow, QTextEdit, QPushButton, QLabel, QFrame
from gui.round_toggle_switch import CustomRoundToggleSwitch
from gui.modular_slider import ModularSlider
from model.model import Model
from testing import layers, training_set_X, training_set_y, testing_set_X, testing_set_y
import numpy as np
from gui.layer_section import DynamicSection

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        central_widget = QWidget()
        self.layout = QVBoxLayout(central_widget)
        self.setWindowTitle('Neural Network Training')
        self.setMinimumSize(QSize(500, 500))
        self.setStyleSheet('background-color: #282b30')
        
        # Add section widget for layer creations
        self.add_button = QPushButton('Add Section')
        self.add_button.clicked.connect(self.add_section)
        self.layout.addWidget(self.add_button)
        # Keeping track of layers
        self.sections = []

        # Parameters and parameters layout
        parameters_layout = QVBoxLayout()

        # Create instances of ModularSlider with different values
        self.learning_rate= ModularSlider('Learning rate', 0, 1000, float) 
        parameters_layout.addWidget(self.learning_rate)

        self.batch_size = ModularSlider('Batch size', 0, 1024, int)
        parameters_layout.addWidget(self.batch_size)

        self.num_epochs = ModularSlider('Number of epochs', 0, 1000, int)
        parameters_layout.addWidget(self.num_epochs)

        # Toggle switch (checkbok)
        label_early_stopping = QLabel('Early stopping')
        label_early_stopping.setStyleSheet('color: #fff; font-size: 18px')
        parameters_layout.addWidget(label_early_stopping)

        # Actual checkbox with styling
        self.early_stopping = CustomRoundToggleSwitch('Early stopping toggle')
        parameters_layout.addWidget(self.early_stopping)

        self.early_stopping_patience = ModularSlider('Early stopping patience', 0, 10, int)
        parameters_layout.addWidget(self.early_stopping_patience)

        parameters_layout.setContentsMargins(200,20,200,20)    
        self.layout.addLayout(parameters_layout)

        # Create a QTextEdit widget for displaying text
        self.text_edit = QTextEdit(self)
        self.text_edit.setStyleSheet('color: #fff')
        self.text_edit.setReadOnly(True)  # Set read-only mode
        self.layout.addWidget(self.text_edit)

        # Create frame for buttom
        button_frame = QFrame(self)
        self.layout.addWidget(button_frame)
        button_frame_layout = QVBoxLayout(button_frame)
        # Create a QPushButton to trigger the function
        self.btn = QPushButton('Train model', self)
        self.btn.setStyleSheet('background-color: #489BE8; color: #000; border-radius: 10px; padding: 10px;')
        self.btn.clicked.connect(self.train_model)
        
        button_frame_layout.addWidget(self.btn)
        button_frame_layout.setContentsMargins(500, 10, 500, 10)

        self.text_training = ''
        self.model = Model(layers=layers, update_text_callback=self.update_text_training)
        
        self.setCentralWidget(central_widget)

    def add_section(self):
        # Create a new section and add it to the layout
        section = DynamicSection()
        self.layout.addWidget(section)
        self.sections.append(section)

    def update_text_training(self, new_text):
        self.text_training = new_text
        self.text_edit.append(new_text)

    def train_model(self):
        #model = Model(layers=layers)
        self.model.train_model(batch_size=self.batch_size.value, num_epochs=self.num_epochs.value, 
                          learning_rate=self.learning_rate.value,data_X=training_set_X,
                          data_y=training_set_y, training=True, early_stopping=self.early_stopping.isChecked(), 
                          early_stopping_patience=self.early_stopping_patience.value, regularization='l1')
        predictions = self.model.testing_model(data_X=testing_set_X)

    # For binary classification, the prediction is the index of the maximum value in the last layer's output
        # /!\ need to have something for more than binary classification
        predicted_classes = np.argmax(predictions, axis=1)

        accuracy = np.mean(predicted_classes == testing_set_y)
        print(f"Test accuracy: {accuracy}")
        self.text_edit.append(f"Test accuracy: {accuracy}")

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()
