from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QMainWindow, QTextEdit, QPushButton, QLabel, QScrollArea, QFileDialog
from gui.round_toggle_switch import CustomRoundToggleSwitch
from gui.modular_slider import ModularSlider
from model.model import Model
import numpy as np
from gui.layer_section import DynamicSection
from gui.button import Button
from layer.create_modular_layers import ModularLayer
from split_training_data import split_training_data
from load_csv_data import load_csv_data
from gui.error_message import show_error_message


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__() # PyQT super 

        # List of layers for the model, empty at first
        self.layers_features, self.layers_neurons, self.layers_activation  = [], [], []

        central_widget = QWidget()
        
        # Open file button
        self.layout = QVBoxLayout(central_widget)
        self.setWindowTitle('Neural Network Training')
        self.setMinimumSize(QSize(500, 500))
        self.setStyleSheet('background-color: #282b30')
        self.file_button = QPushButton('Open file')
        self.file_button.setStyleSheet('background-color: #489BE8; color: #000; border-radius: 10px; padding: 10px;')
        self.file_button.clicked.connect(self.open_file_path)
        self.file_path = ''
        self.layout.addWidget(self.file_button)

        # Parameters and parameters layout
        self.parameters_layout = QVBoxLayout()

        # Create instances of ModularSlider with different values
        self.learning_rate= ModularSlider('Learning rate', 0, 1000, float) 
        self.parameters_layout.addWidget(self.learning_rate)

        self.batch_size = ModularSlider('Batch size', 0, 1024, int)
        self.parameters_layout.addWidget(self.batch_size)

        self.num_epochs = ModularSlider('Number of epochs', 0, 1000, int)
        self.parameters_layout.addWidget(self.num_epochs)

        # Toggle switch (checkbox)
        label_early_stopping = QLabel('Early stopping')
        label_early_stopping.setStyleSheet('color: #fff; font-size: 18px')
        self.parameters_layout.addWidget(label_early_stopping)

        # Actual checkbox with styling
        self.early_stopping = CustomRoundToggleSwitch('Early stopping toggle')
        self.parameters_layout.addWidget(self.early_stopping)

        self.early_stopping_patience = ModularSlider('Early stopping patience', 0, 10, int)
        self.parameters_layout.addWidget(self.early_stopping_patience)

        self.parameters_layout.setContentsMargins(0,20,0,20)    

        # Layout for layers and layer creation
        self.layer_layout = QVBoxLayout()

        # Add section widget for layer creations
        self.add_button = QPushButton('Add Section')
        self.add_button.setStyleSheet('background-color: #489BE8; color: #000; border-radius: 10px; padding: 10px;')
        self.add_button.clicked.connect(self.add_section)
        self.layer_layout.addWidget(self.add_button)

        # Wrap layer_layout in a QScrollArea
        layer_scroll_area = QScrollArea()
        layer_scroll_area.setWidgetResizable(True)
        layer_scroll_area.setWidget(QWidget())  # Dummy widget for content

        # Set the layout of the dummy widget to layer_layout
        layer_scroll_area.widget().setLayout(self.layer_layout)
        # Layout for top section with all hyperparameters (parameters and layers)
        self.hyperparameters_layout = QHBoxLayout()
        self.hyperparameters_layout.addLayout(self.parameters_layout)
        self.hyperparameters_layout.addWidget(layer_scroll_area)

        # Keeping track of layers
        self.sections = []

        self.layout.addLayout(self.hyperparameters_layout)

        # Create a QTextEdit widget for displaying text
        self.text_edit = QTextEdit(self)
        self.text_edit.setStyleSheet('color: #fff')
        self.text_edit.setReadOnly(True)  # Set read-only mode
        self.layout.addWidget(self.text_edit)

        self.btn_train = Button('Train model', self.train_model)
        self.layout.addWidget(self.btn_train)

        self.btn_save = Button('Save model', self.save_model)
        self.layout.addWidget(self.btn_save)

        self.text_training = ''
        
        self.setCentralWidget(central_widget)

    def add_section(self):
        # Create a new section and add it to the layout
        section = DynamicSection(self)
        self.layer_layout.addWidget(section)
        self.sections.append(section)
        section.remove_section_signal.connect(self.remove_section)
    
    def remove_section(self, section):
        # Remove the specified section from the layout and the sections list
        self.layout.removeWidget(section)
        section.deleteLater()
        self.sections.remove(section)

    def update_text_training(self, new_text):
        self.text_training = new_text
        self.text_edit.append(new_text)

    def train_model(self):
        # Remove all layers for next user trials of the model
        self.empty_layers()
        self.model = None
        try:
            training = True
            loss_function_used = None
            self.createLayer()
            X, y, number_features, number_classes = load_csv_data(self.file_path)
            print(f'Number of features: {number_features}')
            print(f'Number of classes: {number_classes}')
            training_set_X, training_set_y, testing_set_X, testing_set_y = split_training_data(X, y, training_size=0.8)
            self.model = Model(self.layers, update_text_callback=self.update_text_training)
            self.model.train_model(self.num_epochs.value,self.batch_size.value, self.learning_rate.value,training_set_X,
                            training_set_y, training,loss_function_used, self.early_stopping.isChecked(), self.early_stopping_patience.value, 
                            None, plot_loss=True)
            predictions = self.model.testing_model(data_X=testing_set_X)

            predicted_classes = np.argmax(predictions, axis=1)
            # Min class labels for dataset that don't start at 0 since they will run an index error
            min_class_label = np.min(testing_set_y)
            accuracy = np.mean(predicted_classes == (testing_set_y - min_class_label))
            print(f"Test accuracy: {accuracy}")
            self.text_edit.append(f"Test accuracy: {accuracy}")
        except Exception as e:
            error_message = str(e)
            show_error_message(error_message)

    def createLayer(self):
        try:
            self.layers_features = [int(section.edit_x.text()) for section in self.sections]
            self.layers_neurons = [int(section.edit_y.text()) for section in self.sections]
            self.layers_activation = [section.activation_dropdown.currentData() for section in self.sections]

        except ValueError:
            print('Invalid value')

        self.layers = ModularLayer.create_modular_layers(self.layers_features, self.layers_neurons, self.layers_activation)

    def empty_layers(self):
        self.layers_features = []
        self.layers_neurons = []
        self.layers_activation = []

    def save_model(self):
        try:
            self.model.save_model(self.layers, self.early_stopping.isChecked(), self.early_stopping_patience.value, None, name='model') 
        except Exception as e:
            error_message = str(e)
            show_error_message(error_message)

    def open_file_path(self):
        try:
            # Open a file dialog to select a path
            selected_path, _ = QFileDialog.getOpenFileName(self, "Select Path", "", "CSV Files (*.csv)")

            if selected_path:
                self.file_path = selected_path
                # Do something with the selected path (e.g., print it)
                print("Selected Path:", selected_path)
        except Exception as e:
            error_message = str(e)
            show_error_message(error_message)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()