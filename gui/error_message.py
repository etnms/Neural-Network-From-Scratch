from PyQt6.QtWidgets import QMessageBox

def show_error_message(error_message):
        error_box = QMessageBox()
        error_box.setIcon(QMessageBox.Icon.Critical)
        error_box.setWindowTitle("Error")
        error_box.setText("An error occurred:")
        error_box.setInformativeText(error_message)
        error_box.exec()