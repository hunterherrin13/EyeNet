import sys
import threading
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
import time

class WorkerThread(threading.Thread):
    def __init__(self, function):
        super().__init__()
        self.function = function
        self.running = True

    def run(self):
        self.function()

    def stop(self):
        self.running = False

    def is_running(self):
        return self.running

class GUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Detection_NET GUI - HUNTER HERRIN")
        self.setMinimumSize(400, 300)  # Set minimum size
        self.setMaximumSize(800, 600)  # Set maximum size

        layout = QVBoxLayout()
        self.setLayout(layout)

        # Button Rows
        button_layout1 = QHBoxLayout()
        button_layout2 = QHBoxLayout()
        layout.addLayout(button_layout1)
        layout.addLayout(button_layout2)

        # Create buttons with associated functions
        self.buttons = []  # Store buttons for later use
        for i in range(5):
            button = QPushButton(f"Button {i+1}")
            button.setStyleSheet(get_button_style())
            button.clicked.connect(self.create_thread(self.functions[i]))
            button_layout1.addWidget(button)
            self.buttons.append(button)

        # Terminate Functions Button
        self.terminate_button = QPushButton("Terminate Functions")
        self.terminate_button.setStyleSheet(get_button_style())
        self.terminate_button.clicked.connect(self.terminate_threads)
        button_layout2.addWidget(self.terminate_button)

        # Image Display
        self.image_label = QLabel()
        layout.addWidget(self.image_label)
        layout.setAlignment(Qt.AlignCenter)

        # Worker Threads
        self.worker_threads = []

    def create_thread(self, func):
        def wrapper():
            thread = WorkerThread(func)
            self.worker_threads.append(thread)
            thread.start()
        return wrapper

    def terminate_threads(self):
        for thread in self.worker_threads:
            thread.stop()
        self.update_button_style()  # Update button style
        self.show_warning_message()  # Show warning message

    def update_button_style(self):
        for button in self.buttons + [self.terminate_button]:  # Include the terminate button
            button.setStyleSheet(get_button_style_red())

    def show_warning_message(self):
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Warning)
        msg.setText("SW kernel must be reset.")
        msg.setWindowTitle("Warning")
        msg.exec_()

    def function1(self):
        # Placeholder function for Button 1
        for i in range(1000):
            if not self.worker_threads[0].is_running():  # Check if thread should stop
                break
            print("Function 1")
            time.sleep(1)

    def function2(self):
        # Placeholder function for Button 2
        print("Function 2")

    def function3(self):
        # Placeholder function for Button 3
        print("Function 3")

    def function4(self):
        # Placeholder function for Button 4
        print("Function 4")

    def function5(self):
        # Placeholder function for Button 5
        print("Function 5")

    @property
    def functions(self):
        return [self.function1, self.function2, self.function3, self.function4, self.function5]


def get_button_style():
    return """
    QPushButton {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    QPushButton:hover {
        background-color: #45a049;
    }
    """

def get_button_style_red():
    return """
    QPushButton {
        background-color: #FF0000; /* Red */
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
    }
    QPushButton:hover {
        background-color: #FF3333; /* Lighter red */
    }
    """

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())