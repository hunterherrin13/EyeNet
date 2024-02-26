import sys
import threading
import time
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

class WorkerThread(threading.Thread):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        self.running = True

    def run(self):
        while self.running:
            time.sleep(1)
            print("Background Task")

    def stop(self):
        self.running = False

class GUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Multithreading GUI")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.title_label = QLabel("Image Viewer")
        layout.addWidget(self.title_label)

        # Button Rows
        button_layout1 = QHBoxLayout()
        button_layout2 = QHBoxLayout()
        layout.addLayout(button_layout1)
        layout.addLayout(button_layout2)

        for i in range(3):
            button_layout1.addWidget(QPushButton(f"Button {i+1}"))
            button_layout2.addWidget(QPushButton(f"Button {i+4}"))

        # Image Display
        self.image_label = QLabel()
        pixmap = QPixmap("image.jpg")  # Replace "image.jpg" with your image file
        self.image_label.setPixmap(pixmap)
        layout.addWidget(self.image_label)
        layout.setAlignment(Qt.AlignCenter)

        # Start Worker Thread
        self.worker_thread = WorkerThread(self)
        self.worker_thread.start()

    def closeEvent(self, event):
        self.worker_thread.stop()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = GUI()
    gui.show()
    sys.exit(app.exec_())