import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel

app = QApplication(sys.argv)
window = QMainWindow()
window.setWindowTitle("PyQt5 Test")
label = QLabel("Hello, PyQt5!", window)
window.setCentralWidget(label)
window.resize(400, 300)
window.show()
sys.exit(app.exec_())