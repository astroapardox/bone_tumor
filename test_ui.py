from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QMovie
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit, QFileDialog
from PyQt5.QtGui import QIcon
import sys
import os
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report
import numpy as np
import cv2 as cv
import cv2

class Ui_userinterface(object):
    def setupUi(self, userinterface):
        userinterface.setObjectName("userinterface")
        userinterface.resize(661, 415)
        userinterface.setMaximumSize(QtCore.QSize(661, 765))
        userinterface.setLayoutDirection(QtCore.Qt.LeftToRight)
        userinterface.setStyleSheet("*\n""{\n""font: 20pt \"Ebrima\";\n""}\n""#Form{\n""background:rgb(22, 22, 22)\n""}\n""QFrame{\n""background:rgb(22, 22, 22);\n""}\n""")
        self.frame = QtWidgets.QFrame(userinterface)
        self.frame.setGeometry(QtCore.QRect(20, 130, 621, 241))
        self.frame.setStyleSheet("QFrame{\n""background:rgb(22, 22, 22);\n""border-radius:15px;\n""}")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.browse = QtWidgets.QPushButton(self.frame)
        self.browse.setGeometry(QtCore.QRect(430, 30, 131, 41))
        self.browse.setStyleSheet("QPushButton{\n""background:rgb(0, 170, 255);\n""color:#fff;\n""border-radius:15px;\n""}\n""QPushButton:hover{\n""background:rgb(24, 194, 255)\n""}")
        self.browse.setObjectName("browse")
        self.image_line = QtWidgets.QLineEdit(self.frame)
        self.image_line.setGeometry(QtCore.QRect(50, 30, 351, 41))
        self.image_line.setStyleSheet("QLineEdit{\n""border-radius:15px;\n""background:rgb(67, 67, 67);\n""color:rgb(255, 255, 255);\n""}\n""QLineEdit:hover{\n""border: 2px solid rgb(85, 170, 255)\n""}\n""\n""\n""")
        self.image_line.setAlignment(QtCore.Qt.AlignCenter)
        self.image_line.setClearButtonEnabled(False)
        self.image_line.setObjectName("image_line")
        self.tester = QtWidgets.QPushButton(self.frame)
        self.tester.setGeometry(QtCore.QRect(230, 150, 161, 41))
        self.tester.setStyleSheet("QPushButton{\n""background:rgb(0, 170, 255);\n""color:#fff;\n""border-radius:15px;\n""}\n""QPushButton:hover{\n""background:rgb(85, 255, 0)\n""}")
        self.tester.setObjectName("tester")       
        self.background = QtWidgets.QFrame(userinterface)
        self.background.setGeometry(QtCore.QRect(0, 0, 661, 660))
        self.background.setStyleSheet("QFrame{\n""background:rgb(32, 32, 32);\n""}")
        self.background.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.background.setFrameShadow(QtWidgets.QFrame.Raised)
        self.background.setObjectName("background")
        self.frame_2 = QtWidgets.QFrame(self.background)
        self.frame_2.setGeometry(QtCore.QRect(200, 60, 261, 41))
        self.frame_2.setStyleSheet("QFrame{\n""background:rgb(22, 22, 22);\n""border-radius:15px;\n""}")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setGeometry(QtCore.QRect(70, 0, 161, 41))
        self.label.setStyleSheet("color:rgb(189, 189, 189)")
        self.label.setObjectName("label")
        self.background.raise_()
        self.frame.raise_()
        self.retranslateUi(userinterface)
        QtCore.QMetaObject.connectSlotsByName(userinterface)
        self.browse.clicked.connect(self.openFileNameDialog)
        self.tester.clicked.connect(self.Test)
        
    def openFileNameDialog(self):           
        filename = QFileDialog.getOpenFileName()
        self.path = filename[0]
        if self.path:
                self.image_line.setText(filename[0])

    def Test(self):    
        BS = 8
        data = []
        new_model = tf.keras.models.load_model('tumor.model')
        im = cv.imread(self.path)
        image = cv.cvtColor(im, cv.COLOR_BGR2RGB)
        image = cv.resize(image, (224, 224))
        data.append(image)
        data = np.array(data) / 255.0

        predIdxs = new_model.predict(data)
        prob_normal = predIdxs[0][1] * 100;
        prob_cancer  = predIdxs[0][0] * 100;

        print("Probabilite d'etre sain: %.2f" % prob_normal)
        print("Probabilite du Cancer: %.2f" % prob_cancer)

        if prob_normal > 40 : 
                image = cv.resize(image, (600, 600))
                text = "{:.2f}%".format(prob_normal)
                cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                cv2.imshow("Sain", image)
                cv.waitKey()
        else: 
                image = cv.resize(image, (600, 600))
                text = "{:.2f}%".format(prob_cancer)
                cv2.putText(image, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
                cv2.imshow("Cancer",image)
                cv.waitKey()

    def retranslateUi(self, userinterface):
        _translate = QtCore.QCoreApplication.translate
        userinterface.setWindowTitle(_translate("userinterface", "Prediction"))
        self.browse.setText(_translate("userinterface", "Parcourir"))
        self.image_line.setPlaceholderText(_translate("userinterface", "Image a tester"))
        self.tester.setText(_translate("userinterface", "Tester !"))
        self.label.setText(_translate("userinterface", "Prediction"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    userinterface = QtWidgets.QWidget()
    ui = Ui_userinterface()
    ui.setupUi(userinterface)
    userinterface.show()
    sys.exit(app.exec_())