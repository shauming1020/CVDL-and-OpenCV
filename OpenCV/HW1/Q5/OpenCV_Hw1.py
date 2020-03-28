# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'CVDL_Hw1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(327, 300)
        self.hw5_1 = QtWidgets.QPushButton(Form)
        self.hw5_1.setGeometry(QtCore.QRect(60, 38, 201, 21))
        self.hw5_1.setObjectName("hw5_1")
        self.hw5_2 = QtWidgets.QPushButton(Form)
        self.hw5_2.setGeometry(QtCore.QRect(60, 78, 201, 21))
        self.hw5_2.setObjectName("hw5_2")
        self.hw5_3 = QtWidgets.QPushButton(Form)
        self.hw5_3.setGeometry(QtCore.QRect(60, 118, 201, 21))
        self.hw5_3.setObjectName("hw5_3")
        self.hw5_4 = QtWidgets.QPushButton(Form)
        self.hw5_4.setGeometry(QtCore.QRect(60, 158, 201, 23))
        self.hw5_4.setObjectName("hw5_4")
        self.hw5_5 = QtWidgets.QPushButton(Form)
        self.hw5_5.setGeometry(QtCore.QRect(60, 218, 201, 23))
        self.hw5_5.setObjectName("hw5_5")
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setGeometry(QtCore.QRect(150, 188, 113, 20))
        self.lineEdit.setObjectName("lineEdit")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(60, 190, 91, 20))
        self.label.setObjectName("label")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.hw5_1.setText(_translate("Form", "5.1 Show Train Images"))
        self.hw5_2.setText(_translate("Form", "5.2 Show Hyperparameters"))
        self.hw5_3.setText(_translate("Form", "5.3 Train 1 Epoch"))
        self.hw5_4.setText(_translate("Form", "5.4 Show Training Result"))
        self.hw5_5.setText(_translate("Form", "5.5 Inference"))
        self.lineEdit.setText(_translate("Form", "(0~9999)"))
        self.label.setText(_translate("Form", "Test Image Index:"))

