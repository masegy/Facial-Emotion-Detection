# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main.ui'
#
# Created by: PyQt5 UI code generator 5.15.0
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1159, 560)
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setGeometry(QtCore.QRect(0, -30, 1161, 631))
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setStyleSheet("#tab{\n"
"background-image: url(:/newPrefix/background.png);\n"
"}")
        self.tab.setObjectName("tab")
        self.startButton = QtWidgets.QPushButton(self.tab)
        self.startButton.setGeometry(QtCore.QRect(70, 450, 121, 31))
        self.startButton.setStyleSheet("#startButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(46, 192, 182);\n"
"    border: 0px solid;\n"
"    font-family:Ubuntu;\n"
"    font-size:15px;\n"
"    border-radius:5px;\n"
"}\n"
"#startButton:hover:pressed{    \n"
"    background-color: rgb(46, 192, 182);\n"
"}")
        self.startButton.setObjectName("startButton")
        self.tabWidget.addTab(self.tab, "")
        self.tabCon = QtWidgets.QWidget()
        self.tabCon.setStyleSheet("#tabCon{\n"
"background-color: rgb(255, 255, 255);\n"
"}")
        self.tabCon.setObjectName("tabCon")
        self.frame = QtWidgets.QFrame(self.tabCon)
        self.frame.setGeometry(QtCore.QRect(0, 0, 171, 601))
        self.frame.setStyleSheet("background-color: rgb(1, 22, 39);")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.camButton = QtWidgets.QPushButton(self.frame)
        self.camButton.setGeometry(QtCore.QRect(30, 30, 111, 41))
        self.camButton.setStyleSheet("#camButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(1, 22, 39);\n"
"    border: 0px solid;\n"
"    font-family:Ubuntu;\n"
"    font-size:15px;\n"
"}\n"
"#camButton:hover:pressed{    \n"
"    background-color: rgb(46, 192, 182);\n"
"}")
        self.camButton.setObjectName("camButton")
        self.haarButton = QtWidgets.QPushButton(self.frame)
        self.haarButton.setGeometry(QtCore.QRect(30, 90, 111, 41))
        self.haarButton.setStyleSheet("#haarButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(1, 22, 39);\n"
"    border: 0px solid;\n"
"    font-family:Ubuntu;\n"
"    font-size:15px;\n"
"}\n"
"#haarButton:hover:pressed{    \n"
"    background-color: rgb(46, 192, 182);\n"
"}")
        self.haarButton.setObjectName("haarButton")
        self.modButton = QtWidgets.QPushButton(self.frame)
        self.modButton.setGeometry(QtCore.QRect(30, 150, 111, 41))
        self.modButton.setStyleSheet("#modButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(1, 22, 39);\n"
"    border: 0px solid;\n"
"    font-family:Ubuntu;\n"
"    font-size:15px;\n"
"}\n"
"#modButton:hover:pressed{    \n"
"    background-color: rgb(46, 192, 182);\n"
"}")
        self.modButton.setObjectName("modButton")
        self.monButton = QtWidgets.QPushButton(self.frame)
        self.monButton.setGeometry(QtCore.QRect(30, 490, 111, 41))
        self.monButton.setStyleSheet("#monButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(46, 192, 182);\n"
"    border: 0px solid;\n"
"    font-family:Ubuntu;\n"
"    font-size:15px;\n"
"    border-radius:5px;\n"
"}\n"
"#monButton:hover:pressed{    \n"
"    background-color: rgb(46, 192, 182);\n"
"}")
        self.monButton.setObjectName("monButton")
        self.processButton = QtWidgets.QPushButton(self.frame)
        self.processButton.setGeometry(QtCore.QRect(30, 210, 111, 41))
        font = QtGui.QFont()
        font.setFamily("Ubuntu")
        font.setPointSize(-1)
        self.processButton.setFont(font)
        self.processButton.setStyleSheet("#processButton{\n"
"    color: rgb(255, 255, 255);\n"
"    background-color: rgb(1, 22, 39);\n"
"    border: 0px solid;\n"
"    font-family:Ubuntu;\n"
"    font-size:15px;\n"
"}\n"
"#processButton:hover:pressed{    \n"
"    background-color: rgb(46, 192, 182);\n"
"}")
        self.processButton.setObjectName("processButton")
        self.camLabel = QtWidgets.QLabel(self.tabCon)
        self.camLabel.setGeometry(QtCore.QRect(180, 20, 641, 541))
        self.camLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.camLabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.camLabel.setLineWidth(1)
        self.camLabel.setText("")
        self.camLabel.setObjectName("camLabel")
        self.probLabel = QtWidgets.QLabel(self.tabCon)
        self.probLabel.setGeometry(QtCore.QRect(830, 20, 321, 271))
        self.probLabel.setFrameShape(QtWidgets.QFrame.Box)
        self.probLabel.setFrameShadow(QtWidgets.QFrame.Raised)
        self.probLabel.setLineWidth(1)
        self.probLabel.setText("")
        self.probLabel.setObjectName("probLabel")
        self.label = QtWidgets.QLabel(self.tabCon)
        self.label.setGeometry(QtCore.QRect(850, 310, 291, 101))
        self.label.setStyleSheet("font: 20pt \"Helvetica\";\n"
"color:rgb(1, 22, 39);")
        self.label.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.tabCon)
        self.label_2.setGeometry(QtCore.QRect(850, 430, 291, 51))
        self.label_2.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_2.setStyleSheet("font: 24pt \"Helvetica\";\n"
"color: rgb(46, 192, 182);")
        self.label_2.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.tabWidget.addTab(self.tabCon, "")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Pengenalan Emosi Wajah"))
        self.startButton.setText(_translate("Form", "Mulai"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("Form", "Tab 1"))
        self.camButton.setText(_translate("Form", "Rekam"))
        self.haarButton.setText(_translate("Form", "Haarcascade"))
        self.modButton.setText(_translate("Form", "Model"))
        self.monButton.setText(_translate("Form", "Monitoring"))
        self.processButton.setText(_translate("Form", "Processing"))
        self.label.setText(_translate("Form", "<html><head/><body><p align=\"center\">Emosi</p><p align=\"center\">Dominan:</p></body></html>"))
        self.label_2.setText(_translate("Form", "<html><head/><body><p align=\"center\"><br/></p></body></html>"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tabCon), _translate("Form", "Tab 2"))
import resources


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
