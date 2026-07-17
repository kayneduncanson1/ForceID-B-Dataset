from __future__ import print_function
import os
import sys
import time
from datetime import datetime
import numpy as np
import pandas as pd
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from vicon_dssdk import ViconDataStream

"""This script was used to run the GUI to acquire GRF data using the Vicon DataStream SDK. The Python (conda)
environment .yml file supplied in this repository currently does not contain the vicon_dssdk module. Running the script
requires installing the vicon_dssdk module using pip (see Vicon DataStream SDK documentation at
https://help.vicon.com/space/DSSDK112)."""


class MyWindow(QMainWindow):

    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.msgBox_PDError = QtWidgets.QMessageBox(self)
        self.msgBox_NotSaved = QtWidgets.QMessageBox(self)
        self.setGeometry(500, 50, 970, 970)
        self.setWindowTitle("ForceID-Study-3 GUI")
        self.initUI()
        self.setStyleSheet("QMainWindow {background-color: black}")

    def initUI(self):

        # Create a box containing a detailed workflow for the user:
        self.groupBox = QtWidgets.QGroupBox(self)
        self.groupBox.setTitle("Participant details")
        self.groupBox.setFont(QFont('Verdana', 20, QFont.Bold))
        self.groupBox.setAlignment(100)
        self.groupBox.setStyleSheet("color: white")
        self.groupBox.setGeometry(20, 10, 950, 950)
        self.groupBox.setObjectName("groupBox")

        # Create a mini-box with a field to enter the participant's ID number:
        self.ID_No = QtWidgets.QLabel(self.groupBox)
        self.ID_No.setText("ID number")
        self.ID_No.setFont(QFont('Verdana', 12, QFont.Bold))
        self.ID_No.setStyleSheet("color: white")
        self.ID_No.setGeometry(20, 100, 300, 100)
        self.ID_No.setObjectName("ID_No")

        self.Enter_ID = QtWidgets.QLineEdit(self.groupBox)
        self.Enter_ID.setGeometry(600, 100, 300, 100)
        self.Enter_ID.setFont(QFont('Verdana', 12, QFont.Bold))
        self.Enter_ID.setStyleSheet("background-color: black")
        self.Enter_ID.setObjectName("Enter_ID")

        # Create a mini-box with a field to enter the participant's session number:
        self.SesNo = QtWidgets.QLabel(self.groupBox)
        self.SesNo.setText("Session number")
        self.SesNo.setFont(QFont('Verdana', 12, QFont.Bold))
        self.SesNo.setStyleSheet("color: white")
        self.SesNo.setGeometry(QtCore.QRect(20, 250, 150, 100))
        self.SesNo.setWordWrap(True)
        self.SesNo.setObjectName("Ses_No")

        self.Enter_SesNo = QtWidgets.QLineEdit(self.groupBox)
        self.Enter_SesNo.setGeometry(QtCore.QRect(600, 250, 300, 100))
        self.Enter_SesNo.setFont(QFont('Verdana', 12, QFont.Bold))
        self.Enter_SesNo.setStyleSheet("background-color: black")
        self.Enter_SesNo.setObjectName("Enter_Ses_No")

        # Create a mini-box with a field to enter the participant's body mass with and without any carried object(s)
        # in kilograms:
        self.Body_Mass_Object = QtWidgets.QLabel(self.groupBox)
        self.Body_Mass_Object.setText("Body mass - object (kg)")
        self.Body_Mass_Object.setFont(QFont('Verdana', 12, QFont.Bold))
        self.Body_Mass_Object.setGeometry(QtCore.QRect(20, 400, 500, 100))

        self.Enter_BodyMass_Object = QtWidgets.QLineEdit(self.groupBox)
        self.Enter_BodyMass_Object.setGeometry(QtCore.QRect(600, 400, 300, 100))
        self.Enter_BodyMass_Object.setFont(QFont('Verdana', 12, QFont.Bold))
        self.Enter_BodyMass_Object.setStyleSheet("background-color: black")
        self.Enter_BodyMass_Object.setObjectName("Enter_BodyMass")

        self.Body_Mass = QtWidgets.QLabel(self.groupBox)
        self.Body_Mass.setText("Body mass - no object (kg)")
        self.Body_Mass.setFont(QFont('Verdana', 12, QFont.Bold))
        self.Body_Mass.setGeometry(QtCore.QRect(20, 550, 550, 100))

        self.Enter_BodyMass = QtWidgets.QLineEdit(self.groupBox)
        self.Enter_BodyMass.setGeometry(QtCore.QRect(600, 550, 300, 100))
        self.Enter_BodyMass.setFont(QFont('Verdana', 12, QFont.Bold))
        self.Enter_BodyMass.setStyleSheet("background-color: black")
        self.Enter_BodyMass.setObjectName("Enter_BodyMass")

        # Create a mini-box with a field to enter the participant's height in meters:
        self.Participant_Height = QtWidgets.QLabel(self.groupBox)
        self.Participant_Height.setText("Height (m)")
        self.Participant_Height.setFont(QFont('Verdana', 12, QFont.Bold))
        self.Participant_Height.setGeometry(QtCore.QRect(20, 700, 300, 100))

        self.Enter_Height = QtWidgets.QLineEdit(self.groupBox)
        self.Enter_Height.setGeometry(QtCore.QRect(600, 700, 300, 100))
        self.Enter_Height.setFont(QFont('Verdana', 12, QFont.Bold))
        self.Enter_Height.setStyleSheet("background-color: black")
        self.Enter_Height.setObjectName("Enter_Height")

        # Create a button to save participant details and initialize the GUI for data acquisition:
        self.Button_Initialize = QtWidgets.QPushButton(self.groupBox)
        self.Button_Initialize.setGeometry(QtCore.QRect(170, 825, 300, 100))
        self.Button_Initialize.setFont(QFont('Verdana', 15, QFont.Bold))
        self.Button_Initialize.setStyleSheet("QPushButton {background-color: darkgreen; text-align: center}")
        self.Button_Initialize.setObjectName("Button_Initialize")
        self.Button_Initialize.setText("Save details")
        self.Button_Initialize.clicked.connect(self.initialize)

        # Create a RECORD button to acquire force platform data:
        self.Button_RECORD = QtWidgets.QPushButton(self)
        self.Button_RECORD.setGeometry(QtCore.QRect(600, 835, 200, 100))
        self.Button_RECORD.setFont(QFont('Verdana', 15, QFont.Bold))
        self.Button_RECORD.setStyleSheet("QPushButton {background-color: black; text-align: center}")
        self.Button_RECORD.setObjectName("Button_RECORD")
        self.Button_RECORD.setIcon(QIcon('RECORD.jpg'))
        self.Button_RECORD.setIconSize(QtCore.QSize(199, 99))
        self.Button_RECORD.clicked.connect(self.record)

    def initialize(self):

        data_export = [self.Enter_ID.text(), ',', self.Enter_SesNo.text(), ',', self.Enter_BodyMass_Object.text(), ',',
                       self.Enter_BodyMass.text(), ',', self.Enter_Height.text()]

        # Apply constraints on field entries for session details. Namely:
        # - ID number contains 3 digits.
        # - Session number must be from 1-5 (based on our data acquisition period of five days).
        # - Body mass without object must be in the 10s or 100s and be to one decimal place.
        # - Body mass with object must meet the above criteria or be 'NA'.
        # - For the first session, a height measurement must be entered. Height entry must be four units long (including
        #   the decimal) and comprise a single digit followed by two decimal places (e.g., 1.78).
        # - For subsequent sessions, the height entry must be 'NA' as a height measurement is not taken according to the
        #   experimental protocol.
        if len(self.Enter_ID.text()) != 3 \
                or self.Enter_SesNo.text() not in ['1', '2', '3', '4', '5'] \
                or len(self.Enter_BodyMass.text()) not in [4, 5] \
                or self.Enter_BodyMass.text()[-2] != '.' \
                or ((len(self.Enter_BodyMass_Object.text()) not in [4, 5]
                     or self.Enter_BodyMass_Object.text()[-2] != '.')
                    and self.Enter_BodyMass_Object.text() != 'NA') \
                or (self.Enter_SesNo.text() == '1'
                    and (len(self.Enter_Height.text()) != 4
                         or self.Enter_Height.text()[1] != '.')) \
                or (self.Enter_SesNo.text() in ['2', '3', '4', '5'] and self.Enter_Height.text() != 'NA'):

            # If any of the above constraints are not met, create an error message box:
            self.msgBox = QtWidgets.QMessageBox(self)
            self.msgBox.setIcon(QMessageBox.Critical)
            self.msgBox.setWindowTitle("Error Dialog")
            self.msgBox.setStyleSheet("color: black")
            self.msgBox.setFont((QFont('Verdana', 10, QFont.Bold)))
            self.msgBox.setText("Error")
            self.msgBox.setInformativeText("Participant details are incomplete\n\n"
                                           "You must enter all details to proceed")

            # Provide further detail on the error(s) in the message box:
            if len(self.Enter_ID.text()) != 3:

                ID_error = "ID number must contain 3 digits e.g. 014"
                x = [ID_error]
                msg = '\n'.join(x)
                self.msgBox.setDetailedText(str(msg))

            else:

                ID_error = ""

            if self.Enter_SesNo.text() not in ['1', '2', '3', '4', '5']:

                Ses_No_Error = "Session no. must be between 1 and 5"
                x = [ID_error, Ses_No_Error]
                msg = '\n'.join(x)
                self.msgBox.setDetailedText(str(msg))

            else:

                Ses_No_Error = ""

            if len(self.Enter_BodyMass.text()) not in [4, 5] \
                    or self.Enter_BodyMass.text()[-2] != '.':

                BodyMass_error = "Body mass without object must be provided (to one decimal place)"
                x = [ID_error, Ses_No_Error, BodyMass_error]
                msg = '\n'.join(x)
                self.msgBox.setDetailedText(str(msg))

            else:

                BodyMass_error = ""

            if (len(self.Enter_BodyMass_Object.text()) not in [4, 5]
                or self.Enter_BodyMass_Object.text()[-2] != '.') \
                    and self.Enter_BodyMass_Object.text() != 'NA':

                BodyMass_Object_error = "Body mass with object must be either NA or provided (to one decimal place)"
                x = [ID_error, Ses_No_Error, BodyMass_error, BodyMass_Object_error]
                msg = '\n'.join(x)
                self.msgBox.setDetailedText(str(msg))

            else:

                BodyMass_Object_error = ""

            if self.Enter_SesNo.text() == '1' \
                    and (len(self.Enter_Height.text()) != 4
                         or self.Enter_Height.text()[1] != '.'):

                Height_error_1 = "Height must be provided for session 1 (to 2 decimal places)"
                x = [ID_error, Ses_No_Error, BodyMass_error, BodyMass_Object_error, Height_error_1]
                msg = '\n'.join(x)
                self.msgBox.setDetailedText(str(msg))

            else:

                Height_error_1 = ""

            if self.Enter_SesNo.text() in ['2', '3', '4', '5'] and self.Enter_Height.text() != 'NA':
                Height_error_2 = "Height not needed for sessions 2-5 - enter NA"
                x = [ID_error, Ses_No_Error, BodyMass_error, BodyMass_Object_error, Height_error_1, Height_error_2]
                msg = '\n'.join(x)
                self.msgBox.setDetailedText(str(msg))

            self.msgBox.exec()

        # If session details satisfy the constraints:
        else:

            # Change the current working directory to the path where GRF data will be saved:
            os.chdir('./Datasets/fi-b-all/ID dirs')

            # Create a participant ID folder/directory:
            folder = self.Enter_ID.text()

            # If the ID directory exists:
            if os.path.exists(folder):

                # Make it the working directory:
                os.chdir(folder)

            # Otherwise, if the ID directory doesn't exist:
            else:

                # Make the ID directory and then set it as the working directory:
                os.mkdir(folder)
                os.chdir(folder)

            # Make a file name for a txt file containing the session details:
            fname = '_'.join([self.Enter_ID.text(), 'S' + self.Enter_SesNo.text()])

            # If the file already exists:
            if os.path.isfile(fname + '.txt'):

                # Create an error message box to notify that the file already exists and avoid overwrite:
                QtWidgets.QMessageBox.critical(self, 'Error', 'Details file already exists for this session -'
                                                              'double check ID and session no.')

            # If the file doesn't exist:
            else:

                # Create the file and write the session details:
                with open(fname + '.txt', 'w') as text_file:

                    text_file.writelines(data_export)

                # Notify the operator of successful initialization, reminding them of the object condition order if
                # applicable:
                if self.Enter_BodyMass_Object.text() != 'NA':

                    print('Details saved. Record when ready. Record in this order:\n'
                          '1. WITH object(s)\n'
                          '2. WITH object(s)\n'
                          '3. WITHOUT object(s)\n'
                          '4. WITHOUT object(s)')

                else:

                    print('Details saved, record when ready.')

    def record(self):

        # Get the datetime stamp of recording start:
        dt = datetime.now().strftime("%y_%m_%d-%H_%M_%S")

        # Initialize the Vicon Datastream SDK client:
        client = ViconDataStream.Client()

        start = time.time()
        end = time.time()
        timeElapsed = end - start
        recordLength = 20 # seconds

        frames = []
        trial_data_fp1 = []
        trial_data_fp2 = []
        trial_data_fp3 = []
        # occludeds_all = []

        # Connect to the client:
        print('Connecting')
        while not client.IsConnected():
            print('.')
            client.Connect('localhost:801')

        try:
            while client.IsConnected() and timeElapsed < recordLength:

                client.SetBufferSize(1)
                client.EnableDeviceData()
                client.SetStreamMode(ViconDataStream.Client.StreamMode.EServerPush)

                end = time.time()
                timeElapsed = end - start

                # If the client can get the frame from the stream:
                if client.GetFrame():

                    # Record the frame number:
                    frames.append(client.GetFrameNumber())

                    # Get the device names to distinguish between force platforms:
                    devices = client.GetDeviceNames()

                    # For each force platform:
                    for idx, (deviceName, deviceType) in enumerate(devices):

                        # Get the output:
                        deviceOutputDetails = client.GetDeviceOutputDetails(deviceName)

                        frame_data = []
                        # occludeds = [] # For occluded frames.

                        # For each channel (i.e., component) in the output:
                        for outputName, componentName, unit in deviceOutputDetails:

                            # Get the measured values:
                            FP_values, occluded = client.GetDeviceOutputValues(deviceName, outputName,
                                                                               componentName)
                            # occludeds.append(occluded)

                            if componentName != 'Cz':

                                frame_data.append(FP_values)

                        # The order of force platforms in the set of devices from the client was FP2, FP3, FP1:
                        if idx == 0:

                            trial_data_fp2.append(frame_data)

                        elif idx == 1:

                            trial_data_fp3.append(frame_data)

                        else:

                            trial_data_fp1.append(frame_data)

                        # occludeds_all.append(occludeds)

        except ViconDataStream.DataStreamException as e:
            print("Error")

        # Convert the data from each force platform to an array and reshape to T x C, where T = number of time steps
        # (i.e., frames) and C = number of channels = 8 (Fx, Fy, Fz, Mx, My, Mz, Cx and Cy):
        trial_data_fp1 = np.array(trial_data_fp1).transpose((0, 2, 1)).reshape(-1, 8)
        trial_data_fp2 = np.array(trial_data_fp2).transpose((0, 2, 1)).reshape(-1, 8)
        trial_data_fp3 = np.array(trial_data_fp3).transpose((0, 2, 1)).reshape(-1, 8)

        # Combine the data from each force platform along the channel axis such that there are 3 x C = 24 columns in the
        # final data object to be written to a csv file:
        trial_data = np.concatenate((trial_data_fp1, trial_data_fp2, trial_data_fp3), axis=1)
        df = pd.DataFrame(trial_data)

        # Create a file name with the ID no., session no. and date time stamp. Note that trial numbers were assigned
        # (based on metadata) and added to file names after data acquisition when preliminarily organizing the dataset:
        fname = '_'.join([self.Enter_ID.text(), 'S' + self.Enter_SesNo.text(), dt])

        # Write the data object to a csv file:
        df.to_csv(fname + '.csv')

        print("Complete with " + str(trial_data.shape[0]) + " frames recorded over " + str(timeElapsed) + " seconds.")


if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)
    data = MyWindow()

    def window():

        app = QApplication(sys.argv)
        win = MyWindow()
        win.show()
        sys.exit(app.exec_())

    window()
