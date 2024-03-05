"""  
Spectrum Instrumentation GmbH (c) 2024

99_dds_graphical_user_interface.py

Simple GUI - This example shows the DDS functionality with 23 carriers with a GUI that can use the different DDS features

Example for analog replay cards (AWG) for the the M4i and M4x card-families with installed DDS option.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm

import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QCheckBox, QDoubleSpinBox
from PyQt5 import uic

def exec_click():
    global ui, dds, channels

    enable = False
    freq_time_div = ui.frequencyTimeDivider.value()
    amp_time_div = ui.amplitudeTimeDivider.value()

    if len(channels) > 1:
        conn_ch1 = ui.buttonGroup_01.checkedButton().text()
        if not conn_ch1 == "Channel 0":
            dds.cores_on_channel(1, 
                                spcm.SPCM_DDS_CORE8,  spcm.SPCM_DDS_CORE9,  spcm.SPCM_DDS_CORE10, spcm.SPCM_DDS_CORE11, # Flex core block 8 - 11
                                spcm.SPCM_DDS_CORE20) # Fixed core 20
        else:
            dds.cores_on_channel(1, spcm.SPCM_DDS_CORE20) # Fixed core 20
        
    if len(channels) > 2:
        conn_ch2 = ui.buttonGroup_02.checkedButton().text()
        if not conn_ch2 == "Channel 0":
            dds.cores_on_channel(2,
                                spcm.SPCM_DDS_CORE12, spcm.SPCM_DDS_CORE13, spcm.SPCM_DDS_CORE14, spcm.SPCM_DDS_CORE15, # Flex core block 12 - 15
                                spcm.SPCM_DDS_CORE21) # Fixed core 21
        else:
            dds.cores_on_channel(2, spcm.SPCM_DDS_CORE21) # Fixed core 21
    
    if len(channels) > 2:
        conn_ch3 = ui.buttonGroup_03.checkedButton().text()
        if not conn_ch3 == "Channel 0":
            dds.cores_on_channel(3,
                                spcm.SPCM_DDS_CORE16, spcm.SPCM_DDS_CORE17, spcm.SPCM_DDS_CORE18, spcm.SPCM_DDS_CORE19, # Flex core block 16 - 19
                                spcm.SPCM_DDS_CORE22) # Fixed core 22
        else:
            dds.cores_on_channel(3, spcm.SPCM_DDS_CORE22) # Fixed core 22

    if dds:
        dds.freq_ramp_stepsize(freq_time_div)
        dds.amp_ramp_stepsize(amp_time_div)

    for core in dds:
        enable = ui.findChild(QCheckBox, "enable_{}".format(core.index)).isChecked()
        freq_MHz = ui.findChild(QDoubleSpinBox, "frequency_{}".format(core.index)).value()
        amp = float(ui.findChild(QDoubleSpinBox, "amplitude_{}".format(core.index)).value())
        phas_deg = float(ui.findChild(QDoubleSpinBox, "phase_{}".format(core.index)).value())
        freq_slope_MHz_s = float(ui.findChild(QDoubleSpinBox, "frequencySlope_{}".format(core.index)).value())
        amp_slope_1_s = float(ui.findChild(QDoubleSpinBox, "amplitudeSlope_{}".format(core.index)).value())

        if enable:
            print("Frequency {}: {} MHz".format(core.index, freq_MHz))
            print("Amplitude {}: {}".format(core.index, amp))
            print("Phase {}: {} deg".format(core.index, phas_deg))
            print("Frequency slope {}: {} MHz/s".format(core.index, freq_slope_MHz_s))
            print("Amplitude slope {}: {} 1/s".format(core.index, amp_slope_1_s))
        
        if enable:
            core.amp(amp)
            core.freq(freq_MHz * pow(10, 6))
            core.phase(phas_deg)
            core.frequency_slope(freq_slope_MHz_s * pow(10,6))
            core.amplitude_slope(amp_slope_1_s)
        else:
            core.amp(0)

    if dds:
        dds.exec_now()
        dds.write_to_card()


class gui_window(QMainWindow):
    def __init__(self):
        global channels
        super(gui_window, self).__init__()
        uic.loadUi("{}/99_dds_graphical_user_interface.ui".format(os.path.dirname(os.path.abspath(__file__))),self)
        
        if len(channels) < 4:
            self.groupBox_HW3.setEnabled(False)
            self.radioButton_03_3.setEnabled(False)
        if len(channels) < 3:
            self.groupBox_HW2.setEnabled(False)
            self.radioButton_02_2.setEnabled(False)
        if len(channels) < 2:
            self.groupBox_HW1.setEnabled(False)
            self.radioButton_01_1.setEnabled(False)

card : spcm.Card
# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:             # if you want to open the first card of a specific type

    # setup card for DDS
    card.card_mode(spcm.SPC_REP_STD_DDS)

    # Setup the card
    channels = spcm.Channels(card)
    channels.enable(True)
    channels.amp(1000) # 1000 mV
    card.write_setup()

    # Setup DDS
    dds = spcm.DDS(card)
    dds.reset()

    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_NONE)
    dds.exec_at_trg()
    dds.write_to_card()

    # Start the card
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    # Start the test
    app = QApplication(sys.argv)

    ui = gui_window()
    ui.pushButton.clicked.connect(exec_click)
    ui.show()

    try:
        sys.exit(app.exec_())
    except:
        print("Exiting")