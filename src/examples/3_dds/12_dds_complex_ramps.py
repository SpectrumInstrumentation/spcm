"""  
Spectrum Instrumentation GmbH (c) 2024

12_dds_complex_ramps.py

Complex ramping multiple carriers - Ramping the frequency of 20 carriers from a one setting to another using s-shaped ramps

Example for analog replay cards (AWG) for the the M4i and M4x card-families with installed DDS option.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import numpy as np
import matplotlib.pyplot as plt

# A generator function for s-shaped ramps
def generate_function(t, parameters):
    if parameters["ramp_type"] == 'cosine':
        # cosine
        y = (0.5 - 0.5*np.cos(np.pi*t))
    elif parameters["ramp_type"] == 'square':
        # square
        a = np.concatenate([(0,), np.ones(((num_segments+1)//2,)), -np.ones(((num_segments+1)//2,))])
        v = np.cumsum(a)*parameters["time_s"]/num_segments
        y = np.cumsum(v)*parameters["time_s"]/num_segments
    elif parameters["ramp_type"] == '3rd-order':
        # 3rd order
        b = 4*np.concatenate([(0,), np.ones(((num_segments+1)//4,)), -np.ones(((num_segments+1)//2,)), np.ones(((num_segments+1)//4,))])
        a = np.cumsum(b)*parameters["time_s"]/num_segments
        v = np.cumsum(a)*parameters["time_s"]/num_segments
        y = np.cumsum(v)*parameters["time_s"]/num_segments
    y = parameters["startFreq_Hz"] + y/y[-1] * (parameters["endFreq_Hz"] - parameters["startFreq_Hz"])
    return y

# A generator for get piecewise-linear approximated functions
def calculate_slope(t, y):
    # Slopes along the lines
    t_diff = np.diff(t)
    y_diff = np.diff(y)
    return np.divide(y_diff, t_diff)


card : spcm.Card
# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO) as card:            # if you want to open the first card of a specific type

    # setup card for DDS
    card.card_mode(spcm.SPC_REP_STD_DDS)

    # Setup the channels
    channels = spcm.Channels(card)
    channels.enable(True)
    channels.amp(1 * units.V)
    card.write_setup()
    
    # Setup DDS
    dds = spcm.DDS(card)
    dds.reset()

    # Start the DDS test
    num_cores = len(dds)
    # 20 Carriers from 90 to 110 MHz
    first_init_freq_Hz  = 90 * units.MHz
    delta_init_freq_Hz  =  1 * units.MHz
    # to 20 Carriers from 95 to 105 MHz
    first_final_freq_Hz = 95  * units.MHz
    delta_final_freq_Hz = 500 * units.kHz 
    
    # Ramp settings
    num_segments = 16
    total_time_s = 5.0 * units.s
    ramp_type = 'cosine' # 'cosine' or 'square' or '3rd-order'

    # STEP 0 - Initialize frequencies
    dds.trg_timer(2.0 * units.s)
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
    for core in dds:
        core.amp(45 * units.percent / num_cores)
        core.freq(first_init_freq_Hz + int(core) * delta_init_freq_Hz)
    dds.exec_at_trg()
    dds.write_to_card()

    # STEP 1 - Start the ramp
    period_s = total_time_s / num_segments # seconds
    dds.trg_timer(period_s) # after 5.0 s stop the ramp
    # Define the parameters
    parameters = []
    slopes = np.zeros((num_cores, num_segments))
    # Show the results
    plt.figure(figsize=(7,7))
    for core in dds:
        parameters = {
            "startFreq_Hz": first_init_freq_Hz + core.index * delta_init_freq_Hz, 
            "endFreq_Hz": first_final_freq_Hz + core.index * delta_final_freq_Hz, 
            "time_s": total_time_s, 
            "ramp_type": ramp_type
            }

        # Define the function
        t = np.linspace(0, 1, num_segments+1, endpoint=True)
        y = generate_function(t, parameters)

        t_s = t * parameters["time_s"]
        # points = np.array([t_s, y]).T
        sl_core = calculate_slope(t_s, y)
        slopes[core, :] = sl_core.to_base_units().magnitude


        plt.plot(t_s, y, 'ok')
        t_fine_s = np.linspace(t_s[0], t_s[1], 2, endpoint=True)
        for j, sl in enumerate(sl_core):
            plt.plot(t_s[j] + t_fine_s, y[j] + sl*(t_fine_s), '--')
        
    # plt.legend()
    # plt.xlabel('t(s)')
    # plt.ylabel('y(Hz)')

    plt.show(block=False)

    # Do the slopes
    for j in range(num_segments):
        for core in dds:
            core.frequency_slope(slopes[core][j]) # Hz/s
        dds.exec_at_trg()

    # STEP 2 - Stop the ramp
    for core in dds:
        core.frequency_slope(0) # Hz/s
        core.freq(first_final_freq_Hz + core.index * delta_final_freq_Hz)
    dds.exec_at_trg()
    dds.write_to_card()

    # Start command including enable of trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Press Enter to Exit")


