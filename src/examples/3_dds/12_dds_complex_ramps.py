"""  
Spectrum Instrumentation GmbH (c) 2024

12_dds_complex_ramps.py

Complex ramping multiple carriers - Ramping the frequency of 20 carriers from a one setting to another using s-shaped ramps

Example for analog replay cards (AWG) for the the M4i and M4x card-families with installed DDS option.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
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
def calculate_slope(points):
    # Slopes along the lines
    difference = np.diff(points, axis=0)
    return np.divide(difference[:,1], difference[:,0])


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
    channels.amp(1000) # 1000 mV
    card.write_setup()
    
    # Setup DDS
    dds = spcm.DDS(card)
    dds.reset()

    # Start the DDS test
    num_freq = dds.num_cores()
    # 20 Carriers from 90 to 110 MHz
    first_init_freq_Hz  = 90e6
    delta_init_freq_Hz  =  1e6
    # to 20 Carriers from 95 to 105 MHz
    first_final_freq_Hz = 95e6
    delta_final_freq_Hz =  5e5
    
    # Ramp settings
    num_segments = 16
    total_time_s = 5.0 # seconds
    ramp_type = 'cosine' # 'cosine' or 'square' or '3rd-order'

    # STEP 0 - Initialize frequencies
    dds.trg_timer(2.0)
    dds.trg_src(spcm.SPCM_DDS_TRG_SRC_TIMER)
    for i in range(num_freq):
        dds.amp(i, 0.45 / num_freq)
        dds.freq(i, first_init_freq_Hz + i * delta_init_freq_Hz)
    dds.exec_at_trg()
    dds.write_to_card()

    # STEP 1 - Start the ramp
    period_s = total_time_s / num_segments # seconds
    dds.trg_timer(period_s) # after 5.0 s stop the ramp
    # Define the parameters
    parameters = []
    slopes = np.zeros((num_freq, num_segments))
    # Show the results
    plt.figure(figsize=(7,7))
    for i in range(num_freq):
        parameters = {
            "startFreq_Hz": first_init_freq_Hz + i * delta_init_freq_Hz, 
            "endFreq_Hz": first_final_freq_Hz + i * delta_final_freq_Hz, 
            "time_s": total_time_s, 
            "ramp_type": ramp_type
            }

        # Define the function
        t = np.linspace(0, 1, num_segments+1, endpoint=True)
        y = generate_function(t, parameters)

        t_s = t * parameters["time_s"]
        points = np.array([t_s, y]).T
        slopes[i, :] = calculate_slope(points)


        plt.plot(*points.T, 'ok')
        t_fine_s = np.linspace(t_s[0], t_s[1], 2, endpoint=True)
        for j, sl in enumerate(slopes[i]):
            plt.plot(t_s[j] + t_fine_s, y[j] + sl*(t_fine_s), '--')
        
    # plt.legend()
    plt.xlabel('t(s)')
    plt.ylabel('y(Hz)')

    plt.show(block=False)

    # Do the slopes
    for j in range(num_segments):
        for i in range(num_freq):
            dds.frequency_slope(i, slopes[i][j]) # Hz/s
        dds.exec_at_trg()

    # STEP 2 - Stop the ramp
    for i in range(num_freq):
        dds.frequency_slope(i, 0) # Hz/s
        dds.freq(i, first_final_freq_Hz + i * delta_final_freq_Hz)
    dds.exec_at_trg()
    dds.write_to_card()

    # Start command including enable of trigger engine
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_FORCETRIGGER)

    input("Press Enter to Exit")


