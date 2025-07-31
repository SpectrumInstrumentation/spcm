"""
Spectrum Instrumentation GmbH (c)

2_trg_acq_external_trigger.py

Shows a simple Standard acquisition (recording) mode example using an external trigger.

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import numpy as np
import matplotlib.pyplot as plt


card : spcm.Card

with spcm.Card('/dev/spcm0') as card: # if you want to open a specific card
    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_SINGLE)     # single trigger standard mode
    card.timeout(5 * units.s)                     # timeout 5 s

    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL) # clock mode internal PLL
    clock.sample_rate(10 * units.percent)
    
    # setup the channels
    channel0, = spcm.Channels(card, card_enable=spcm.CHANNEL0) # enable channel 0
    channel0.amp(1000 * units.mV)

    ### Trigger setup section ###
    trigger = spcm.Trigger(card, clock=clock)
    trigger.or_mask(spcm.SPC_TMASK_EXT0) # or SPC_TMASK_EXT[1-4] (if available) for external trigger

    selected_trigger_mode = input("There are several trigger modes available. Please select one of the following modes by entering the corresponding number and press <ENTER>:\n" \
    "1: Trigger on positive edge of external trigger\n" \
    "2: Trigger on negative edge of external trigger\n" \
    "3: Trigger on both edges of external trigger\n" \
    "4: Trigger on positive edge of external trigger with re-arm\n"
    "5: Trigger on negative edge of external trigger with re-arm\n" \
    "6: Trigger on entering a window defined by level0 and level1\n" \
    "7: Trigger on leaving a window defined by level0 and level1\n")

    try:
        tm = int(selected_trigger_mode)
    except ValueError:
        print("Invalid input. Defaulting to trigger on positive edge of external trigger.")
        tm = 1

    if tm == 1:
        # Trigger when a signal on the external trigger input crosses 500 mV from below to above (positive slope)
        trigger.ext0_mode(spcm.SPC_TM_POS) # trigger on positive edge of external trigger
        trigger.ext0_level0(0.5 * units.V)  # trigger level for external trigger
    elif tm == 2:
        # Trigger when a signal on the external trigger input crosses 500 mV from above to below (negative slope)
        trigger.ext0_mode(spcm.SPC_TM_NEG) # trigger on negative edge of external trigger
        trigger.ext0_level0(0.5 * units.V)  # trigger level for external trigger
    elif tm == 3:
        # Trigger when a signal on the external trigger input crosses 500 mV from above to below (negative slope) and from below to above (positive slope)
        trigger.ext0_mode(spcm.SPC_TM_BOTH) # trigger on negative and positive edge of external trigger
        trigger.ext0_level0(0.5 * units.V)  # trigger level for external trigger
    elif tm == 4:
        # Re-arm the trigger when crossing through level1 and then trigger when a signal on the external trigger input crosses 500 mV from below to above (positive slope)
        trigger.ext0_mode(spcm.SPC_TM_POS | spcm.SPC_TM_REARM) # re-arm the trigger with a positive slope through level1 and trigger on positive edge of external trigger through level0
        trigger.ext0_level0(0.5 * units.V)  # trigger level for external trigger
        trigger.ext0_level1(0.0 * units.V)  # re-arm level for external trigger
    elif tm == 5:
        # Re-arm the trigger when crossing through level1 and then trigger when a signal on the external trigger input crosses 0 mV from above to below (negative slope)
        trigger.ext0_mode(spcm.SPC_TM_NEG | spcm.SPC_TM_REARM) # re-arm the trigger with a negative slope through level1 and trigger on negative edge of external trigger through level0
        trigger.ext0_level0(0.1 * units.V)  # re-arm level for external trigger
        trigger.ext0_level1(-0.1 * units.V)  # trigger level for external trigger
    elif tm == 6:
        # Window trigger for entering signals.
        trigger.ext0_mode(spcm.SPC_TM_WINENTER) # trigger on entering a window defined by level0 and level1
        trigger.ext0_level0(0.5 * units.V)  # upper level for the window trigger
        trigger.ext0_level1(-0.5 * units.V)  # lower level for the window trigger
    elif tm == 7:
        # Window trigger for leaving signals.
        trigger.ext0_mode(spcm.SPC_TM_WINLEAVE) # trigger on leaving a window defined by level0 and level1
        trigger.ext0_level0(0.5 * units.V)  # upper level for the window trigger
        trigger.ext0_level1(-0.5 * units.V)  # lower level for the window trigger
    else:
        print("Invalid input. Stopping execution.")
        exit()
    #############################

    # define the data buffer
    data_transfer = spcm.DataTransfer(card)
    num_samples = 512 * units.S
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.post_trigger(num_samples // 2)
    
    # start card and wait until recording is finished
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)

    print("Finished acquiring...")

    # Start DMA transfer and wait until the data is transferred
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)

    # Plot the acquired data
    time_data_s = data_transfer.time_data()
    fig, ax = plt.subplots()
    unit_data_V = channel0.convert_data(data_transfer.buffer[channel0, :], units.V)
    print(channel0)
    print("\tMinimum: {:.3~P}".format(np.min(unit_data_V)))
    print("\tMaximum: {:.3~P}".format(np.max(unit_data_V)))
    ax.plot(time_data_s, unit_data_V, label=f"{channel0}")
    ax.yaxis.set_units(units.mV)
    ax.xaxis.set_units(units.us)
    ax.axvline(0, color='k', linestyle='--', label='Trigger')
    ax.legend()
    plt.show()
