"""
Spectrum Instrumentation GmbH (c)

4_trg_acq_channel_trigger.py

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
    # The following card families support special clock mode 22xx and 44xx
    card_family = card.family()
    print(f"Card family {card_family:02x}xx")

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
    trigger.or_mask(spcm.SPC_TMASK_NONE)
    trigger.ch_or_mask0(channel0.ch_mask())
    
    trigger_modes = {
        1: [spcm.SPC_TM_POS, 0.1 * units.V, None, None], # trigger on positive edge of channel signal
        2: [spcm.SPC_TM_NEG, 0.1 * units.V, None, None], # trigger on negative edge of channel signal
        3: [spcm.SPC_TM_BOTH, 0.1 * units.V, None, None], # trigger on both edges of channel signal
        4: [spcm.SPC_TM_POS | spcm.SPC_TM_REARM, 0.1 * units.V, 0.0 * units.V, None], # re-arm the trigger with a positive slope through level1 and trigger on positive edge of channel signal through level0
        5: [spcm.SPC_TM_NEG | spcm.SPC_TM_REARM, 0.1 * units.V, -0.1 * units.V, None], # re-arm the trigger with a negative slope through level1 and trigger on negative edge of channel signal through level0
        6: [spcm.SPC_TM_POS | spcm.SPC_TM_PW_GREATER, 0.1 * units.V, None, 1 * units.us], # Channel pulse width trigger for positive pulses that are above the threshold for more than the pulse width
        7: [spcm.SPC_TM_NEG | spcm.SPC_TM_PW_GREATER, 0.1 * units.V, None, 1 * units.us], # Channel pulse width trigger for negative pulses that are below the threshold for more than the pulse width
        8: [spcm.SPC_TM_POS | spcm.SPC_TM_PW_SMALLER, 0.1 * units.V, None, 100 * units.us], # Channel pulse width trigger for positive pulses that are above the threshold for less than the pulse width
        9: [spcm.SPC_TM_NEG | spcm.SPC_TM_PW_SMALLER, 0.1 * units.V, None, 100 * units.us], # Channel pulse width trigger for negative pulses that are below the threshold for less than the pulse width
        10: [spcm.SPC_TM_WINENTER, 0.1 * units.V, 0.0 * units.V, None], # trigger on entering a window defined by level0 and level1
        11: [spcm.SPC_TM_WINLEAVE, 0.1 * units.V, 0.0 * units.V, None], # trigger on leaving a window defined by level0 and level1
        12: [spcm.SPC_TM_WINENTER | spcm.SPC_TM_PW_GREATER, 0.1 * units.V, 0.0 * units.V, 0.5 * units.us], # trigger on entering a window defined by level0 and level1 and staying there longer than the pulse width
        13: [spcm.SPC_TM_WINLEAVE | spcm.SPC_TM_PW_GREATER, 0.1 * units.V, 0.0 * units.V, 0.5 * units.us], # trigger on leaving a window defined by level0 and level1 and staying outside longer than the pulse width
        14: [spcm.SPC_TM_WINENTER | spcm.SPC_TM_PW_SMALLER, 0.1 * units.V, 0.0 * units.V, 100 * units.us], # trigger on entering a window defined by level0 and level1 and staying there shorter than the pulse width
        15: [spcm.SPC_TM_WINLEAVE | spcm.SPC_TM_PW_SMALLER, 0.1 * units.V, 0.0 * units.V, 100 * units.us], # trigger on leaving a window defined by level0 and level1 and staying outside shorter than the pulse width
        }
    trigger_modes_str = {
        1:  " 1: Trigger on positive edge of channel signal",
        2:  " 2: Trigger on negative edge of channel signal",
        3:  " 3: Trigger on both edges of channel signal",
        4:  " 4: Trigger on positive edge of channel signal with re-arm",
        5:  " 5: Trigger on negative edge of channel signal with re-arm",
        6:  " 6: Channel pulse width trigger for positive pulses that are above the threshold for more than the pulse width",
        7:  " 7: Channel pulse width trigger for negative pulses that are below the threshold for more than the pulse width",
        8:  " 8: Channel pulse width trigger for positive pulses that are above the threshold for less than the pulse width",
        9:  " 9: Channel pulse width trigger for negative pulses that are below the threshold for less than the pulse width",
        10: "10: Trigger on entering a window defined by level0 and level1",
        11: "11: Trigger on leaving a window defined by level0 and level1",
        12: "12: Trigger on entering a window defined by level0 and level1 and staying there longer than the pulse width",
        13: "13: Trigger on leaving a window defined by level0 and level1 and staying outside longer than the pulse width",
        14: "14: Trigger on entering a window defined by level0 and level1 and staying there shorter than the pulse width",
        15: "15: Trigger on leaving a window defined by level0 and level1 and staying outside shorter than the pulse width",
    }
    trigger_modes_families = {
        1: [0x22, 0x44, 0x59],
        2: [0x22, 0x44, 0x59],
        3: [0x22, 0x44, 0x59],
        4: [0x22, 0x44, 0x59],
        5: [0x22, 0x44, 0x59],
        6: [0x59],
        7: [0x59],
        8: [0x59],
        9: [0x59],
        10: [0x22, 0x44, 0x59],
        11: [0x22, 0x44, 0x59],
        12: [0x59],
        13: [0x59],
        14: [0x59],
        15: [0x59]
    }

    question_str = "Please select one of the following trigger modes by entering the corresponding number and press <ENTER>:\n"
    for mode, description in trigger_modes_str.items():
        if card_family in trigger_modes_families[mode]:
            question_str += f"{description}\n"
        else:
            question_str += f"{description} (not supported by card family {card_family:02x}xx)\n"
    selected_trigger_mode = input(question_str)

    try:
        tm = int(selected_trigger_mode)
    except ValueError:
        print("Invalid input. Stopping execution.")
        exit()

    if card_family not in trigger_modes_families[tm]:
        print(f"Trigger mode {tm} is not supported by card family {card_family:02x}xx. Stopping execution.")
        exit()

    # Re-arm the trigger when crossing through level1 and then trigger when a signal on the channel input crosses 500 mV from below to above (positive slope)
    trigger.ch_mode(channel0, trigger_modes[tm][0]) # re-arm the trigger with a positive slope through level1 and trigger on positive edge of channel input through level0
    trigger.ch_level0(channel0, trigger_modes[tm][1])  # trigger level for channel input
    if trigger_modes[tm][2] is not None: trigger.ch_level1(channel0, trigger_modes[tm][2])  # re-arm level for channel input
    if trigger_modes[tm][3] is not None: trigger.ch_pulsewidth(channel0, trigger_modes[tm][3])  # pulse width for the window trigger
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
