"""
Spectrum Instrumentation GmbH (c)

5_trg_acq_channel_trigger_gated.py

Shows a simple Standard acquisition (recording) mode example using an external trigger for the gate signal.

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

#TODO: Need to test this with a REAL card, not a demo card. There are problems with memory handling on the demo card.

import spcm
from spcm import units

import matplotlib.pyplot as plt


card : spcm.Card

with spcm.Card('/dev/spcm4', verbose=True) as card:                         # if you want to open a specific card
    
    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_GATE)
    card.timeout(5 * units.s)                     # timeout 5 s
    
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL) # clock mode internal PLL
    clock.sample_rate(10 * units.percent)
    
    # setup the channels
    channel0, = spcm.Channels(card, card_enable=spcm.CHANNEL0) # enable channel 0
    channel0.amp(1000 * units.mV)

    ### Trigger setup section ###
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_NONE)
    trigger.ch_or_mask0(channel0.ch_mask())

    selected_trigger_mode = input("There are several trigger modes available. Please select one of the following modes by entering the corresponding number and press <ENTER>:\n" \
    "1: Gate signal when channel signal is above the trigger level.\n" \
    "2: Gate signal when channel signal is below the trigger level.\n" \
    "3: Gate signal when channel signal is within the window defined by level0 and level1.\n" \
    "4: Gate signal when channel signal is outside the window defined by level0 and level1.\n" \
    "5: Channel hysteresis trigger on positive edge.\n" \
    "6: Channel hysteresis trigger on negative edge.\n" \
    "7: Channel re-arm hysteresis trigger on positive edge.\n" \
    "8: Channel re-arm hysteresis trigger on negative edge.\n" \
    "9: High level hysteresis trigger.\n" \
    "10: Low level hysteresis trigger.\n")
    try:
        tm = int(selected_trigger_mode)
    except ValueError:
        print("Invalid input. Defaulting to trigger on positive edge of channel signal.")
        tm = 1
    
    if tm == 1:
        # Gate signal generated when a signal on the channel 0 input is higher then 500 mV
        trigger.ch_mode(channel0, spcm.SPC_TM_HIGH) # gate when channel 0 is above the trigger level
        trigger.ch_level0(channel0, 0.5 * units.V)  # trigger level for channel 0
    elif tm == 2:
        # Gate signal generated when a signal on the channel 0 input is lower then 500 mV
        trigger.ch_mode(channel0, spcm.SPC_TM_LOW) # gate when channel 0 is below the trigger level
        trigger.ch_level0(channel0, 0.5 * units.V)  # trigger level for channel 0
    elif tm == 3:
        # Gate signal generated when a signal on the channel 0 input is within the range of 0 mV to 500 mV
        trigger.ch_mode(channel0, spcm.SPC_TM_INWIN) # gate when channel 0 is within the window
        trigger.ch_level0(channel0, 0.5 * units.V)  # upper trigger level of gate window for channel 0
        trigger.ch_level1(channel0, 0.0 * units.V)  # lower trigger level of gate window for channel 0
    elif tm == 4:
        # Gate signal generated when a signal on the channel 0 input is outside the range of 0 mV to 500 mV
        trigger.ch_mode(channel0, spcm.SPC_TM_OUTSIDEWIN) # gate when channel 0 is outside the window
        trigger.ch_level0(channel0, 0.5 * units.V)  # upper trigger level of gate window for channel 0
        trigger.ch_level1(channel0, 0.0 * units.V)  # lower trigger level of gate window for channel 0
    elif tm == 5:
        # Channel hysteresis trigger on positive edge.
        trigger.ch_mode(channel0, spcm.SPC_TM_POS | spcm.SPC_TM_HYSTERESIS) 
        trigger.ch_level0(channel0, 0.5 * units.V)  # start the gate when the channel signal has a positive slope through this level
        trigger.ch_level1(channel0, 0.0 * units.V)  # stop the gate when the channel signal passes through this level
    elif tm == 6:
        # Channel hysteresis trigger on negative edge.
        trigger.ch_mode(channel0, spcm.SPC_TM_NEG | spcm.SPC_TM_HYSTERESIS) 
        trigger.ch_level0(channel0, 0.5 * units.V)  # start the gate when the channel signal has a negative slope through this level
        trigger.ch_level1(channel0, 0.0 * units.V)  # stop the gate when the channel signal passes through this level
    elif tm == 7:
        # Channel re-arm hysteresis trigger on positive edge.
        trigger.ch_mode(channel0, spcm.SPC_TM_POS | spcm.SPC_TM_HYSTERESIS | spcm.SPC_TM_REARM)
        trigger.ch_level0(channel0, 0.5 * units.V)  # start the gate when the channel signal has a positive slope through this level
        trigger.ch_level1(channel0, 0.0 * units.V)  # re-arm the trigger when passing through this level and stop the gate when the channel signal passes through this level
    elif tm == 8:
        # Channel re-arm hysteresis trigger on negative edge.
        trigger.ch_mode(channel0, spcm.SPC_TM_NEG | spcm.SPC_TM_HYSTERESIS | spcm.SPC_TM_REARM)
        trigger.ch_level0(channel0, 0.0 * units.V)  # start the gate when the channel signal has a negative slope through this level
        trigger.ch_level1(channel0, 0.5 * units.V)  # re-arm the trigger when passing through this level and stop the gate when the channel signal passes through this level
    elif tm == 9:
        # High level hysteresis trigger
        trigger.ch_mode(channel0, spcm.SPC_TM_HIGH | spcm.SPC_TM_HYSTERESIS)
        trigger.ch_level0(channel0, 0.5 * units.V)  # start the gate when the channel signal is above this level
        trigger.ch_level1(channel0, 0.0 * units.V)  # stop the gate when the channel signal is below this level
    elif tm == 10:
        # Low level hysteresis trigger
        trigger.ch_mode(channel0, spcm.SPC_TM_LOW | spcm.SPC_TM_HYSTERESIS)
        trigger.ch_level0(channel0, 0.0 * units.V)  # start the gate when the channel signal is below this level
        trigger.ch_level1(channel0, 0.5 * units.V)  # stop the gate when the channel signal is above this level
    else:
        print("Invalid input. Stopping execution.")
        exit()
    #############################

    num_samples = 16 * units.KiS
    max_num_gates = 128 # the maximum number of gates to be acquired

    pre_trigger = 16 * units.S
    post_trigger = 16 * units.S
    # define the data buffer
    data_transfer = spcm.Gated(card, max_num_gates=max_num_gates)
    data_transfer.memory_size(num_samples)
    data_transfer.pre_trigger(pre_trigger)
    data_transfer.post_trigger(post_trigger)
    data_transfer.allocate_buffer(num_samples)
    
    # start card and wait until recording is finished
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)

    print("Finished acquiring...")

    # Start DMA transfer and wait until the data is transferred
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)

    # Plot the acquired data
    fig, ax = plt.subplots()
    for gate in data_transfer:
        time_range = data_transfer.current_time_range(return_unit=units.us)
        gate_start, gate_end = data_transfer.current_timestamps(return_unit=units.us)
        print(f"Gate {data_transfer.iterator_index} - time period: {gate_start} to {gate_end} - number of points: {len(gate[0, :])}")
        unit_data_V = channel0.convert_data(gate[channel0, :], units.V)
        ax.plot(time_range, unit_data_V, '.', label=f"{channel0}")
        ax.axvline(gate_start, color='b', linestyle='-', alpha=0.7, label='Gate')
        ax.axvline(gate_end, color='b', linestyle='-', alpha=0.7, label='Gate')
        ax.axvspan(gate_start, gate_end, alpha=0.1, color='b')
    ax.yaxis.set_units(units.mV)
    plt.show()
