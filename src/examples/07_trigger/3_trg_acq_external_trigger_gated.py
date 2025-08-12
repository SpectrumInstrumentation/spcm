"""
Spectrum Instrumentation GmbH (c)

3_trg_acq_external_trigger_gated.py

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

with spcm.Card('/dev/spcm0', verbose=True) as card:                         # if you want to open a specific card
    # The following card families support special clock mode 22xx and 44xx
    card_family = card.family()
    print(f"Card family {card_family:02x}xx")

    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_GATE)
    card.timeout(50 * units.s)                     # timeout 50 s

    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL) # clock mode internal PLL
    sample_rate = clock.sample_rate(max=True, return_unit=units.MHz)
    print(f"Sample rate set to {sample_rate}.")
    
    # setup the channels
    channel0, = spcm.Channels(card, card_enable=spcm.CHANNEL0) # enable channel 0
    if card_family in [0x22, 0x44]:
        channel0.coupling(spcm.COUPLING_DC)  # set channel 0 coupling to DC
    if card_family in [0x44, 0x59]:
        channel0.termination(1) # set the termination to 50 Ohm for 44xx or 59xx cards
    channel0.amp(1000 * units.mV)

    ### Trigger setup section ###
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_EXT0) # or SPC_TMASK_EXT[1-4] (if available) for external trigger
    
    trigger_modes = {
        1: [spcm.SPC_TM_HIGH, 0.1 * units.V, None], # gate signal when external trigger is above the trigger level
        2: [spcm.SPC_TM_LOW, 0.1 * units.V, None], # gate signal when external trigger is below the trigger level
        3: [spcm.SPC_TM_INWIN, 0.1 * units.V, -0.1 * units.V], # gate signal when external trigger is within the window defined by level0 and level1
        4: [spcm.SPC_TM_OUTSIDEWIN, 0.1 * units.V, 0.0 * units.V] # gate signal when external trigger is outside the window defined by level0 and level1
    }
    trigger_modes_str = {
        1: "1: Gate signal when external trigger is above the trigger level",
        2: "2: Gate signal when external trigger is below the trigger level",
        3: "3: Gate signal when external trigger is within the window defined by level0 and level1",
        4: "4: Gate signal when external trigger is outside the window defined by level0 and level1"
    }
    trigger_modes_families = {
        1: [0x22, 0x44, 0x59],
        2: [0x22, 0x44, 0x59],
        3: [0x22, 0x44],
        4: [0x22, 0x44],
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

    # Re-arm the trigger when crossing through level1 and then trigger when a signal on the external trigger input crosses 500 mV from below to above (positive slope)
    trigger.ext0_mode(trigger_modes[tm][0]) # re-arm the trigger with a positive slope through level1 and trigger on positive edge of external trigger through level0
    trigger.ext0_level0(trigger_modes[tm][1])  # trigger level for external trigger
    if trigger_modes[tm][2] is not None: trigger.ext0_level1(trigger_modes[tm][2])  # re-arm level for external trigger
    #############################

    num_samples = 16 * units.KiS
    max_num_gates = 128 # the maximum number of gates to be acquired

    pre_trigger = 16 * units.S
    post_trigger = 16 * units.S
    # define the data buffer
    data_transfer = spcm.Gated(card, max_num_gates=max_num_gates, verbose=True)
    data_transfer.memory_size(num_samples)
    data_transfer.pre_trigger(pre_trigger)
    data_transfer.post_trigger(post_trigger)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.polling(True)
    
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
