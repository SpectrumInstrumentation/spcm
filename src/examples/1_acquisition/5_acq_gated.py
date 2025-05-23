"""
Spectrum Instrumentation GmbH (c)

5_acq_gated.py

Shows a simple Standard acquisition (recording) mode example using only the few necessary commands
- connect a function generator that generates a sine wave with 10-100 kHz frequency and 200 mV amplitude to channel 0
- connect a gate trigger that does for example a 50% duty cycle square wave with the same frequency to the external trigger input ext0

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units # spcm uses the pint library for unit handling (units is a UnitRegistry object)

import matplotlib.pyplot as plt


card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type
    
    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_GATE)
    card.timeout(5 * units.s)                     # timeout 5 s

    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_EXT0) # trigger set to external
    trigger.and_mask(spcm.SPC_TMASK_NONE)      # no AND mask
    trigger.ext0_mode(spcm.SPC_TM_POS)   # set trigger mode
    trigger.ext0_coupling(spcm.COUPLING_DC)  # trigger coupling
    trigger.ext0_level0(0.75 * units.V)
    trigger.termination(1)

    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)            # clock mode internal PLL
    clock.sample_rate(max=True)
    
    # setup the channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1) # enable channel 0
    channels.amp(1000 * units.mV)
    channels.offset(0 * units.mV)
    channels.termination(1)
    channels.coupling(spcm.COUPLING_DC)

    num_samples = 16 * units.KiS
    max_num_gates = 128 # the maximum number of gates to be acquired

    pre_trigger = 16 * units.S
    post_trigger = 16 * units.S
    # define the data buffer
    gated_transfer = spcm.Gated(card, max_num_gates=max_num_gates)
    gated_transfer.memory_size(num_samples)
    gated_transfer.pre_trigger(pre_trigger)
    gated_transfer.post_trigger(post_trigger)
    gated_transfer.allocate_buffer(num_samples)

    # Start DMA transfer
    gated_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    
    # start card
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

    print("Finished acquiring...")

    # Plot the acquired data
    fig, ax = plt.subplots()
    for gate in gated_transfer:
        time_range = gated_transfer.current_time_range(return_unit=units.us)
        gate_start, gate_end = gated_transfer.current_timestamps(return_unit=units.us)
        print(f"Gate {gated_transfer.iterator_index} - time period: {gate_start} to {gate_end} - number of points: {len(gate[0, :])}")
        for channel in channels:
            unit_data_V = channel.convert_data(gate[channel, :], units.V)
            ax.plot(time_range, unit_data_V, '.', label=f"{channel}")
        ax.axvline(gate_start, color='b', linestyle='-', alpha=0.7, label='Gate')
        ax.axvline(gate_end, color='b', linestyle='-', alpha=0.7, label='Gate')
        ax.axvspan(gate_start, gate_end, alpha=0.1, color='b')
    ax.yaxis.set_units(units.mV)
    plt.show()
