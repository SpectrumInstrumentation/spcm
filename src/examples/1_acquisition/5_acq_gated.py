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

import numpy as np
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
    clock.sample_rate(20 * units.MHz, return_unit=units.MHz)
    
    # setup the channels
    channel0, = spcm.Channels(card, card_enable=spcm.CHANNEL0) # enable channel 0
    channel0.amp(1000 * units.mV)
    channel0.offset(0 * units.mV)
    channel0.termination(1)
    channel0.coupling(spcm.COUPLING_DC)

    num_samples = 128 * units.KiSa
    pre_trigger = 128
    post_trigger = 32
    # define the data buffer
    data_transfer = spcm.Gated(card)
    data_transfer.memory_size(num_samples)
    data_transfer.pre_trigger(pre_trigger)
    data_transfer.post_trigger(post_trigger)
    data_transfer.allocate_buffer(num_samples)

    num_timestamps = 16
    # setup timestamps
    ts = spcm.TimeStamp(card)
    ts.mode(spcm.SPC_TSMODE_STARTRESET, spcm.SPC_TSCNT_INTERNAL)
    ts.allocate_buffer(num_timestamps)

    # Create second buffer
    ts.start_buffer_transfer(spcm.M2CMD_EXTRA_STARTDMA)

    # Start DMA transfer
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    
    # start card
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA, spcm.M2CMD_EXTRA_WAITDMA)

    # Get the number of gates acquired
    num_gates = 4 # + trigger.trigger_counter()
    print(f"Num gates: {num_gates}")

    print("Finished acquiring...")

    # Plot the acquired data
    time_data_s = data_transfer.time_data()
    fig, ax = plt.subplots()
    unit_data_V = channel0.convert_data(data_transfer.buffer[channel0, :], units.V)
    print(channel0)
    print("\tMinimum: {:.3~P}".format(np.min(unit_data_V)))
    print("\tMaximum: {:.3~P}".format(np.max(unit_data_V)))
    start = 0
    for i in range(num_gates):
        ax.axvline(ts.buffer[2*i,   0], color='r', linestyle='--', label='Gate')
        ax.axvline(ts.buffer[2*i+1, 0], color='g', linestyle='--', label='Gate')
        start_t = ts.buffer[2*i, 0] - pre_trigger
        end_t   = ts.buffer[2*i+1, 0] + post_trigger
        end = start + end_t - start_t
        x_range = np.arange(start_t, end_t)
        ax.plot(x_range, unit_data_V[start:end], '.', label=f"{channel0}")
        start = end
    # ax.legend()
    ax.yaxis.set_units(units.mV)
    ax.xaxis.set_units(units.us)
    plt.show()
