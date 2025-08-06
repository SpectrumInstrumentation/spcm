"""
Spectrum Instrumentation GmbH (c)

4_clk_acq_special_clock.py

An example to show a simple recording while using the internal PLL clock in special clock mode.

Example for analog recording cards (digitizers) for the the M4i and M4x card-series.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import numpy as np
import matplotlib.pyplot as plt


card : spcm.Card

# Opening a specific card
with spcm.Card('/dev/spcm0') as card:            
    # The following card families support special clock mode 22xx and 44xx
    card_family = card.family()
    print(f"Card family {card_family:02x}xx", end="")
    if card_family in [0x22, 0x44]:
        print(" is supported.")
    else:
        print(" is not supported.")
        exit()

    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_SINGLE)     # single trigger standard mode
    card.timeout(5 * units.s)                     # timeout 5 s

    ### Clock setup section ###
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL) # clock mode internal PLL
    sample_rate = clock.sample_rate(6.71 * units.MHz, special_clock=True, auto_adjust=True, return_unit=units.MHz)
    clock.output(True)  # enable the clock output

    print("Special clock mode: ")
    print(f"Sampling rate: {sample_rate}")
    print(f"Clock output: {clock.clock_output_frequency(return_unit=units.MHz)}")
    print(f"Oversampling factor: {clock.oversampling_factor()}")
    ###########################
    
    # setup the channels
    channel0, = spcm.Channels(card, card_enable=spcm.CHANNEL0) # enable channel 0
    if card_family in [0x22, 0x44]:
        channel0.coupling(spcm.COUPLING_DC)  # set channel 0 coupling to DC
    if card_family in [0x44, 0x59]:
        channel0.termination(1) # set the termination to 50 Ohm for 44xx or 59xx cards
    channel0.amp(500 * units.mV)  # set channel 0 amplitude to 500 mV
    channel0.offset(-250 * units.mV)
    print(f"Adjusted channel 0 conversion factor: {channel0.special_clock_adjust()}")

    # define the data buffer
    data_transfer = spcm.DataTransfer(card)
    data_transfer.duration(100*units.us, post_trigger_duration=80*units.us)
    
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
