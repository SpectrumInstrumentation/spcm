"""
Spectrum Instrumentation GmbH (c)

1_acq_single.py

Shows a simple Standard mode example using only the few necessary commands

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units # spcm uses the pint library for unit handling (units is a UnitRegistry object)
units.default_format = "~P" # see https://pint.readthedocs.io/en/stable/user/formatting.html
units.mpl_formatter = "{:~P}" # see https://pint.readthedocs.io/en/stable/user/plotting.html

import numpy as np
import matplotlib.pyplot as plt


card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type
    
    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_SINGLE)     # single trigger standard mode
    card.timeout(5 * units.s)                     # timeout 5 s

    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_NONE)       # trigger set to none #software
    trigger.and_mask(spcm.SPC_TMASK_NONE)      # no AND mask
    # delay = trigger.delay(100 * units.us, return_unit=units.us)
    # print(f"Trigger delay: {delay}")

    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)            # clock mode internal PLL
    # we'll try to set the samplerate to 20 MHz
    sample_rate = clock.sample_rate(20 * units.MHz, return_unit=units.Hz)
    print(f"Sample rate: {sample_rate}")
    
    # setup the channels
    channels = spcm.Channels(card) # enable all channels
    amplitude_V = 200 * units.mV
    channels.amp(amplitude_V)
    channels[0].offset(-100 * units.percent)
    channels.termination(1)
    channels.coupling(spcm.COUPLING_DC)
    # max_sample_value = card.max_sample_value()

    # Channel triggering
    trigger.ch_or_mask0(channels[0].ch_mask())
    trigger.ch_mode(channels[0], spcm.SPC_TM_POS)
    ch_level = trigger.ch_level0(channels[0], 200 * units.mV, return_unit=units.mV)
    print(f"Channel trigger level: {ch_level}")

    # define the data buffer
    num_samples = 1 * units.KiS # 1 KibiSample = 1024 samples
    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.post_trigger(num_samples // 2) # half of the total number of samples after trigger event
    # Start DMA transfer
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)
    
    # start card
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)

    print("Finished acquiring...\n")

    # Plot the acquired data
    time_data_s = np.arange(num_samples)/sample_rate
    fig, ax = plt.subplots()
    for channel in channels:
        unit_data_V = channel.convert_data(data_transfer.buffer[channel.index, :], units.V)
        unit_data_N = unit_data_V * 10 * units.N / units.V
        print("Channel {}".format(channel.index))
        print("\tMinimum: {:.3~P}".format(np.min(unit_data_V)))
        print("\tMaximum: {:.3~P}".format(np.max(unit_data_V)))
        ax.plot(time_data_s, unit_data_V, label=f"Channel {channel.index}")
    ax.yaxis.set_units(units.V)
    ax.xaxis.set_units(units.us)
    ax.legend()
    plt.show()
