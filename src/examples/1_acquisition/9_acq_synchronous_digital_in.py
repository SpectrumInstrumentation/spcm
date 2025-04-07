"""
Spectrum Instrumentation GmbH (c)

9_acq_synchronous_digital_in.py

A Standard acquisition (recording) mode example including synchronous bits from the digital inputs.
- connect a function generator that generates a sine wave with 10-100 kHz frequency and 200 mV amplitude to channel 0
- connect a digital signal to the digital input x0, x1 or x2

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
    card.card_mode(spcm.SPC_REC_STD_SINGLE) # single trigger standard mode
    card.timeout(5 * units.s)               # timeout 5 s

    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_NONE)    # trigger set to none
    trigger.and_mask(spcm.SPC_TMASK_NONE)   # no AND mask

    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)          # clock mode internal PLL
    clock.sample_rate(20 * units.MHz, return_unit=units.MHz)
    
    # setup the channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1) # enable channel 0 and 1
    channels.amp(1000 * units.mV)
    channels.offset(0 * units.mV)
    channels.termination(1)
    channels.coupling(spcm.COUPLING_DC)

    # Channel triggering on channel 0
    trigger.ch_or_mask0(channels[0].ch_mask())
    trigger.ch_mode(channels[0], spcm.SPC_TM_POS)
    trigger.ch_level0(channels[0], 0 * units.mV, return_unit=units.mV)

    # define the data buffer
    num_samples = 8 * units.KiS
    num_samples_magnitude = num_samples.to_base_units().magnitude
    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.post_trigger(num_samples//2)
    # Start DMA transfer
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

    # add synchronous digital outputs to the data
    synchronous_io = spcm.SynchronousDigitalIOs(data_transfer, channels, digin2bit=True) # digin2bit is only using for the 44xx family of cards
    if not (card.family() == 0x22 or card.family() == 0x23 or card.family() == 0x44):
        # for the 22xx, 23xx and 44xx families there is a fixed setup for the digital outputs
        num_buffers = 4
        index = synchronous_io.allocate_buffer(num_buffers=num_buffers)
        for index in range(num_buffers):
            synchronous_io.setup(buffer_index=index, channel=channels[index % len(channels)], xios=[index % synchronous_io.num_xio_lines])
    
    # start card
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_DATA_WAITDMA)
    print("Finished acquiring...")

    synchronous_io.process()

    # Plot the acquired data
    time_data = data_transfer.time_data()
    
    fig, ax = plt.subplots(1 + len(synchronous_io), 1, sharex=True, gridspec_kw={'height_ratios': [5, *([1]*len(synchronous_io))]})
    ax_num = 0
    for channel in channels:
        unit_data = channel.convert_data(data_transfer.buffer[channel, :], units.V)
        print(channel)
        print("\tMinimum: {:.3~P}".format(np.min(unit_data)))
        print("\tMaximum: {:.3~P}".format(np.max(unit_data)))
        ax[ax_num].plot(time_data, unit_data, label=f"{channel}")
    ax[ax_num].yaxis.set_units(units.mV)
    ax[ax_num].xaxis.set_units(units.us)
    ax[ax_num].axvline(0, color='k', linestyle='--', label='Trigger')
    ax_num += 1

    for index in synchronous_io:
        ax[ax_num].step(time_data, index, label=f"X{synchronous_io.current_xio()}")
        ax[ax_num].set_ylabel(f"X{synchronous_io.current_xio()}")
        ax[ax_num].set_yticks([0, 1])
        ax[ax_num].set_ylim(-0.2, 1.2)
        ax[ax_num].xaxis.set_units(units.us)
        ax_num += 1

    plt.show()
