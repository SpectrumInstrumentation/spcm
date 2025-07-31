"""
Spectrum Instrumentation GmbH (c)

3_clk_acq_direct_external_clock.py

An example to show a simple recording while using the direct external clock mode. The digitizer will use the external clock directly without a PLL.

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm
from spcm import units

import numpy as np
import matplotlib.pyplot as plt


card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
# with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type
with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
    # The following card families support external clocking mode: M2p.59xx, M2p.65xx, M2p75xx, M4i77xx, M4x77xx
    card_family = card.family()
    print(f"Card family {card_family:02x}xx", end="")
    if card_family in [0x59, 0x75, 0x77]:
        print(" is supported.")
    else:
        print(" is not supported.")
        exit()

    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_STD_SINGLE)     # single trigger standard mode
    card.timeout(5 * units.s)                     # timeout 5 s
    function_type = card.function_type()

    ###########################
    ### Clock setup section ###
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_EXTERNAL)    # or SPC_CM_EXTERNAL0 and SPC_CM_EXTERNAL1
    clock.output(True)  # enable the clock output

    #    |  59xx and 65xx |     77xx |         75xx
    # ===|================|==========|================
    #  0 |    5 kOhm      | 4.7 kOhm | several kOhm
    #  1 |    50 Ohm      |  75  Ohm |     110  Ohm
    clock.termination(1) # set the clock input termination

    # 59xx and 77xx cards support clock threshold settings
    if card_family in [0x59, 0x77]:
        threshold = clock.threshold(100 * units.mV, return_unit=units.mV)  # set the clock threshold to 100 mV
        print(f"Clock threshold settings: currently {threshold}, possible values are between {clock.threshold_min(return_unit=units.mV)} and {clock.threshold_max(return_unit=units.mV)} in steps of {clock.threshold_step(return_unit=units.mV)}")

    # 75xx, 77xx cards support clock delay and edge selection settings
    if card_family in [0x75, 0x77]:
        clock.delay(280 * units.ps)
        print(f"Clock delay settings: currently {clock.delay(return_unit=units.ps)}, possible values are between {clock.delay_min(return_unit=units.ps)} and {clock.delay_max(return_unit=units.ps)} in steps of {clock.delay_step(return_unit=units.ps)}")
        clock.edge(spcm.SPCM_EDGE_RISING) # set the clock edge to rising, other options are SPCM_EDGE_FALLING and SPCM_EDGE_BOTH
        print(f"Clock edge settings: currently {clock.edge()}, possible values are {spcm.SPCM_EDGE_RISING} (SPCM_EDGE_RISING), {spcm.SPCM_EDGE_FALLING} (SPCM_EDGE_FALLING) and {spcm.SPCM_EDGES_BOTH} (SPCM_EDGES_BOTH)")
    ###########################

    channels = spcm.Channels(card)

    # define the data buffer
    data_transfer = spcm.DataTransfer(card)
    num_samples = 128 * units.S
    data_transfer.memory_size(num_samples)
    data_transfer.allocate_buffer(num_samples)
    data_transfer.post_trigger(num_samples // 2)
    
    # start card and wait until recording is finished
    card.start(spcm.M2CMD_CARD_ENABLETRIGGER, spcm.M2CMD_CARD_WAITREADY)

    print("Finished acquiring...")

    # Start DMA transfer and wait until the data is transferred
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)

    # Plot the acquired data
    time_data = data_transfer.time_data()
    fig, ax = plt.subplots()
    if function_type == spcm.SPCM_TYPE_DI or function_type == spcm.SPCM_TYPE_DIO:
        # For digital cards
        unit_data = data_transfer.unpackbits()[:, 0]
        ax.step(time_data, data_transfer.bit_buffer[:, 0])
    else:
        # For analog cards
        unit_data = data_transfer.buffer[0, :]
        ax.plot(time_data, unit_data)
        ax.yaxis.set_units(units.mV)
    print("\tMinimum: {}".format(np.min(unit_data)))
    print("\tMaximum: {}".format(np.max(unit_data)))
    ax.xaxis.set_units(units.us)
    ax.axvline(0, color='k', linestyle='--', label='Trigger')
    ax.legend()
    plt.show()
