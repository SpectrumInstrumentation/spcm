"""  
Spectrum Instrumentation GmbH (c) 2024

3_sync_acq_gen.py

Shows a simple example using synchronized replay and record cards

Example for analog recording cards (digitizers) and arbitrary waveform generators (AWG) for the M2p card-families with Starhub synchronization.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""


import spcm
from spcm import units

import numpy as np
import matplotlib.pyplot as plt


def setupCardAD(card : spcm.Card):
    """
    Setup AD card

    Parameters
    ----------
    card : spcm.Card
        the card object to communicate with the card
    """
    # set up the mode
    card.card_mode(spcm.SPC_REC_FIFO_SINGLE)
    
    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.amp(1 * units.V)

    # setup trigger
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)
    trigger.and_mask(0)

    # set up clock
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    clock.sample_rate(5 * units.MHz)

    card.timeout(5 * units.s)

    return channels

def setupCardDA(card : spcm.Card):
    """
    Setup DA card

    Parameters
    ----------
    card : spcm.Card
        the card object to communicate with the card
    """
    num_samples = spcm.KIBI(64)

    card.card_mode(spcm.SPC_REP_STD_CONTINUOUS)

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.enable(True)
    channels.amp(1 * units.V)
    channels.stop_level(spcm.SPCM_STOPLVL_HOLDLAST)

    # setup clock
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    clock.sample_rate(5 * units.MHz)
    clock.output(0)

    # setup trigger
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)
    trigger.and_mask(0)

    # setup buffer
    data_transfer = spcm.DataTransfer(card)
    data_transfer.memory_size(num_samples)
    data_transfer.loops(0)
    data_transfer.allocate_buffer(num_samples)

    # calculate sine waveform
    sample_space = np.linspace(0, 1, num_samples)
    data_transfer.buffer[:] = 5000 * np.sin(2.0 * np.pi * sample_space)

    # we define the buffer for transfer and start the DMA transfer
    print("Starting the DMA transfer and waiting until data is in board memory")
    data_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, spcm.M2CMD_DATA_WAITDMA)
    print("... data has been transferred to board memory")

# open cards
card_identifiers = ["/dev/spcm0", "/dev/spcm1"]
sync_identifier  = "sync0"

# open cards and sync
with spcm.CardStack(card_identifiers=card_identifiers, sync_identifier=sync_identifier) as stack:

    card_DA : spcm.Card = None
    card_AD : spcm.Card = None
    for card in stack.cards:
        # read type, function and sn and check for A/D card
        if card.function_type() == spcm.SPCM_TYPE_AO:
            card_DA = card
            print(f"DA card found: {card_DA}")
        elif card.function_type() == spcm.SPCM_TYPE_AI:
            card_AD = card
            print(f"AD card found: {card_AD}")

    if not card_AD or not card_DA:
        raise spcm.SpcmException(text="Invalid cards ...")

    # setup DA card
    setupCardDA(card_DA)

    # setup AD card
    channels_AD = setupCardAD(card_AD)

    # setup star-hub
    stack.sync_enable(True)

    # settings for the FIFO mode buffer handling
    samples_to_acquire = 10 * units.MiS
    num_samples = 4 * units.MiS
    notify_samples = 8 * units.KiS

    # buffer settings for Fifo transfer
    data_transfer_AD = spcm.DataTransfer(card_AD)
    data_transfer_AD.memory_size(num_samples)
    data_transfer_AD.pre_trigger(8)
    data_transfer_AD.allocate_buffer(num_samples)
    data_transfer_AD.notify_samples(notify_samples)
    data_transfer_AD.to_transfer_samples(samples_to_acquire)
    data_transfer_AD.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

    stack.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    print(f"Acquisition stops after {samples_to_acquire} are transferred")

    print("Press Ctrl+C to stop")

    total_samples = 0
    plot_data = np.array([], dtype=np.int16)
    channel = 0

    try:
        for data_block in data_transfer_AD:
            plot_data = np.concatenate((plot_data, data_block[channel, :]))
    except KeyboardInterrupt:
        print("Stopped by user")
        pass

    time_data = data_transfer_AD.time_data(plot_data.shape[0])
    data = channels_AD[0].convert_data(plot_data)
    plt.figure()
    plt.plot(time_data, data)
    plt.show()

    # stop star-hub
    stack.stop()
