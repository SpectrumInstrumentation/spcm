"""  
Spectrum Instrumentation GmbH (c) 2024

3_sync_acq_gen.py

Shows a simple example using synchronized replay and record cards

Example for analog recording cards (digitizers) and arbitrary waveform generators (AWG) for the M2p card-families with Starhub synchronization.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""


import spcm
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
    channels.amp(1000)  # 1000 mV

    # setup trigger
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)
    trigger.and_mask(0)

    # set up clock
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    clock.sample_rate(spcm.MEGA(5))

    card.timeout(5000)

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
    channels.amp(1000)  # 1000 mV
    channels.stop_level(spcm.SPCM_STOPLVL_HOLDLAST)

    # setup clock
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    clock.sample_rate(spcm.MEGA(5))
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
        sn = card.sn()
        fnc_type = card.function_type()
        card_name = card.product_name()
        if fnc_type == spcm.SPCM_TYPE_AO:
            card_DA = card
            print("DA card found: {0} sn {1:05d}".format(card_name, sn))
        elif fnc_type == spcm.SPCM_TYPE_AI:
            card_AD = card
            print("AD card found: {0} sn {1:05d}".format(card_name, sn))

    if not card_AD or not card_DA:
        raise spcm.SpcmException(text="Invalid cards ...")

    # setup DA card
    setupCardDA(card_DA)

    # setup AD card
    setupCardAD(card_AD)

    # setup star-hub
    stack.sync_enable(True)

    # settings for the FIFO mode buffer handling
    samples_to_acquire_MiS = 10 # 10 MiS
    num_samples = spcm.MEBI(4)
    notify_samples = spcm.KIBI(8)

    # buffer settings for Fifo transfer
    data_transfer_AD = spcm.DataTransfer(card_AD)
    data_transfer_AD.memory_size(num_samples)
    data_transfer_AD.pre_trigger(8)
    data_transfer_AD.allocate_buffer(num_samples)
    data_transfer_AD.notify_samples(notify_samples)
    data_transfer_AD.to_transfer_samples(spcm.MEBI(samples_to_acquire_MiS))
    data_transfer_AD.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

    stack.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    print("Acquisition stops after {} MSamples are transferred".format(samples_to_acquire_MiS))

    print("Press Ctrl+C to stop")

    total_samples = 0
    plot_data = np.array([], dtype=np.int16)
    channel = 0

    try:
        for data_block in data_transfer_AD:
            plot_data = np.concatenate((plot_data, data_block[channel, :]))
    except KeyboardInterrupt:
        print("\nStopped by user")
        pass

    x_range = np.arange(0, len(plot_data)) / spcm.MEGA(5) # 5 MHz Sample Rate
    plt.figure()
    plt.plot(x_range, plot_data)
    plt.show()

    # stop star-hub
    stack.stop()
