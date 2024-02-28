"""  
Spectrum Instrumentation GmbH (c) 2024

2_sync_acq_fifo.py

Shows a simple multi-threaded FIFO mode example using only the
few necessary commands.

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families with Starhub synchronization.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""


import threading
import spcm
import numpy as np

class CardThread (threading.Thread):
    """
    thread that handles the data transfer for a card
    One instance will be started for each card.
    """

    def __init__ (self, index : int, data_transfer : spcm.DataTransfer):
        threading.Thread.__init__(self)
        self.index    = index     # index of card (only used for output)
        self.data_transfer = data_transfer  # DMA buffer for the card

    def run(self):
        minimum = 32767
        maximum = -32768
        for data_block in self.data_transfer:
            minimum = np.min([minimum, np.min(data_block)])
            maximum = np.max([maximum, np.max(data_block)])

        # print the calculated results
        print("\n{0} Finished... Minimum: {1:d} Maximum: {2:d}".format(self.index, minimum, maximum))


card_identifiers = ["/dev/spcm0", "/dev/spcm1"]
sync_identifier  = "sync0"

# open cards and sync
with spcm.CardStack(card_identifiers=card_identifiers, sync_identifier=sync_identifier) as stack:
    
    channels = spcm.Channels(stack=stack, stack_enable=[spcm.CHANNEL0, spcm.CHANNEL0])
    channels.amp(1000) # 1000 mV

    data_transfer = []
    for card in stack.cards:
        # read type, function and sn and check for A/D card
        sn   = card.sn()
        fnc_type        = card.function_type()
        card_name       = card.product_name()
        if fnc_type != spcm.SPCM_TYPE_AI:
            raise spcm.SpcmException(text="This is an example for A/D cards.\nCard: {0} sn {1:05d} not supported by example\n".format(card_name, sn))
        print("Found: {0} sn {1:05d}".format(card_name, sn))

        card.card_mode(spcm.SPC_REC_FIFO_SINGLE) # single FIFO mode
        card.timeout(5000) # timeout 5 s

        trigger = spcm.Trigger(card)
        trigger.or_mask(spcm.SPC_TMASK_SOFTWARE) # trigger set to software
        trigger.and_mask(0)

        # we try to set the samplerate to 20 MHz on internal PLL, no clock output
        clock = spcm.Clock(card)
        clock.mode(spcm.SPC_CM_INTPLL) # clock mode internal PLL
        sample_rate = clock.sample_rate(spcm.MEGA(20))
        clock.output(0) # no clock output

        # define the data buffer
        num_samples = spcm.MEBI(2)
        notify_samples = spcm.KIBI(8)
        dt = spcm.DataTransfer(card)
        dt.memory_size(num_samples)
        dt.pre_trigger(1024) # 1k of pretrigger data at start of FIFO mode
        dt.allocate_buffer(num_samples)
        dt.notify_samples(notify_samples)
        dt.to_transfer_samples(spcm.MEBI(8))
        dt.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, direction=spcm.SPCM_DIR_CARDTOPC)
        data_transfer.append(dt)

    # setup star-hub
    num_cards = len(card_identifiers)
    stack.sync_enable(True)

    # find star-hub carrier card and set it as clock master
    for i, card in enumerate(stack.cards):
        features = card.features()
        if features & (spcm.SPCM_FEAT_STARHUB5 | spcm.SPCM_FEAT_STARHUB16):
            break

    # start all cards using the star-hub handle
    stack.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    # for each card we start a thread that controls the data transfer
    list_threads = []
    for i, card in enumerate(stack.cards):
        thread = CardThread(i, data_transfer[i])
        list_threads.append(thread)
        thread.start()

    # wait until all threads have finished
    for x in list_threads:
        x.join()
