"""  
Spectrum Instrumentation GmbH (c) 2024

1_sync_acq_single.py

Shows a simple standard single mode example with two or more synchronized cards using only the
few necessary commands.

Example for analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families with Starhub synchronization.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""


import spcm
from spcm import units

import matplotlib.pyplot as plt


card_identifiers = ["/dev/spcm0", "/dev/spcm1"]
sync_identifier  = "sync0"

# open cards and sync
with spcm.CardStack(card_identifiers=card_identifiers, sync_identifier=sync_identifier) as stack:
    
    channels = spcm.Channels(stack=stack, stack_enable=[spcm.CHANNEL0, spcm.CHANNEL0])
    channels.amp(1 * units.V)

    data_transfer = []
    for card in stack.cards:
        # read type, function and sn and check for A/D card
        if card.function_type() != spcm.SPCM_TYPE_AI:
            raise spcm.SpcmException(f"This is an example for A/D cards.\n{card} not supported by example\n")
        print(f"Found: {card}")

        card.card_mode(spcm.SPC_REC_FIFO_SINGLE) # single FIFO mode
        card.timeout(5 * units.s) # timeout 5 s

        trigger = spcm.Trigger(card)
        features = card.features()
        if features & (spcm.SPCM_FEAT_STARHUB5 | spcm.SPCM_FEAT_STARHUB16):
            # set star-hub carrier card as trigger master
            trigger.or_mask(spcm.SPC_TMASK_EXT0)
            trigger.ext0_mode(spcm.SPC_TM_POS)
            trigger.ext0_level0(1.5 * units.V)
            trigger.ext0_coupling(spcm.COUPLING_DC)
            trigger.termination(1)
        else:
            trigger.or_mask(spcm.SPC_TMASK_NONE)
        trigger.and_mask(spcm.SPC_TMASK_NONE)

        # we try to set the samplerate to 20 MHz on internal PLL, no clock output
        clock = spcm.Clock(card)
        clock.mode(spcm.SPC_CM_INTPLL) # clock mode internal PLL
        sample_rate = clock.sample_rate(20 * units.MHz)
        clock.clock_output(False)

        # define the data buffer
        num_samples = 128 * units.KiS
        dt = spcm.DataTransfer(card)
        dt.memory_size(num_samples)
        dt.allocate_buffer(num_samples)
        dt.post_trigger(num_samples * 3 // 4)
        dt.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA, direction=spcm.SPCM_DIR_CARDTOPC)
        data_transfer.append(dt)

    # setup star-hub
    num_cards = len(card_identifiers)
    stack.sync_enable(True)

    # start all cards using the star-hub handle
    stack.start(spcm.M2CMD_CARD_ENABLETRIGGER)

    # for each card we wait for the data from the DMA transfer
    plt.figure()
    time_data_s = data_transfer[0].time_data()
    print("Waiting for trigger...")
    for dt in data_transfer:
        dt.card.cmd(spcm.M2CMD_DATA_WAITDMA)
        print(f"{dt.card} finished transfer")
        plt.plot(time_data_s, dt.buffer[0], label=f"{dt.card}")
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Voltage [V]")
    plt.title("Acquired data")
    plt.show()
    print("Finished acquiring...")

