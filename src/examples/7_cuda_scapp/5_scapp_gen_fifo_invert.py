"""
Spectrum Instrumentation GmbH (c)

5_scapp_gen_fifo_sine.py

Example that shows how to combine the CUDA DMA transfer with the generation of data. The example uses the GPU to generate a sine wave using CuPy,
which is then also inverted on the GPU and sends to the card using RDMA from the SCAPP option.

For analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""


import spcm
from spcm import units

import cupy as cp

card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AO, verbose=False) as card:            # if you want to open the first card of a specific type

    # do a simple standard setup
    card.card_mode(spcm.SPC_REP_FIFO_SINGLE)     # single trigger standard mode
    card.timeout(5 * units.s)

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0 | spcm.CHANNEL1)
    channels.enable(True)
    channels.output_load(50 * units.ohm)
    channels.amp(1 * units.V)

    # we try to use the max samplerate
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(max=True, return_unit=(units.MHz))
    sample_rate_magnitude = sample_rate.to_base_units().magnitude
    print(f"Used Sample Rate: {sample_rate}")
    max_value = card.max_sample_value()
    
    # Setup a data transfer object with CUDA DMA
    num_samples    =  32 * units.MiS # MebiSamples = 1024 * 1024 Samples
    notify_samples = 512 * units.KiS
    notify_samples_magnitude = notify_samples.to_base_units().magnitude

    scapp_transfer = spcm.SCAPPTransfer(card, direction=spcm.Direction.Generation)
    scapp_transfer.notify_samples(notify_samples)
    scapp_transfer.allocate_buffer(num_samples)
    scapp_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

    cp_dtype = scapp_transfer.numpy_type()
    
    output_divider = 10
    time_range = cp.arange(notify_samples_magnitude, dtype=cp.float32) / sample_rate_magnitude
    data_raw_gpu = cp.empty((len(channels), notify_samples_magnitude), dtype=cp_dtype, order='F')
    frequency1 = 1000 # Hz
    frequency2 = 1002 # Hz

    phase = 0
    
    started = False
    try:
        counter = 0
        for card_buffer in scapp_transfer:
            # waits for a memory block to become available for writing from the gpu memory to the card using scapp

            # ... this is the point where the data can be generated on the gpu

            # Generate 2 sine waves with data on the gpu
            card_buffer[0, :] = cp.sin(2*cp.pi*((time_range+phase)*frequency1)) * 0.5 * max_value
            card_buffer[1, :] = cp.sin(2*cp.pi*((time_range+phase)*frequency2)) * 0.5 * max_value
            phase += notify_samples_magnitude / sample_rate_magnitude

            fill_size = scapp_transfer.fill_size_promille()

            # Start the card when buffer is half-full
            if fill_size > 800 and not started:
                card.start(spcm.M2CMD_CARD_ENABLETRIGGER)
                print("\n\nCard started...\n")
                started = True
            
            # Check the filling of the on board memory
            if counter % output_divider == 0:
                print("Fill size: {}".format(fill_size), end="\r")
            
            counter += 1

    except KeyboardInterrupt:
        print("Ctrl+C pressed and generation stopped")






