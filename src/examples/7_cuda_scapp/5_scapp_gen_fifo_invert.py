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
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    channels.enable(True)
    channels.output_load(50 * units.ohm)
    channels.amp(1 * units.V)

    # we try to use the max samplerate
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(250 * units.MHz, return_unit=(units.MHz))
    print(f"Used Sample Rate: {sample_rate}")
    max_value = card.max_sample_value()
    
    # Setup a data transfer object with CUDA DMA
    notify_samples = spcm.KIBI(128)
    num_samples    = spcm.MEBI(1)

    scapp_transfer = spcm.SCAPPTransfer(card, direction=spcm.Direction.Generation)
    scapp_transfer.notify_samples(notify_samples)
    scapp_transfer.allocate_buffer(num_samples)
    scapp_transfer.start_buffer_transfer(spcm.M2CMD_DATA_STARTDMA)

    # setup the kernel
    num_threads_per_block = 1024
    num_blocks = notify_samples // num_threads_per_block

    # CUDA kernel
    with open("cuda_kernels/invert.cu", "r") as f:
        kernel_src = f.read()
        name_exp = ['CudaKernelInvert<short>']
    CudaKernelInvert = cp.RawKernel(code=kernel_src, name_expressions=name_exp)

    cp_dtype = scapp_transfer.numpy_type()
    if cp_dtype == cp.int16:
        invert_function = CudaKernelInvert.get_function(name_exp[0])
    else:
        raise ValueError("Only 16-bit data types are supported.")
    
    time_range = cp.arange(notify_samples) / sample_rate.to_base_units().magnitude
    data_raw_gpu = cp.zeros_like(time_range, dtype=cp.int16)
    frequency = 1000 # Hz
    
    started = False
    phase = 0
    try:
        for card_buffer in scapp_transfer:
            # Generate sine wave data on the GPU
            data_raw_gpu[:] = cp.sin(2*cp.pi*(time_range*frequency+phase)) * 0.5 * max_value
            phase += time_range[-1]*frequency

            # Invert the sine wave
            CudaKernelInvert((num_blocks,), (num_threads_per_block,), (data_raw_gpu, card_buffer))
            fill_size = scapp_transfer.fill_size_promille()
            print("Fillsize: {}".format(fill_size), end="\r")

            # Start the card when buffer is half-full
            if fill_size > 800 and not started:
                card.start(spcm.M2CMD_CARD_ENABLETRIGGER)
                print("\n\nCard started...\n")
                started = True

    except KeyboardInterrupt:
        print("Ctrl+C pressed and generation stopped")






