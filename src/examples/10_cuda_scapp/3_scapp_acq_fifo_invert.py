"""
Spectrum Instrumentation GmbH (c)

3_scapp_acq_fifo_invert.py

Example that shows how to combine the CUDA DMA transfer with the acquisition of data. The example uses FIFO recording mode
to acquire data then send the data through rdma to the GPU using the SCAPP add-on, which inverts the data and sends it back
to the host memory. On the host memory the data is plotted continuously, using matplotlib.

For analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""


import spcm
from spcm import units

import numpy as np
import matplotlib.pyplot as plt
import cupy as cp

card : spcm.Card

# with spcm.Card('/dev/spcm0') as card:                         # if you want to open a specific card
# with spcm.Card('TCPIP::192.168.1.10::inst0::INSTR') as card:  # if you want to open a remote card
# with spcm.Card(serial_number=12345) as card:                  # if you want to open a card by its serial number
with spcm.Card(card_type=spcm.SPCM_TYPE_AI) as card:            # if you want to open the first card of a specific type

    # do a simple standard setup
    card.card_mode(spcm.SPC_REC_FIFO_SINGLE)     # single trigger standard mode
    card.timeout(5 * units.s)

    # setup trigger engine
    trigger = spcm.Trigger(card)
    trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    # setup channels
    channels = spcm.Channels(card, card_enable=spcm.CHANNEL0)
    amplitude = channels[0].amp(1 * units.V, return_unit=units.V)
    max_value = card.max_sample_value()

    # we try to use the max samplerate
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(10 * units.MHz, return_unit=(units.MHz))
    print(f"Used Sample Rate: {sample_rate}")
    
    # Setup a data transfer object with CUDA DMA
    num_samples = 8 * units.MiS
    notify_samples = 64 * units.KiS
    num_samples_magnitude = num_samples.to_base_units().magnitude
    notify_samples_magnitude = notify_samples.to_base_units().magnitude

    scapp_transfer = spcm.SCAPPTransfer(card, direction=spcm.Direction.Acquisition)
    scapp_transfer.notify_samples(notify_samples)
    scapp_transfer.allocate_buffer(num_samples)
    scapp_transfer.start_buffer_transfer()

    # invert data on GPU
    data_processed_gpu = cp.zeros((len(channels), notify_samples_magnitude), dtype = scapp_transfer.numpy_type())

    # elementwise kernel to invert the data
    kernel_invert = cp.ElementwiseKernel(
        'T rawData', # input data can be any integer
        'T processedData', # output data of the same type as the input
        'processedData = -rawData', # the inversion
        'invert') # name of the kernel

    card.start(spcm.M2CMD_CARD_ENABLETRIGGER | spcm.M2CMD_DATA_STARTDMA)
    
    # plot function
    fig, ax = plt.subplots()
    time_range = np.arange(notify_samples, dtype=np.float32) / sample_rate
    line1, = ax.plot(time_range, np.zeros_like(time_range))
    line2, = ax.plot(time_range, np.zeros_like(time_range))
    ax.set_ylim([-1.2*max_value, 1.2*max_value])  # range of Y axis
    ax.xaxis.set_units(units.us)
    plt.show(block=False)
    plt.draw()

    for data_raw_gpu in scapp_transfer:
        # waits for a block to become available after the data is transferred from the card to the gpu memory using scapp

        # ... this is the point to do anything with the data on the gpu

        # start kernel on the gpu to process the transfered data
        kernel_invert(data_raw_gpu, data_processed_gpu)
        
        # after kernel has finished we copy processed data from the gpu to the host cpu
        data_raw_cpu       = cp.asnumpy(data_raw_gpu)
        data_processed_cpu = cp.asnumpy(data_processed_gpu)
 
        # now the processed data is availablle to the host cpu and can be used for plotting
        line1.set_ydata(data_raw_cpu)
        line2.set_ydata(data_processed_cpu)
        fig.canvas.draw()
        fig.canvas.flush_events()




