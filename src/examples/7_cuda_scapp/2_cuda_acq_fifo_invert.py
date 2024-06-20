"""
Spectrum Instrumentation GmbH (c)

2_cuda_acq_fifo_invert.py

Example that shows how to combine the CUDA DMA transfer with the acquisition of data. The example uses FIFO recording mode
to acquire data then send the data through dma to the host system, the host system sends the data to the GPU which inverts
the data and sends it back to the host memory. On the host memory the data is plotted continuously, using matplotlib.

For analog recording cards (digitizers) for the the M2p, M4i, M4x and M5i card-families.

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

    # we try to use the max samplerate
    clock = spcm.Clock(card)
    clock.mode(spcm.SPC_CM_INTPLL)
    sample_rate = clock.sample_rate(10 * units.MHz, return_unit=(units.MHz))
    print(f"Used Sample Rate: {sample_rate}")
    
    # Setup a data transfer object with CUDA DMA
    notify_samples = spcm.KIBI(2)
    num_samples    = spcm.MEBI(8)

    cuda_object   = spcm.CUDA(card, 0, scapp=False)
    cuda_transfer = spcm.CUDATransfer(card, cuda_object)
    cuda_transfer.notify_samples(notify_samples)

    # CUDA kernel
    CudaKernelInvert = cuda_object.create_kernel(function_name="CudaKernelInvert", src = '''\
        #define int8 char
        extern "C" __global__ void CudaKernelInvert (int8* pcIn, int8* pcOut) 
            {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            pcOut[i] = -1 * pcIn[i]; 
            }
        ''')

    cuda_transfer.allocate_buffer(num_samples)
    cuda_transfer.start_buffer_transfer()

    card.start(spcm.M2CMD_CARD_ENABLETRIGGER | spcm.M2CMD_DATA_STARTDMA)
    
    # plot function
    fig, ax = plt.subplots()
    time_range = np.arange(notify_samples) / sample_rate
    line1, = ax.plot(time_range, np.zeros_like(time_range))
    line2, = ax.plot(time_range, np.zeros_like(time_range))
    ax.set_ylim([-100, 100])  # range of Y axis
    ax.xaxis.set_units(units.us)
    ax.yaxis.set_units(units.mV)
    plt.show(block=False)
    plt.draw()

    for card_to_cpu, cpu_to_gpu, gpu_to_cpu in cuda_transfer:
        # this is the point to do anything with the data on the GPU

        cpu_to_gpu.array[:] = card_to_cpu[:]
        cpu_to_gpu.copy_to_device()

        # start kernel on the GPU to process the transfered data
        threads_per_block = 1024
        num_blocks = notify_samples // threads_per_block
        kernel_arguments = (cpu_to_gpu, gpu_to_cpu)
        CudaKernelInvert.launch(num_blocks, threads_per_block, kernel_arguments)
        
        # after kernel has finished we copy processed data from GPU to host
        cpu_to_gpu.copy_to_host()
        gpu_to_cpu.copy_to_host()
 
        # now the processed data is in the host memory
        line1.set_ydata(cpu_to_gpu.view)
        line2.set_ydata(gpu_to_cpu.view)
        fig.canvas.draw()
        fig.canvas.flush_events()




