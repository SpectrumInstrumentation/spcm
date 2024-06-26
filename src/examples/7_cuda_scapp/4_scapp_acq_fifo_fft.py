"""
Spectrum Instrumentation GmbH (c)

4_scapp_acq_fifo_fft.py

Example that shows how to combine the CUDA DMA transfer with the acquisition of data. The example uses FIFO recording mode
to acquire data then send the data through rdma to the GPU using the SCAPP add-on, which takes the Fast-Fourier-Transform (FFT)
of the data and sends it back to the host memory. On the host memory the data is plotted continuously, using matplotlib.

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

    plot_divider = 10 # plot 1 in 1 data blocks
    
    # Setup a data transfer object with CUDA DMA
    notify_samples = 64 * units.KiS
    notify_samples_magnitude = notify_samples.to_base_units().magnitude
    num_samples = 8 * units.MiS

    num_thread_per_block = 1024
    num_blocks = notify_samples_magnitude // num_thread_per_block

    scapp_transfer = spcm.SCAPPTransfer(card, direction=spcm.Direction.Acquisition)
    scapp_transfer.notify_samples(notify_samples)
    scapp_transfer.allocate_buffer(num_samples)
    scapp_transfer.start_buffer_transfer()
    
    # length of FFT result
    num_fft_samples = notify_samples_magnitude // 2 + 1
    num_fft_blocks = num_fft_samples // num_thread_per_block

    # allocate memory on GPU
    data_volt_gpu = cp.zeros(notify_samples_magnitude, dtype = cp.float32)
    spectrum_gpu = cp.zeros(num_fft_samples, dtype = cp.float32)

    # convert raw data to volt
    CupyKernelConvertSignalToVolt = cp.RawKernel(r'''
        extern "C" __global__
        void CudaKernelScale(const short* anSource, float* afDest, double dFactor) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            afDest[i] = ((float)anSource[i]) * dFactor;
        }
        ''', 'CudaKernelScale')

    # scale the FFT result
    CupyKernelScaleFFTResult = cp.RawKernel(r'''
        extern "C" __global__
        void CudaScaleFFTResult (complex<float>* pcompDest, const complex<float>* pcompSource, int lLen) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            pcompDest[i].real (pcompSource[i].real() / (lLen / 2 + 1)); // divide by length of signal
            pcompDest[i].imag (pcompSource[i].imag() / (lLen / 2 + 1)); // divide by length of signal
        }
        ''', 'CudaScaleFFTResult', translate_cucomplex=True)

    # calculate real spectrum from complex FFT result
    CupyKernelFFTToSpectrum = cp.RawKernel(r'''
        extern "C" __global__
        void CudaKernelFFTToSpectrum (const complex<float>* pcompSource, float* pfDest) {
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            pfDest[i] = sqrt (pcompSource[i].real() * pcompSource[i].real() + pcompSource[i].imag() * pcompSource[i].imag());
        }
        ''', 'CudaKernelFFTToSpectrum', translate_cucomplex=True)

    # convert to dBFS
    CupyKernelSpectrumToDBFS = cp.RawKernel(r'''
    extern "C" __global__
    void CudaKernelToDBFS (float* pfDest, const float* pfSource, int lIR_V) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        pfDest[i] = 20. * log10f (pfSource[i] / lIR_V);
    }
    ''', 'CudaKernelToDBFS')
    
    # plot function
    fig, ax = plt.subplots()
    freq = np.linspace(0, 1, num_fft_samples - 1) * sample_rate / 2
    line, = ax.plot(freq, np.zeros_like(freq))
    ax.set_ylim([-140.0, 10.0])  # range of Y axis
    ax.xaxis.set_units(units.MHz)
    plt.show(block=False)
    plt.draw()

    card.start(spcm.M2CMD_CARD_ENABLETRIGGER | spcm.M2CMD_DATA_STARTDMA)

    counter = 0
    for data_raw_gpu in scapp_transfer:
        # this is the point to do anything with the data on the GPU
        CupyKernelConvertSignalToVolt((num_blocks,), (num_thread_per_block,), (data_raw_gpu, data_volt_gpu, amplitude.to(units.V).magnitude / max_value))
        
        # calculate the FFT
        fftdata_gpu = cp.fft.fft(data_volt_gpu)

        # scale the FFT result
        CupyKernelScaleFFTResult((num_blocks,), (num_thread_per_block,), (fftdata_gpu, fftdata_gpu, num_fft_samples))

        # calculate real spectrum from complex FFT result
        CupyKernelFFTToSpectrum((num_fft_blocks,), (num_thread_per_block,), (fftdata_gpu, spectrum_gpu))

        # convert to dBFS
        CupyKernelSpectrumToDBFS((num_fft_blocks,), (num_thread_per_block,), (spectrum_gpu, spectrum_gpu, 1))

        # after kernel has finished we copy processed data from GPU to host
        spectrum_cpu = cp.asnumpy(spectrum_gpu)
 
        # now the processed data is in the host memory
        if counter % plot_divider == 0:
            line.set_ydata(spectrum_cpu[:-1])
            fig.canvas.draw()
            fig.canvas.flush_events()
        counter += 1





