# -*- coding: utf-8 -*-

from spcm.constants import SPCM_BUF_DATA
from .classes_card import Card
from .classes_channels import Channels
from .classes_data_transfer import DataTransfer
from .pyspcm import spcm_dwDefTransfer_i64, SPCM_BUF_DATA, SPCM_DIR_CARDTOGPU, SPCM_DIR_GPUTOCARD, SPCM_DIR_CARDTOPC, SPCM_DIR_PCTOCARD,\
     c_void_p, c_int, c_float

import numpy as np
from enum import Enum

from cuda import cuda, cudart, nvrtc

import pint
from .classes_unit_conversion import UnitConversion
from . import units

def _cudaGetErrorEnum(error):
    if isinstance(error, cuda.CUresult):
        err, name = cuda.cuGetErrorName(error)
        return name if err == cuda.CUresult.CUDA_SUCCESS else "<unknown>"
    elif isinstance(error, cudart.cudaError_t):
        return cudart.cudaGetErrorName(error)[1]
    elif isinstance(error, nvrtc.nvrtcResult):
        return nvrtc.nvrtcGetErrorString(error)[1]
    else:
        raise RuntimeError('Unknown error type: {}'.format(error))

def checkCudaErrors(result):
    if result[0].value:
        raise RuntimeError("CUDA error code={}({})".format(result[0].value, _cudaGetErrorEnum(result[0])))
    if len(result) == 1:
        return None
    elif len(result) == 2:
        return result[1]
    else:
        return result[1:]
    



class CUDABuffer:
    """
    CUDA buffer object.
    
    TODO: Add support for different data types and number of channels.
    
    """

    cuda_object : "CUDADevice" = None
    dtype : np.dtype = None
    bytes_per_sample : int = 0
    num_channels : int = 1
    size : int = 0
    offset_size : int = 0
    length_size : int = 0
    buffer : cuda.CUdeviceptr = 0
    array : np.ndarray = None

    _samples : int = 0
    _offset_samples : int = 0
    _length_samples : int = 0

    @property
    def samples(self):
        return self._samples
    
    @samples.setter
    def samples(self, value):
        self._samples = value
        self.size = value * self.bytes_per_sample * self.num_channels
    
    @property
    def offset_samples(self):
        return self._offset_samples
    
    @offset_samples.setter
    def offset_samples(self, value):
        self._offset_samples = value
        self.offset_size = value * self.bytes_per_sample * self.num_channels

    @property
    def length_samples(self):
        return self._length_samples
    
    @length_samples.setter
    def length_samples(self, value):
        self._length_samples = value
        self.length_size = value * self.bytes_per_sample * self.num_channels

    def __init__(self, cuda_object, samples : int, dtype, bytes_per_sample : int, num_channels : int = 1, device_only : bool = False):
        self.cuda_object = cuda_object
        self.dtype = dtype
        self.bytes_per_sample = bytes_per_sample
        self.samples = samples
        self.num_channels = num_channels
        self.buffer = checkCudaErrors(cudart.cudaMalloc(self.size))
        if not device_only:
            self.array = np.ndarray((samples,), dtype = dtype)
            self.set_view(0, samples)
    
    def __del__(self):
        checkCudaErrors(cudart.cudaFree(self.buffer))

    def copy_to_host(self):
        """
        Copy the data from the CUDA buffer to the host buffer.
        """
        checkCudaErrors(cudart.cudaMemcpy(self.array.ctypes.data, self.buffer, self.size, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))
        return self.array
    
    def copy_to_device(self):
        """
        Copy the data from the host buffer to the CUDA buffer.
        """
        checkCudaErrors(cudart.cudaMemcpy(self.buffer, self.array.ctypes.data, self.size, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))
    
    def set_view(self, offset_samples : int, length_samples : int):
        """
        Set the view of the buffer.

        Parameters
        ----------
        offset : int
            Offset of the view.
        view_size : int
            Size of the view.
        """
        self.offset_samples = offset_samples
        self.length_samples = length_samples
        # print(f"{self.offset_samples = }, {self.length_samples = }")
        # print(f"{self.offset_size = }, {self.length_size = }")
        self.view = self.array[self.offset_samples:self.offset_samples + self.length_samples]

    
class CUDATransfer(DataTransfer):

    cuda_object : "CUDA" = None

    # General CUDA buffer
    gpu_to_cpu_buffer : CUDABuffer = None

    # RDMA specific buffers
    card_gpu_buffer : CUDABuffer = None # both for acquisition and generation

    # DMA specific buffers
    card_cpu_buffer = None # both for acquisition and generation # TODO remove and use cpu_to_gpu_buffer directly
    cpu_to_gpu_buffer : CUDABuffer = None

    def __init__(self, card, cuda_object : "CUDA", direction : DataTransfer.Direction = DataTransfer.Direction.Acquisition):
        super().__init__(card)
        self.cuda_object = cuda_object
        self.card_cpu_buffer = self._np_buffer # An alias for the buffer # TODO remove and use cpu_to_gpu_buffer directly
        self.direction = direction

    def allocate_buffer(self, num_samples : int, no_reshape = False) -> None:
        """
        Memory allocation for the buffer that is used for communicating with the card

        Parameters
        ----------
        num_samples : int | pint.Quantity = None
            use the number of samples an get the number of active channels and bytes per samples directly from the card
        no_reshape : bool = False
            If True, the buffer will not be reshaped to the number of channels and bytes per sample
        """
        
        super().allocate_buffer(num_samples, no_reshape) # TODO still needed for the buffer allocation, but should be removed in the future

        # Allocate CUDA buffers
        self.gpu_to_cpu_buffer = CUDABuffer(self.cuda_object.device, self._notify_samples, self.numpy_type(), self.bytes_per_sample, len(self.cuda_object.channels))
        self.cpu_to_gpu_buffer = CUDABuffer(self.cuda_object.device, self._notify_samples, self.numpy_type(), self.bytes_per_sample, len(self.cuda_object.channels))
        if self.cuda_object.scapp:
            self.card_gpu_buffer = CUDABuffer(self.cuda_object.device, self.buffer_samples, self.numpy_type(), self.bytes_per_sample, len(self.cuda_object.channels))
            flag = 1
            checkCudaErrors(cuda.cuPointerSetAttribute(flag, cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, self.card_gpu_buffer.buffer))
        else:
            self.card_cpu_buffer = self.buffer
    
    def start_buffer_transfer(self, *args, notify_samples = None, transfer_length = None) -> None:
        """
        Setup an RDMA transfer.

        Parameters
        ----------
        gpu_buffer : cuda.CUdeviceptr
            GPU buffer to transfer data to.
        direction : int
            Direction of the transfer.
        notify_samples : int
            Size of the notification buffer.
        """

        self.notify_samples(notify_samples)
        self.buffer_samples = transfer_length

        # Define transfer CUDA buffers
        if self.cuda_object.scapp:
            if self.direction == self.Direction.Acquisition:
                direction = SPCM_DIR_CARDTOGPU
            else:
                direction = SPCM_DIR_GPUTOCARD
            self.card._check_error(spcm_dwDefTransfer_i64(self.card._handle, SPCM_BUF_DATA, direction, self.notify_size, c_void_p(self.card_gpu_buffer.buffer), 0, self.buffer_size))
        else:
            if self.direction == self.Direction.Acquisition:
                direction = SPCM_DIR_CARDTOPC
            else:
                direction = SPCM_DIR_PCTOCARD
            self._c_buffer = self.buffer.ctypes.data_as(c_void_p)
            self.card._check_error(spcm_dwDefTransfer_i64(self.card._handle, SPCM_BUF_DATA, direction, self.notify_size, self._c_buffer, 0, self.buffer_size))

        # Execute additional commands if available
        if args:
            cmd = 0
            for arg in args:
                cmd |= arg
            self.card.cmd(cmd)
            self.card._print("... CUDA data transfer started")
        
    def __next__(self) -> tuple:
        """
        Get the next available user position.
        """
        super().__next__()
        if self.cuda_object.scapp:
            # print(self.current_user_pos, self._notify_samples)
            self.card_gpu_buffer.set_view(self.current_user_pos, self._notify_samples)
            if self.direction == self.Direction.Acquisition:
                return self.card_gpu_buffer, self.gpu_to_cpu_buffer
            else:
                return self.card_gpu_buffer, self.cpu_to_gpu_buffer
        else:
            # self.card_to_cpu_buffer.set_view(self.avail_user_pos(), self._notify_samples)
            user_pos = self.current_user_pos
            length = self.notify_size
            return self.buffer[:, user_pos:user_pos+length], self.cpu_to_gpu_buffer, self.gpu_to_cpu_buffer
    

class CUDAKernel:
    device : "CUDADevice" = None
    function_name : str = None
    src : str = None
    arch : str = None
    threads_per_block : int = 0
    num_blocks : int = 0
    module : cuda.CUmodule = None
    kernel : cuda.CUfunction = None

    def __init__(self, device : "CUDADevice", function_name : str, src : str, num_blocks : int = 1, threads_per_block : int = 1024, arch : str = None):
        """
        Initialize a CUDA kernel object.

        Parameters
        ----------
        device : CUDADevice
            CUDADevice object.
        function_name : str
            Name of the kernel function. This name needs to be unique.
        src : str
            CUDA source code to compile.
        num_blocks : int (optional)
            Number of blocks. Default is 1.
        threads_per_block : int (optional)
            Number of threads per block. Default is 1024.
        arch : str (optional)
            Architecture of the GPU device. Default is None.
        """

        self.device = device
        self.function_name = function_name
        self.src = src
        self.num_blocks = num_blocks
        self.threads_per_block = threads_per_block
        self.arch = arch
        self.module = None
        self.kernel = None
        self.compile()

    def compile(self, src : str = None):
        """
        Compile the CUDA source code to a kernel function.

        Parameters
        ----------
        src : str (optional)
            CUDA source code to compile.
        """
        if src is None and self.src is None:
            raise ValueError("No source code provided")
        
        # Create program
        prog = checkCudaErrors(nvrtc.nvrtcCreateProgram(str.encode(self.src), str.encode("{}.cu".format(self.function_name)), 0, [], []))

        if self.arch is None:
            major = checkCudaErrors(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMajor, self.device.index))
            minor = checkCudaErrors(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrComputeCapabilityMinor, self.device.index))
            nvrtc_major, nvrtc_minor = checkCudaErrors(nvrtc.nvrtcVersion())
            use_cubin = (nvrtc_minor >= 1)
            prefix = 'sm' if use_cubin else 'compute'
            # prefix = 'compute'
            arch_arg = bytes(f'--gpu-architecture={prefix}_{major}{minor}', 'ascii')
        else:
            arch_arg = bytes(f'--gpu-architecture={self.arch}', 'ascii')
        
        # Compile program
        opts = [b"--fmad=true", arch_arg]
        checkCudaErrors(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        
        # Get PTX from compilation
        ptxSize = checkCudaErrors(nvrtc.nvrtcGetPTXSize(prog))
        ptx = b" " * ptxSize
        checkCudaErrors(nvrtc.nvrtcGetPTX(prog, ptx))
        
        # Load PTX as module data and retrieve function
        ptx = np.char.array(ptx)
        # Note: Incompatible --gpu-architecture would be detected here
        self.module = checkCudaErrors(cuda.cuModuleLoadData(ptx.ctypes.data))
        self.kernel = self.get_function(self.function_name)
        
        return self.kernel

    def get_function(self, function_name : str):
        """
        Get a function from the CUDA module.

        Parameters
        ----------
        function_name : str
            Name of the function.

        Returns
        -------
        cuda.CUfunction
            CUDA function.
        """

        return checkCudaErrors(cuda.cuModuleGetFunction(self.module, bytes(function_name, 'ascii')))
    
    def launch(self, kernelParams, gridDim = None, blockDim = None, sharedMemBytes = 0, stream = 0, extra = 0):
        """
        Launch the CUDA kernel function.

        Parameters
        ----------
        gridDim : int or tuple(int, int, int)
            Grid dimensions.
        blockDim : int or tuple(int, int, int)
            Block dimensions.
        kernelParams : list
            Kernel parameters.
        sharedMemBytes : int (optional)
            Shared memory bytes. Default is 0.
        stream : cuda.CUstream (optional)
            CUDA stream. Default is the cuda_object stream.
        extra : list (optional)
            Extra parameters. Default is None.
        """
        
        if gridDim is None:
            gridDim = (self.num_blocks, 1, 1)
        if blockDim is None:
            blockDim = (self.threads_per_block, 1, 1)
        if type(gridDim) is int:
            gridDim = (gridDim, 1, 1)
        if type(blockDim) is int:
            blockDim = (blockDim, 1, 1)
        if stream == 0:
            stream = self.device.cuda_object.stream
        
        # Parse the kernel arguments
        parameters = [[],[]]
        for param in kernelParams:
            if isinstance(param, CUDABuffer):
                parameters[0].append(param.buffer + param.offset_size)
                parameters[1].append(c_void_p)
            elif type(param) is int:
                parameters[0].append(param)
                parameters[1].append(c_int)
            elif type(param) is float:
                parameters[0].append(param)
                parameters[1].append(c_float)
            else:
                raise ValueError("CUDA Kernel parameters: Invalid parameter type")
        parameters = (tuple(parameters[0]), tuple(parameters[1]))
        # print(f"{param.offset_samples = }")
        # print(f"{parameters[0] = }")

        checkCudaErrors(cuda.cuLaunchKernel(self.kernel, *gridDim, *blockDim, sharedMemBytes, stream, parameters, extra))
    
class CUDADevice:

    cuda_object = None
    index : int = 0
    device = None
    device_prop = None
    rdma_supported : bool = False

    def __init__(self, cuda_object, index):
        self.cuda_object = cuda_object
        self.index = index
        self.device = checkCudaErrors(cudart.cudaSetDevice(self.index))
        self.device_prop = checkCudaErrors(cudart.cudaGetDeviceProperties(self.index))
        self.rdma_supported = checkCudaErrors(cudart.cudaDeviceGetAttribute(cudart.cudaDeviceAttr.cudaDevAttrGPUDirectRDMASupported, self.index))
    

class CUDA:
    """
    CUDA manager object.
    """

    card : Card = None
    channels : Channels = None
    device : CUDADevice = None
    stream : cuda.CUstream = None
    scapp : bool = False

    init_flags :int = 0

    def __init__(self, card : Card, channels : Channels, index : int = 0, scapp : bool = False):
        self.card = card
        self.channels = channels
        self.device = CUDADevice(self, index)
        self.stream = None
        self.scapp = scapp and self.device.rdma_supported

        self.init_flags = 0

        self.init()
        self.create_stream()

    def init(self):
        """
        Initialize the CUDA manager object. And free the CUDA device memory.
        """

        checkCudaErrors(cuda.cuInit(self.init_flags))
        checkCudaErrors(cudart.cudaFree(0))
    
    def create_stream(self, flags : int = 0):
        """
        Create a CUDA stream.

        Parameters
        ----------
        flags : int (optional)
            Stream flags. Default is 0.

        Returns
        -------
        cuda.CUstream
            CUDA stream.
        """

        self.stream = checkCudaErrors(cuda.cuStreamCreate(flags))
    
    def create_kernel(self, function_name : str, src : str, num_blocks = None, threads_per_block = None, arch : str = None):
        """
        Initialize a CUDA kernel object.

        Parameters
        ----------
        function_name : str
            Name of the kernel function. This name needs to be unique.
        src : str
            CUDA source code to compile.
        num_blocks : int (optional)
            Number of blocks. Default is 1.
        threads_per_block : int (optional)
            Number of threads per block. Default is 1024.
        arch : str (optional)
            Architecture of the GPU device. Default is None.

        Returns
        -------
        CUDAKernel
            CUDA kernel object.
        """

        return CUDAKernel(self.device, function_name, src, num_blocks, threads_per_block, arch)
    
    


