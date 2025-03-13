# -*- coding: utf-8 -*-
import time
import numpy as np

from .classes_card import Card
from .classes_data_transfer import DataTransfer

from spcm_core import spcm_dwDefTransfer_i64, c_void_p
from spcm_core.constants import *

from .classes_unit_conversion import UnitConversion
from . import units

from .classes_error_exception import SpcmException

_cuda_support = False
try:
    import cupy as cp
    from cuda import cuda, cudart, nvrtc
except ImportError:
    _cuda_support = False
else:
    _cuda_support = True

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
    
    
class SCAPPTransfer(DataTransfer):
    """
    Class for data transfer between the card and the host using the SCAPP API.
    """

    direction : Direction = None

    def __init__(self, card : Card, direction : Direction = Direction.Acquisition):
        if not _cuda_support:
            raise ImportError("CUDA support is not available. Please install the cupy and cuda-python packages.")
        super().__init__(card)
        scapp_feature = bool(self.card._features & SPCM_FEAT_SCAPP)
        if not scapp_feature:
            raise SpcmException(text="The card does not have the SCAPP option installed. SCAPP is a add-on feature that needs to be bought separately, please contact info@spec.de and ask for the SCAPP option for the card with serial number {}".format(self.card.sn()))
        self.direction = direction
        self.iterator_index = 0

    def allocate_buffer(self, num_samples : int) -> None:
        """
        Memory allocation for the buffer that is used for communicating with the card

        Parameters
        ----------
        num_samples : int | pint.Quantity = None
            use the number of samples an get the number of active channels and bytes per samples directly from the card
        """
        
        self.buffer_samples = UnitConversion.convert(num_samples, units.Sa, int)
        # Allocate RDMA buffer
        self.buffer = cp.empty((self.num_channels, self.buffer_samples), dtype = self.numpy_type(), order='F')
        flag = 1
        checkCudaErrors(cuda.cuPointerSetAttribute(flag, cuda.CUpointer_attribute.CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, self.buffer.data.ptr))
    
    def start_buffer_transfer(self, *args, buffer_type=SPCM_BUF_DATA, direction=None, notify_samples=None, transfer_offset=None, transfer_length=None, exception_num_samples=False) -> None:
        """
        Setup an RDMA transfer.

        Parameters
        ----------
        *args : list
            Additional commands that are send to the card.
        direction : int
            the direction of the transfer
        notify_samples : int
            Size of the part of the buffer that is used for notifications.
        transfer_offset : int
            the offset of the transfer
        transfer_length : int
            Total length of the transfer buffer.
        """

        # only change this locally
        if notify_samples is not None:
            notify_size = notify_samples * self.num_channels * self.bytes_per_sample
        else:
            notify_size = self.notify_size
        transfer_offset = UnitConversion.convert(transfer_offset, units.Sa, int)
        transfer_length = UnitConversion.convert(transfer_length, units.Sa, int)

        if self.buffer is None: 
            raise SpcmException(text="No buffer defined for transfer")
        if buffer_type: 
            self.buffer_type = buffer_type
        if direction is None:
            if self.direction == Direction.Acquisition:
                direction = SPCM_DIR_CARDTOGPU
            elif self.direction == Direction.Generation:
                direction = SPCM_DIR_GPUTOCARD
            else:
                raise SpcmException(text="Please define a direction for transfer (SPCM_DIR_CARDTOGPU or SPCM_DIR_GPUTOCARD)")
        
        if self._notify_samples != 0 and np.remainder(self.buffer_samples, self._notify_samples) and exception_num_samples:
            raise SpcmException("The number of samples needs to be a multiple of the notify samples.")

        if transfer_offset:
            transfer_offset_bytes = self.samples_to_bytes(transfer_offset)
        else:
            transfer_offset_bytes = 0

        self.buffer_samples = transfer_length

        # we define the buffer for transfer
        self.card._print("Starting the DMA transfer and waiting until data is in board memory")
        self.card._check_error(spcm_dwDefTransfer_i64(self.card._handle, self.buffer_type, direction, notify_size, c_void_p(self.buffer.data.ptr), transfer_offset_bytes, self.buffer_size))

        # Execute additional commands if available
        if args:
            cmd = 0
            for arg in args:
                cmd |= arg
            self.card.cmd(cmd)
            self.card._print("... SCAPP data transfer started")
    