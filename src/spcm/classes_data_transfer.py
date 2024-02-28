# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from .constants import *

from .pyspcm import c_void_p, spcm_dwDefTransfer_i64

from .classes_functionality import CardFunctionality

from .classes_error_exception import SpcmException, SpcmTimeout

class DataTransfer(CardFunctionality):
    """
    A high-level class to control Data Transfer to and from Spectrum Instrumentation cards.

    This class is an iterator class that implements the functions `__iter__` and `__next__`.
    This allows the user to supply the class to a for loop and iterate over the data that 
    is transferred from or to the card. Each iteration will return a numpy array with a data
    block of size `notify_samples`. In case of a digitizer you can read the data from that 
    block and process it. In case of a generator you can write data to the block and it's 
    then transferred.

    For more information about what setups are available, please have a look at the user manual
    for your specific card.

    Parameters
    ----------
    `buffer` : NDArray[np.int_]
        numpy object that can be used to write data into the spcm buffer
    `buffer_size`: int
        defines the size of the current buffer shared between the PC and the card
    `buffer_type`: int
        defines the type of data in the buffer that is used for the transfer
    `num_channels`: int
        defines the number of channels that are used for the transfer
    `bytes_per_sample`: int
        defines the number of bytes per sample
    `bits_per_sample`: int
        defines the number of bits per sample

    """
    # public
    buffer_size : int = 0
    buffer_type : int
    num_channels : int = 0
    bytes_per_sample : int = 0
    bits_per_sample : int = 0
    _notify_samples : int = 0

    @property
    def buffer(self) -> npt.NDArray[np.int_]:
        """The numpy buffer object that interfaces the Card and can be written and read from"""
        return self._np_buffer
    
    @buffer.setter
    def buffer(self, value) -> None:
        self._np_buffer = value
    
    @buffer.deleter
    def buffer(self) -> None:
        del self._np_buffer

    # private
    _memory_size : int = 0
    _c_buffer = None # Internal numpy ctypes buffer object
    _buffer_alignment : int = 4096
    _np_buffer : npt.NDArray[np.int_] # Internal object on which the getter setter logic is working
    _8bit_mode : bool = False

    def __init__(self, card, *args, **kwargs) -> None:
        """
        Initialize the DataTransfer object with a card object and additional arguments

        Parameters
        ----------
        card : Card
            the card object that is used for the data transfer
        *args : list
            list of additional arguments
        **kwargs : dict
            dictionary of additional keyword arguments
        """
        
        super().__init__(card, *args, **kwargs)
        self.buffer_type = SPCM_BUF_DATA
        self._bytes_per_sample()
        self._bits_per_sample()
        self.num_channels = self.card.active_channels()

    def memory_size(self, memory_size : int = None) -> int:
        """
        Sets the memory size in samples per channel. The memory size setting must be set before transferring 
        data to the card. (see register `SPC_MEMSIZE` in the manual)
        
        Parameters
        ----------
        memory_size : int
            the size of the memory in Bytes
        """

        if memory_size is not None:
            self.card.set_i(SPC_MEMSIZE, memory_size)
        self._memory_size = self.card.get_i(SPC_MEMSIZE)
        return self._memory_size
    
    def notify_samples(self, notify_samples : int) -> None:
        """
        Set the number of samples to notify the user about
        
        Parameters
        ----------
        notify_samples : int
            the number of samples to notify the user about
        """
        self._notify_samples = notify_samples
    
    def loops(self, loops : int = None) -> int:
        """
        Set the number of times the memory is replayed. If set to zero the generation will run continuously until it is 
        stopped by the user.  (see register `SPC_LOOPS` in the manual)
        
        Parameters
        ----------
        loops : int
            the number of loops that the card should perform
        """

        if loops is not None:
            self.card.set_i(SPC_LOOPS, loops)
        return self.card.get_i(SPC_LOOPS)

    def _bits_per_sample(self) -> int:
        """
        Get the number of bits per sample

        Returns
        -------
        int
            number of bits per sample
        """
        if self._8bit_mode:
            return 8
        self.bits_per_sample = self.card.bits_per_sample()
    
    def _bytes_per_sample(self) -> int:
        """
        Get the number of bytes per sample

        Returns
        -------
        int
            number of bytes per sample
        """
        if self._8bit_mode:
            return 1
        self.bytes_per_sample = self.card.bytes_per_sample()
    
    def pre_trigger(self, num_samples : int = None) -> int:
        """
        Set the number of pre trigger samples (see register `SPC_PRETRIGGER` in the manual)
        
        Parameters
        ----------
        num_samples : int
            the number of pre trigger samples
        """

        if num_samples is not None:
            self.card.set_i(SPC_PRETRIGGER, num_samples)
        return self.card.get_i(SPC_PRETRIGGER)
    
    def post_trigger(self, num_samples : int = None) -> int:
        """
        Set the number of post trigger samples (see register `SPC_POSTTRIGGER` in the manual)
        
        Parameters
        ----------
        num_samples : int
            the number of post trigger samples
        """

        if num_samples is not None:
            self.card.set_i(SPC_POSTTRIGGER, num_samples)
        return self.card.get_i(SPC_POSTTRIGGER)
    
    def allocate_buffer(self, num_samples : int) -> None:
        """
        Memory allocation for the buffer that is used for communicating with the card

        Parameters
        ----------
        num_samples : int = None
            use the number of samples an get the number of active channels and bytes per samples directly from the card
        """
        
        sample_type = self.numpy_type()

        if self.bits_per_sample > 1:
            self.buffer_size = num_samples * self.bytes_per_sample * self.num_channels
        else:
            self.buffer_size = num_samples * self.num_channels // 8

        dwMask = self._buffer_alignment - 1

        item_size = sample_type(0).itemsize
        # allocate a buffer (numpy array) for DMA transfer: a little bigger one to have room for address alignment
        databuffer_unaligned = np.empty(((self._buffer_alignment + self.buffer_size) // item_size, ), dtype = sample_type)   # half byte count at int16 sample (// = integer division)
        # two numpy-arrays may share the same memory: skip the begin up to the alignment boundary (ArrayVariable[SKIP_VALUE:])
        # Address of data-memory from numpy-array: ArrayVariable.__array_interface__['data'][0]
        start_pos_samples = ((self._buffer_alignment - (databuffer_unaligned.__array_interface__['data'][0] & dwMask)) // item_size)
        self.buffer = databuffer_unaligned[start_pos_samples:start_pos_samples + (self.buffer_size // item_size)]   # byte address but int16 sample: therefore / 2
        if self.bits_per_sample > 1:
            self.buffer = self.buffer.reshape((self.num_channels, num_samples), order='F')
    
    def start_buffer_transfer(self, *args, buffer_type=SPCM_BUF_DATA, direction=None, notify_samples=0, transfer_offset=0, transfer_length=None) -> None:
        """
        Start the transfer of the data to or from the card  (see the API function `spcm_dwDefTransfer_i64` in the manual)
        
        Parameters
        ----------
        *args : list
            list of additonal arguments that are added as flags to the start dma command
        buffer_type : int
            the type of buffer that is used for the transfer
        direction : int
            the direction of the transfer
        notify_samples : int
            the number of samples to notify the user about
        transfer_offset : int
            the offset of the transfer
        transfer_length : int
            the length of the transfer

        Raises
        ------
        SpcmException
        """

        if self.buffer is None: 
            raise SpcmException(text="No buffer defined for transfer")
        if buffer_type: 
            self.buffer_type = buffer_type
        if direction is None:
            if self.function_type == SPCM_TYPE_AI or self.function_type == SPCM_TYPE_DI:
                direction = SPCM_DIR_CARDTOPC
            elif self.function_type == SPCM_TYPE_AO or self.function_type == SPCM_TYPE_DO:
                direction = SPCM_DIR_PCTOCARD
            else:
                raise SpcmException(text="Please define a direction for transfer (SPCM_DIR_CARDTOPC or SPCM_DIR_PCTOCARD)")
        
        notify_size = 0
        if notify_samples: 
            self._notify_samples = notify_samples
        if self._notify_samples: 
            notify_size = self._notify_samples * self.bytes_per_sample * self.num_channels

        if transfer_offset:
            transfer_offset_bytes = transfer_offset * self.bytes_per_sample * self.num_channels
        else:
            transfer_offset_bytes = 0

        if transfer_length is not None: 
            transfer_length_bytes = transfer_length * self.bytes_per_sample * self.num_channels
        else:
            transfer_length_bytes = self.buffer_size
        
        # we define the buffer for transfer and start the DMA transfer
        self.card._print("Starting the DMA transfer and waiting until data is in board memory")
        self._c_buffer = self.buffer.ctypes.data_as(c_void_p)
        spcm_dwDefTransfer_i64(self.card._handle, self.buffer_type, direction, notify_size, self._c_buffer, transfer_offset_bytes, transfer_length_bytes)
        cmd = 0
        for arg in args:
            cmd |= arg
        self.card.cmd(cmd)
        self.card._print("... data transfer started")
    
    def avail_card_len(self, lAvailSamples : int = 0) -> None:
        """
        Set the amount of data that has been read out of the data buffer (see register `SPC_DATA_AVAIL_CARD_LEN` in the manual)

        Parameters
        ----------
        lAvailSamples : int
            the amount of data that is available for reading
        """

        self.card.set_i(SPC_DATA_AVAIL_CARD_LEN, lAvailSamples * self.bytes_per_sample * self.num_channels)
    
    def avail_user_pos(self, bytes : bool = False) -> int:
        """
        Get the current position of the pointer in the data buffer (see register `SPC_DATA_AVAIL_USER_POS` in the manual)

        Returns
        -------
        int
            pointer position
        """

        user_pos = self.card.get_i(SPC_DATA_AVAIL_USER_POS)
        if not bytes:
            user_pos = user_pos // self.bytes_per_sample // self.num_channels
        return user_pos
    
    def avail_user_len(self, bytes : bool = False) -> int:
        """
        Get the current length of the data in the data buffer (see register `SPC_DATA_AVAIL_USER_LEN` in the manual)

        Returns
        -------
        int
            data length available
        """

        user_len = self.card.get_i(SPC_DATA_AVAIL_USER_LEN)
        if not bytes:
            user_len = user_len // self.bytes_per_sample // self.num_channels
        return user_len
    
    def fill_size_promille(self) -> int:
        """
        Get the fill size of the data buffer (see register `SPC_FILLSIZEPROMILLE` in the manual)

        Returns
        -------
        int
            fill size
        """

        return self.card.get_i(SPC_FILLSIZEPROMILLE)
    
    def wait_dma(self) -> None:
        """
        Wait for the DMA transfer to finish (see register `M2CMD_DATA_WAITDMA` in the manual)
        """

        self.card.cmd(M2CMD_DATA_WAITDMA)
    wait = wait_dma

    def numpy_type(self) -> npt.NDArray[np.int_]:
        """
        Get the type of numpy data from number of bytes

        Returns
        -------
        numpy data type
            the type of data that is used by the card
        """
        if self._8bit_mode:
            return np.uint8
        if self.bits_per_sample == 1:
            if self.num_channels <= 8:
                return np.uint8
            elif self.num_channels <= 16:
                return np.uint16
            elif self.num_channels <= 32:
                return np.uint32
            return np.uint64
        if self.bits_per_sample <= 8:
            return np.int8
        elif self.bits_per_sample <= 16:
            return np.int16
        elif self.bits_per_sample <= 32:
            return np.int32
        return np.int64

    # 8-bit mode
    def data_conversion(self, mode : int = None) -> int:
        """
        Set the data conversion mode (see register `SPC_DATACONVERSION` in the manual)
        
        Parameters
        ----------
        mode : int
            the data conversion mode
        """

        if mode is not None:
            self.card.set_i(SPC_DATACONVERSION, mode)
        mode = self.card.get_i(SPC_DATACONVERSION)
        self._8bit_mode = not (mode == SPCM_DC_NONE)
        return self._8bit_mode
    
    def avail_data_conversion(self) -> int:
        """
        Get the available data conversion modes (see register `SPC_AVAILDATACONVERSION` in the manual)

        Returns
        -------
        int
            the available data conversion modes
        """
        return self.card.get_i(SPC_AVAILDATACONVERSION)
    
    # Iterator methods

    _max_timeout = 64

    _to_transfer_samples = 0
    _current_samples = 0

    def to_transfer_samples(self, samples: int) -> None:
        """
        This method sets the number of samples to transfer

        Parameters
        ----------
        samples : int
            the number of samples to transfer
        """
        self._to_transfer_samples = samples
    
    def __iter__(self):
        """
        This method is called when the iterator is initialized

        Returns
        -------
        DataIterator
            the iterator itself
        """
        return self
    
    def __next__(self) -> npt.ArrayLike:
        """
        This method is called when the next element is requested from the iterator

        Returns
        -------
        npt.ArrayLike
            the next data block
        
        Raises
        ------
        StopIteration
        """
        timeout_counter = 0
        while True:
            try:
                self.wait_dma()
            except SpcmTimeout:
                self.card._print("... Timeout ({})".format(timeout_counter), end='\r')
                timeout_counter += 1
                if timeout_counter > self._max_timeout:
                    raise StopIteration
            else:
                user_len = self.avail_user_len()
                user_pos = self.avail_user_pos()

                self._current_samples += self._notify_samples
                if self._to_transfer_samples != 0 and self._to_transfer_samples < self._current_samples:
                    raise StopIteration

                fill_size = self.fill_size_promille()
                self.card._print("Fill size: {}%  Pos:{:08x} Len:{:08x} Total:{:.2f} MiS / {:.2f} MiS".format(fill_size/10, user_pos, user_len, self._current_samples / MEBI(1), self._to_transfer_samples / MEBI(1)), end='\r')

                self.avail_card_len(self._notify_samples)
                return self.buffer[:, user_pos:user_pos+self._notify_samples]