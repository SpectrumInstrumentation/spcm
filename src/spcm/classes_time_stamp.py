# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from spcm_core.constants import *
from spcm_core import c_void_p, spcm_dwDefTransfer_i64

from .classes_data_transfer import DataTransfer
from .classes_unit_conversion import UnitConversion
from . import units

class TimeStamp(DataTransfer):
    """a class to control Spectrum Instrumentation cards with the timestamp functionality

    For more information about what setups are available, please have a look at the user manual
    for your specific card

    Parameters
    ----------
    ts_mode : int
    transfer_mode : int
    bits_per_ts : int = 0
    bytes_per_ts : int = 16
    """
    ts_mode : int
    transfer_mode : int

    bits_per_ts : int = 0
    bytes_per_ts : int = 16
    
    _notify_timestamps : int = 0
    _to_transfer_timestamps : int = 0

    def __init__(self, card, *args, **kwargs) -> None:
        """
        Initialize the TimeStamp object with a card object

        Parameters
        ----------
        card : Card
            a card object that is used to control the card
        """
        
        super().__init__(card, *args, **kwargs)
        self.buffer_type = SPCM_BUF_TIMESTAMP
        self.bits_per_ts = self.bytes_per_ts * 8
    
    def cmd(self, *args) -> None:
        """
        Execute spcm timestamp commands (see register 'SPC_TIMESTAMP_CMD' in chapter `Timestamp` in the manual)
    
        Parameters
        ----------
        *args : int
            The different timestamp command flags to be executed.
        """

        cmd = 0
        for arg in args:
            cmd |= arg
        self.card.set_i(SPC_TIMESTAMP_CMD, cmd)
    
    def reset(self) -> None:
        """Reset the timestamp counter (see command 'SPC_TS_RESET' in chapter `Timestamp` in the manual)"""
        self.cmd(SPC_TS_RESET)
    
    def mode(self, mode : int, *args : list[int]) -> None:
        """
        Set the mode of the timestamp counter (see register 'SPC_TIMESTAMP_CMD' in chapter `Timestamp` in the manual)

        Parameters
        ----------
        mode : int
            The mode of the timestamp counter
        *args : list[int]
            List of additional commands send with setting the mode
        """
        self.ts_mode = mode
        self.cmd(self.ts_mode, *args)

    def notify_timestamps(self, notify_timestamps : int) -> None:
        """
        Set the number of timestamps to notify the user about
        
        Parameters
        ----------
        notify_timestamps : int
            the number of timestamps to notify the user about
        """
        self._notify_timestamps = notify_timestamps
    
    def allocate_buffer(self, num_timestamps : int) -> None:
        """
        Allocate the buffer for the timestamp data transfer
        
        Parameters
        ----------
        num_timestamps : int
            The number of timestamps to be allocated
        """

        num_timestamps = UnitConversion.convert(num_timestamps, units.S, int)
        self.buffer_size = num_timestamps * self.bytes_per_ts

        dwMask = self._buffer_alignment - 1

        sample_type = np.int64
        item_size = sample_type(0).itemsize
        # allocate a buffer (numpy array) for DMA transfer: a little bigger one to have room for address alignment
        databuffer_unaligned = np.zeros(((self._buffer_alignment + self.buffer_size) // item_size, ), dtype = sample_type)   # half byte count at int16 sample (// = integer division)
        # two numpy-arrays may share the same memory: skip the begin up to the alignment boundary (ArrayVariable[SKIP_VALUE:])
        # Address of data-memory from numpy-array: ArrayVariable.__array_interface__['data'][0]
        start_pos_samples = ((self._buffer_alignment - (databuffer_unaligned.__array_interface__['data'][0] & dwMask)) // item_size)
        self.buffer = databuffer_unaligned[start_pos_samples:start_pos_samples + (self.buffer_size // item_size)]   # byte address but int16 sample: therefore / 2
        self.buffer = self.buffer.reshape((num_timestamps, 2), order='C') # array items per timestamp, because the maximum item size is 8 bytes = 64 bits
    
    def start_buffer_transfer(self, *args, direction=SPCM_DIR_CARDTOPC, notify_timestamps = None, transfer_offset=0, transfer_length=None) -> None:
        """
        Start the transfer of the timestamp data to the card
        
        Parameters
        ----------
        *args : list
            list of additonal arguments that are added as flags to the start dma command
        """
        
        notify_size = 0
        if notify_timestamps is not None: 
            self._notify_timestamps = notify_timestamps
        if self._notify_timestamps: 
            notify_size = self._notify_timestamps * self.bytes_per_ts

        if transfer_offset:
            transfer_offset_bytes = transfer_offset * self.bytes_per_ts
        else:
            transfer_offset_bytes = 0

        if transfer_length is not None: 
            transfer_length_bytes = transfer_length * self.bytes_per_ts
        else:
            transfer_length_bytes = self.buffer_size


        # we define the buffer for transfer and start the DMA transfer
        self.card._print("Starting the Timestamp transfer and waiting until data is in board memory")
        self._c_buffer = self.buffer.ctypes.data_as(c_void_p)
        spcm_dwDefTransfer_i64(self.card._handle, self.buffer_type, direction, notify_size, self._c_buffer, transfer_offset_bytes, transfer_length_bytes)
        cmd = 0
        for arg in args:
            cmd |= arg
        self.card.cmd(cmd)
        self.card._print("... timestamp data transfer started")
    
    def avail_card_len(self, num_timestamps : int) -> None:
        """
        Set the amount of timestamps that is available for reading of the timestamp buffer (see register 'SPC_TS_AVAIL_CARD_LEN' in chapter `Timestamp` in the manual)

        Parameters
        ----------
        num_timestamps : int
            the amount of timestamps that is available for reading
        """
        card_len = num_timestamps * self.bytes_per_ts
        self.card.set_i(SPC_TS_AVAIL_CARD_LEN, card_len)
    
    def avail_user_pos(self) -> int:
        """
        Get the current position of the pointer in the timestamp buffer (see register 'SPC_TS_AVAIL_USER_POS' in chapter `Timestamp` in the manual)

        Returns
        -------
        int
            pointer position in timestamps
        """
        return self.card.get_i(SPC_TS_AVAIL_USER_POS) // self.bytes_per_ts
    
    def avail_user_len(self) -> int:
        """
        Get the current length of the data in timestamps in the timestamp buffer (see register 'SPC_TS_AVAIL_USER_LEN' in chapter `Timestamp` in the manual)

        Returns
        -------
        int
            data length available in number of timestamps
        """
        return self.card.get_i(SPC_TS_AVAIL_USER_LEN) // self.bytes_per_ts
    

    # Iterator methods
    _max_polling = 64

    def to_transfer_timestamps(self, timestamps: int) -> None:
        """
        This method sets the number of timestamps to transfer

        Parameters
        ----------
        timestamps : int
            the number of timestamps to transfer
        """
        self._to_transfer_timestamps = timestamps

    def poll(self, polling_length : int = 0) -> npt.ArrayLike:
        """
        This method is called when polling for timestamps

        Parameters
        ----------
        polling_length : int = 0
            the number of timestamps to poll and wait for

        Returns
        -------
        npt.ArrayLike
            the next data block
        """

        while True:
            user_pos = self.avail_user_pos()
            user_len = self.avail_user_len()
            if not polling_length and user_len >= 1:
                self.avail_card_len(user_len)
                return self.buffer[user_pos:user_pos+user_len, :]
            elif user_len >= polling_length:
                self.avail_card_len(polling_length)
                return self.buffer[user_pos:user_pos+polling_length, :]
