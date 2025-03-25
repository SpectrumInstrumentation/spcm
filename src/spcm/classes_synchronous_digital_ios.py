# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from spcm_core.constants import *

from .classes_card import Card
from .classes_channels import Channels
from .classes_data_transfer import DataTransfer
from .classes_multi_purpose_ios import MultiPurposeIO, MultiPurposeIOs

from .classes_error_exception import SpcmException


class SynchronousDigitalIOs(MultiPurposeIOs):
    """a higher-level abstraction of the CardFunctionality class to implement the Card's synchronuous digital I/O functionality"""

    data_transfer : DataTransfer = None
    channels : Channels = None
    buffer : npt.NDArray = None
    item_size : int = None
    connections : dict = {}
    lowest_used_channel_bit : dict = {}
    setups : dict = {}
    channel_mask = {}

    def __init__(self, data_transfer, channels, digin2bit : bool = False, *args, **kwargs) -> None:
        """
        Constructor for the SynchronousDigitalIOs class
    
        Parameters
        ----------
        data_transfer : DataTransfer
            The card object to communicate with the card
        channels : Channels
            The channels of the card
        digin2bit : bool = False (44xx only)
            If True, the digital input bits will be encoded as the 2 highest valued bits of the available channels
        """

        self.data_transfer = data_transfer
        self.card = data_transfer.card
        self.channels = channels
        self.item_size = data_transfer.bytes_per_sample
        self.lowest_used_channel_bit = {}
        for channel in self.channels:
            self.lowest_used_channel_bit[int(channel)] = self.item_size * 8
            self.channel_mask[int(channel)] = 0
        self.connections = {}
        self.setups = {}
        self._max_buffer_index = -1
        self._buffer_iterator_index = -1
        self.num_xio_lines = self.get_num_xio_lines()

        # For the 22xx and 23xx families there are only fixed xio lines possible
        if self.card.family() == 0x22 or self.card.family() == 0x23 or self.card.family() == 0x44:
            digin2bit &= (self.card.family() == 0x44)
            if digin2bit:
                self.x_mode(0, SPCM_XMODE_DIGIN2BIT)
            else:
                self.x_mode(0, SPCM_XMODE_DIGIN) # Turn-on one line turns on all the xio lines
            num_channels = len(channels)
            if digin2bit:
                num_channels *= 2
            num_buffers = np.min([num_channels, self.num_xio_lines])
            self._allocate_buffer(num_buffers)
            xios  = [0,1,2,0]
            xios2 = [1,2,0,1]
            if (self.card.family() == 0x22 or self.card.family() == 0x23) and (self.card.card_type() & TYP_CHMASK == 0x3 or self.card.card_type() & TYP_CHMASK == 0x1):
                if self.card.get_i(SPC_SAMPLERATE) > 1.25e9: # There is a bug in the 22xx family with one channel cards, see page 132 in the manual
                    print("Warning: this is the bug in the 22xx family with one channel cards, see page 132 in the manual")
                    self.lowest_used_channel_bit[1] -= 1
                    self.connections[0] = {'channel': 1, 'bit': self.lowest_used_channel_bit[1], 'xios': [0]}
                    self.lowest_used_channel_bit[0] -= 1
                    self.connections[1] = {'channel': 0, 'bit': self.lowest_used_channel_bit[0], 'xios': [2]}
                else:
                    self.lowest_used_channel_bit[0] -= 1
                    self.connections[0] = {'channel': 0, 'bit': self.lowest_used_channel_bit[0], 'xios': [0]}
                    self.lowest_used_channel_bit[1] -= 1
                    self.connections[1] = {'channel': 1, 'bit': self.lowest_used_channel_bit[1], 'xios': [2]}
            else:
                for channel in self.channels:
                    self.lowest_used_channel_bit[int(channel)] -= 1
                    self.connections[xios[int(channel)]] = {'channel': int(channel), 'bit': self.lowest_used_channel_bit[int(channel)], 'xios': [xios[int(channel)]]}
                    if digin2bit:
                        self.lowest_used_channel_bit[int(channel)] -= 1
                        self.connections[xios2[int(channel)]] = {'channel': int(channel), 'bit': self.lowest_used_channel_bit[int(channel)], 'xios': [xios2[int(channel)]]}
            pass
            
    def __str__(self) -> str:
        """
        String representation of the SynchronousDigitalIO class
    
        Returns
        -------
        str
            String representation of the SynchronousDigitalIO class
        """
        
        return f"SynchronousDigitalIOs(data_transfer={self.data_transfer})"
    
    __repr__ = __str__

    def __len__(self) -> int:
        """
        Get the number of buffers

        Returns
        -------
        int
            The number of buffers
        """

        return self._num_buffers

    _buffer_iterator_index = -1
    def __iter__(self) -> "SynchronousDigitalIOs":
        """Define this class as an iterator"""
        self._buffer_iterator_index = -1
        return self
        
    def __getitem__(self, index : int):
        """
        Get the buffer at the specified index

        Parameters
        ----------
        index : int
            buffer index
        """

        return self.buffer[index, :]

    def __next__(self) -> MultiPurposeIO:
        """
        This method is called when the next element is requested from the iterator

        Returns
        -------
        MultiPurposeIO
            The next xio line in the iterator
        
        Raises
        ------
        StopIteration
        """
        self._buffer_iterator_index += 1
        if self._buffer_iterator_index >= self._num_buffers:
            self._buffer_iterator_index = -1
            raise StopIteration
        return self.buffer[self._buffer_iterator_index, :]
    
    def current_xio(self) -> int:
        """
        Get the current xio line

        Returns
        -------
        int
            current xio line
        """

        return self.connections[self._buffer_iterator_index]['xios'][0]

    def x_mode(self, xio : int, mode : int = None) -> int:
        """
        Sets the mode of the digital input/output of the card (see register 'SPCM_X0_MODE' in chapter `Multi Purpose I/O Lines` in the manual)
    
        Parameters
        ----------
        mode : int
            The mode of the digital input/output
        
        Returns
        -------
        int
            The mode of the digital input/output
        """

        if mode is not None:
            self.card.set_i(SPCM_X0_MODE + xio, mode)
        return self.card.get_i(SPCM_X0_MODE + xio)

    def digmode(self, channel : int, mode : int = None) -> int:
        """
        Sets the mode of the digital input of the card (see register 'SPC_DIGMODE0' in chapter `Synchronous digital inputs` in the manual)
    
        Parameters
        ----------
        mode : int
            The mode of the digital input
        
        Returns
        -------
        int
            The mode of the digital input
        """

        if mode is not None:
            self.card.set_i(SPC_DIGMODE0 + channel, mode)
        return self.card.get_i(SPC_DIGMODE0 + channel)

    def allocate_buffer(self, num_buffers : int) -> npt.NDArray:
        """
        Allocate the buffers for the digital input/output lines of the card

        Parameters
        ----------
        num_buffers : int
            The number of buffers to allocate
        """

        if self.card.family() == 0x22 or self.card.family() == 0x23 or self.card.family() == 0x44:
            print("The 22xx, 23xx and 44xx families only support fixed xio lines, allocate_buffer() is not necessary and doesn't change the system")
        else:
            self._allocate_buffer(num_buffers)
        return self.buffer
    
    def _allocate_buffer(self, num_buffers : int) -> npt.NDArray:
        """
        Allocate the buffers for the digital input/output lines of the card

        Parameters
        ----------
        num_buffers : int
            The number of buffers to allocate
        """

        self._num_buffers = num_buffers
        # TODO make this usable with multiple replay/recording
        self.buffer = np.zeros((self._num_buffers, self.data_transfer.buffer.shape[-1]), dtype=self.data_transfer.numpy_type())
        return self.buffer

    def setup(self, buffer_index : int, channel, xios : list) -> None:
        """
        Register the connection of a buffer with the corresponding bit in a specific channel of the card

        Parameters
        ----------
        buffer_index : int
            The index of the buffer to be used
        channel : Channel | int
            The channel index
        xios : list | int
            The xio lines that the buffer is connected to
        """

        if self.card.family() == 0x22 or self.card.family() == 0x23 or self.card.family() == 0x44:
            print("The 22xx, 23xx and 44x family only support fixed xio lines, setup() is not necessary and doesn't change the system")
            return

        # Define the buffer
        if isinstance(xios, int):
            xios = [xios]
        self.lowest_used_channel_bit[int(channel)] -= 1
        self.connections[buffer_index] = {'channel': int(channel), 'bit': self.lowest_used_channel_bit[int(channel)], 'xios': xios}

        # Setup the right xio mode
        if self.data_transfer.direction == Direction.Generation:
            bit_mask = SPCM_XMODE_DIGOUTSRC_BIT15 << (15 - self.lowest_used_channel_bit[int(channel)])
            channel_mask = SPCM_XMODE_DIGOUTSRC_CH0 << int(channel)
            for xio in xios:
                self.x_mode(xio, SPCM_XMODE_DIGOUT | channel_mask | bit_mask)
        elif self.data_transfer.direction == Direction.Acquisition:
            bit_mask = DIGMODEMASK_BIT15 >> 5*(15 - self.lowest_used_channel_bit[int(channel)])
            x_mask = SPCM_DIGMODE_X0 + (xios[0]) * (SPCM_DIGMODE_X1 - SPCM_DIGMODE_X0)
            self.channel_mask[int(channel)] |= bit_mask & x_mask
            self.x_mode(xios[0], SPCM_XMODE_DIGIN)
            self.digmode(int(channel), self.channel_mask[int(channel)])
        else:
            raise SpcmException(text="Please specify a data transfer direction: (Acquisition or Generation)")
    
    def process(self, no_shift : bool = False):
        """
        Process the digital input/output lines of the card

        Parameters
        ----------
        no_shift : bool
            If True, no bit shift will be applied
        """

        itemsize_bits = self.item_size * 8
        uint_type = self._int2uint(self.data_transfer.numpy_type())
        if self.data_transfer.direction == Direction.Generation:
            if not no_shift:
                for channel, lowest_bit in self.lowest_used_channel_bit.items():
                    bit_shift = itemsize_bits - lowest_bit
                    self.data_transfer.buffer[channel, :] = (self.data_transfer.buffer[channel, :].view(uint_type) >> bit_shift)

            for key, connection in self.connections.items():
                self.data_transfer.buffer[connection['channel'], :] |= self.buffer[key, :] << connection['bit']
        elif self.data_transfer.direction == Direction.Acquisition:
            for buffer_index in range(self._num_buffers):
                channel_index = int(self.connections[buffer_index]['channel'])
                self.buffer[buffer_index, :] = (self.data_transfer.buffer[channel_index, :] >> self.connections[buffer_index]['bit']) & 0x1

            if not no_shift:
                for channel, lowest_bit in self.lowest_used_channel_bit.items():
                    bit_shift = itemsize_bits - lowest_bit
                    self.data_transfer.buffer[channel, :] = (self.data_transfer.buffer[channel, :].view(uint_type) << bit_shift)
        else:
            raise SpcmException(text="Please specify a data transfer direction: (Acquisition or Generation)")
        
    def _int2uint(self, np_type : npt.DTypeLike) -> npt.DTypeLike:
        """
        Convert the integer data to unsigned integer data
        """

        if np_type == np.int8:
            return np.uint8
        elif np_type == np.int16:
            return np.uint16
        elif np_type == np.int32:
            return np.uint32
        elif np_type == np.int64:
            return np.uint64
        else:
            raise SpcmException(text="The data type is not supported for signed to unsigned conversion")
