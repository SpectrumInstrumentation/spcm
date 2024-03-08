# -*- coding: utf-8 -*-

from .constants import *

from .classes_card import Card
from .classes_card_stack import CardStack

from .classes_error_exception import SpcmException


class Channel:
    """A class to represent a channel of a card only used inside the Channels class in the list of channels"""

    card : Card
    index : int

    def __init__(self, index : int, card : Card) -> None:
        """
        Constructor of the Channel class
    
        Parameters
        ----------
        index : int
            The index of the channel
        card : Card
            The card of the channel
        """

        self.card = card
        self.index = index
    
    def __str__(self) -> str:
        """
        String representation of the Channel class
    
        Returns
        -------
        str
            String representation of the Channel class
        """
        
        return f"Channel({self.index}, {self.card})"
    
    __repr__ = __str__

    def __int__(self) -> int:
        """
        The Channel object acts like an int and returns the index of the channel and can also be used as the index in an array

        Returns
        -------
        int
            The index of the channel
        """
        return self.index
    __index__ = __int__
    
    def __add__(self, other):
        """
        The Channel object again acts like an int and returns the index of the channel plus the other value
        
        Parameters
        ----------
        other : int or float
            The value to be added to the index of the channel
        
        Returns
        -------
        int or float
            The index of the channel plus the other value
        """
        return self.index + other
    
    def enable(self, enable : bool = None) -> bool:
        """
        Enables the analog front-end of the channel of the card (see register `SPC_ENABLEOUT` in the manual)
    
        Parameters
        ----------
        enable : bool
            Turn-on (True) or off (False) the spezific channel

        Returns
        -------
        bool
            The enable state of the specific channel
        """

        if enable is not None:
            self.card.set_i(SPC_ENABLEOUT0 + (SPC_ENABLEOUT1 - SPC_ENABLEOUT0) * self.index, int(enable))
        return bool(self.card.get_i(SPC_ENABLEOUT0 + (SPC_ENABLEOUT1 - SPC_ENABLEOUT0) * self.index))
    enable_out = enable
    
    def path(self, value : int = None) -> int:
        """
        Sets the input path of the channel of the card (see register `SPC_PATH0` in the manual)
    
        Parameters
        ----------
        value : int
            The input path of the specific channel
        
        Returns
        -------
        int
            The input path of the specific channel
        """

        if value is not None:
            self.card.set_i(SPC_PATH0 + (SPC_PATH1 - SPC_PATH0) * self.index, value)
        return self.card.get_i(SPC_PATH0 + (SPC_PATH1 - SPC_PATH0) * self.index)
        
    def amp(self, value : int = None) -> int:
        """
        Sets the output/input range (amplitude) of the analog front-end of the channel of the card in mV (see register `SPC_AMP` in the manual)
    
        Parameters
        ----------
        value : int
            The output range (amplitude) of the specific channel in millivolts
            
        Returns
        -------
        int
            The output range (amplitude) of the specific channel in millivolts
        """

        if value is not None:
            self.card.set_i(SPC_AMP0 + (SPC_AMP1 - SPC_AMP0) * self.index, value)
        return self.card.get_i(SPC_AMP0 + (SPC_AMP1 - SPC_AMP0) * self.index)

    def offset(self, value : int = None) -> int:
        """
        Sets the offset of the analog front-end of the channel of the card in mV (see register `SPC_OFFSET` in the manual)
    
        Parameters
        ----------
        value : int
            The offset of the specific channel in millivolts
            
        Returns
        -------
        int
            The offset of the specific channel in millivolts
        """

        if value is not None:
            self.card.set_i(SPC_OFFS0 + (SPC_OFFS1 - SPC_OFFS0) * self.index, value)
        return self.card.get_i(SPC_OFFS0 + (SPC_OFFS1 - SPC_OFFS0) * self.index)

    def termination(self, value : int) -> None:
        """
        Sets the termination of the analog front-end of the channel of the card (see register `SPC_50OHM0` in the manual)
    
        Parameters
        ----------
        value : int
            The termination of the specific channel
        """

        self.card.set_i(SPC_50OHM0 + (SPC_50OHM1 - SPC_50OHM0) * self.index, value)

    def get_termination(self) -> int:
        """
        Gets the termination of the analog front-end of the channel of the card (see register `SPC_50OHM0` in the manual)
            
        Returns
        -------
        int
            The termination of the specific channel
        """

        return self.card.get_i(SPC_50OHM0 + (SPC_50OHM1 - SPC_50OHM0) * self.index)
    
    def coupling(self, value : int = None) -> int:
        """
        Sets the coupling of the analog front-end of the channel of the card (see register `SPC_ACDC0` in the manual)
    
        Parameters
        ----------
        value : int
            The coupling of the specific channel
            
        Returns
        -------
        int
            The coupling of the specific channel
        """

        if value is not None:
            self.card.set_i(SPC_ACDC0 + (SPC_ACDC1 - SPC_ACDC0) * self.index, value)
        return self.card.get_i(SPC_ACDC0 + (SPC_ACDC1 - SPC_ACDC0) * self.index)
    
    def coupling_offset_compensation(self, value : int = None) -> int:
        """
        Sets the coupling offset compensation of the analog front-end of the channel of the card (see register `SPC_ACDC_OFFS_COMPENSATION0` in the manual)
    
        Parameters
        ----------
        value : int
            The coupling offset compensation of the specific channel
            
        Returns
        -------
        int
            The coupling offset compensation of the specific channel
        """

        if value is not None:
            self.card.set_i(SPC_ACDC_OFFS_COMPENSATION0 + (SPC_ACDC_OFFS_COMPENSATION1 - SPC_ACDC_OFFS_COMPENSATION0) * self.index, value)
        return self.card.get_i(SPC_ACDC_OFFS_COMPENSATION0 + (SPC_ACDC_OFFS_COMPENSATION1 - SPC_ACDC_OFFS_COMPENSATION0) * self.index)
    
    def filter(self, value : int = None) -> int:
        """
        Sets the filter of the analog front-end of the channel of the card (see register `SPC_FILTER0` in the manual)
    
        Parameters
        ----------
        value : int
            The filter of the specific channel
            
        Returns
        -------
        int
            The filter of the specific channel
        """

        if value is not None:
            self.card.set_i(SPC_FILTER0 + (SPC_FILTER1 - SPC_FILTER0) * self.index, value)
        return self.card.get_i(SPC_FILTER0 + (SPC_FILTER1 - SPC_FILTER0) * self.index)
    
    def stop_level(self, value : int = None) -> int:
        """
        Usually the used outputs of the analog generation boards are set to zero level after replay. 
        This is in most cases adequate. In some cases it can be necessary to hold the last sample,
        to output the maximum positive level or maximum negative level after replay. The stoplevel will 
        stay on the defined level until the next output has been made. With this function
        you can define the behavior after replay (see register `SPC_CH0_STOPLEVEL` in the manual)
    
        Parameters
        ----------
        value : int
            The wanted stop behaviour

        Returns
        -------
        int
            The stop behaviour of the specific channel
        """

        if value is not None:
            self.card.set_i(SPC_CH0_STOPLEVEL + self.index * (SPC_CH1_STOPLEVEL - SPC_CH0_STOPLEVEL), value)
        return self.card.get_i(SPC_CH0_STOPLEVEL + self.index * (SPC_CH1_STOPLEVEL - SPC_CH0_STOPLEVEL))

    def custom_stop(self, value : int = None) -> int:
        """
        Allows to define a 16bit wide custom level per channel for the analog output to enter in pauses. The sample format is 
        exactly the same as during replay, as described in the „sample format“ section.
        When synchronous digital bits are replayed along, the custom level must include these as well and therefore allows to 
        set a custom level for each multi-purpose line separately. (see register `SPC_CH0_CUSTOM_STOP` in the manual)
    
        Parameters
        ----------
        value : int
            The custom stop value

        Returns
        -------
        int
            The custom stop value of the specific channel
        """

        if value is not None:
            self.card.set_i(SPC_CH0_CUSTOM_STOP + self.index * (SPC_CH1_CUSTOM_STOP - SPC_CH0_CUSTOM_STOP), value)
        return self.card.get_i(SPC_CH0_CUSTOM_STOP + self.index * (SPC_CH1_CUSTOM_STOP - SPC_CH0_CUSTOM_STOP))
    


class Channels:
    """
    a higher-level abstraction of the CardFunctionality class to implement the Card's channel settings
    """
    
    cards : list[Card] = []
    channels : list[Channel] = []
    num_channels : list[int] = []

    def __init__(self, card : Card = None, card_enable : int = None, stack : CardStack = None, stack_enable : list[int] = None) -> None:
        """
        Constructor of the Channels class

        Parameters
        ----------
        card : Card = None
            The card to be used
        card_enable : int = None
            The bitmask to enable specific channels
        stack : CardStack = None
            The card stack to be used
        stack_enable : list[int] = None
            The list of bitmasks to enable specific channels

        Raises
        ------
        SpcmException
            No card or card stack provided
        """

        self.cards = []
        self.channels = []
        self.num_channels = []
        if card is not None:
            self.cards.append(card)
            if card_enable is not None:
                self.channels_enable(enable_list=[card_enable])
            else:
                self.channels_enable(enable_all=True)
        elif stack is not None:
            self.cards = stack.cards
            if stack_enable is not None:
                self.channels_enable(enable_list=stack_enable)
            else:
                self.channels_enable(enable_all=True)
        else:
            raise SpcmException(text="No card or card stack provided")

    def __str__(self) -> str:
        """
        String representation of the Channels class
    
        Returns
        -------
        str
            String representation of the Channels class
        """
        
        return f"Channels()"
    
    __repr__ = __str__

    def __iter__(self) -> "Channels":
        """Define this class as an iterator"""
        return self
    
    def __getitem__(self, index : int) -> Channel:
        """
        This method is called to access the channel by index

        Parameters
        ----------
        index : int
            The index of the channel
        
        Returns
        -------
        Channel
            the channel at the specific index
        """

        return self.channels[index]
    
    _channel_iterator_index = -1
    def __next__(self) -> Channel:
        """
        This method is called when the next element is requested from the iterator

        Returns
        -------
        Channel
            the next available channel
        
        Raises
        ------
        StopIteration
        """
        self._channel_iterator_index += 1
        if self._channel_iterator_index >= len(self.channels):
            self._channel_iterator_index = -1
            raise StopIteration
        return self.channels[self._channel_iterator_index]
    
    def __len__(self) -> int:
        """Returns the number of channels"""
        return len(self.channels)
    
    def write_setup(self) -> None:
        """Write the setup to the card"""
        self.card.write_setup()
    
    def channels_enable(self, enable_list : list[int] = None, enable_all : bool = False) -> int:
        """
        Enables or disables the channels of all the available cards (see register `SPC_CHENABLE` in the manual)
    
        Parameters
        ----------
        enable_list : list[int] = None
            A list of channels bitmasks to be enable or disable specific channels
        enable_all : bool = False
            Enable all the channels
        
        Returns
        -------
        int
            A list with items that indicate for each card the number of channels that are enabled, or True to enable all channels.
        """

        if enable_all:
            for card in self.cards:
                num_channels = card.num_channels()
                card.set_i(SPC_CHENABLE, (1 << num_channels) - 1)
        elif enable_list is not None:
            for enable, card in zip(enable_list, self.cards):
                card.set_i(SPC_CHENABLE, enable)
        self.channels = []
        self.num_channels = []
        num_channels = 0
        for card in self.cards:
            num_channels = card.get_i(SPC_CHCOUNT)
            self.num_channels.append(num_channels)
            for i in range(num_channels):
                self.channels.append(Channel(i, card))
        return sum(self.num_channels)
        
    # def __getattribute__(self, name):
    #     # print("Calling __getattr__: {}".format(name))
    #     if hasattr(Channel, name):
    #         def wrapper(*args, **kw):
    #             for channel in self.channels:
    #                 getattr(channel, name)(*args, **kw)
    #         return wrapper
    #     else:
    #         return object.__getattribute__(self, name)

    def enable(self, enable : bool) -> None:
        """
        Enables or disables the analog front-end of all channels of the card (see register `SPC_ENABLEOUT` in the manual)
    
        Parameters
        ----------
        enable : bool
            Turn-on (True) or off (False) the spezific channel
        """

        for channel in self.channels:
            channel.enable(enable)
    enable_out = enable
    
    def path(self, value : int) -> None:
        """
        Sets the input path of the analog front-end of all channels of the card (see register `SPC_PATH` in the manual)
    
        Parameters
        ----------
        value : int
            The input path of the specific channel
        """

        for channel in self.channels:
            channel.path(value)
        
    def amp(self, value : int) -> None:
        """
        Sets the output/input range (amplitude) of the analog front-end of all channels of the card in mV (see register `SPC_AMP` in the manual)
    
        Parameters
        ----------
        value : int
            The output range (amplitude) of all channels in millivolts
        """

        for channel in self.channels:
            channel.amp(value)
    
    def offset(self, value : int) -> None:
        """
        Sets the offset of the analog front-end of all channels of the card in mV (see register `SPC_OFFSET` in the manual)
    
        Parameters
        ----------
        value : int
            The offset of all channels in millivolts
        """

        for channel in self.channels:
            channel.offset(value)

    def termination(self, value : int) -> None:
        """
        Sets the termination of the analog front-end of all channels of the card (see register `SPC_50OHM` in the manual)
    
        Parameters
        ----------
        value : int
            The termination of all channels
        """

        for channel in self.channels:
            channel.termination(value)
    
    def coupling(self, value : int) -> None:
        """
        Sets the coupling of the analog front-end of all channels of the card (see register `SPC_ACDC` in the manual)
    
        Parameters
        ----------
        value : int
            The coupling of all channels
        """

        for channel in self.channels:
            channel.coupling(value)
    
    def coupling_offset_compensation(self, value : int) -> None:
        """
        Sets the coupling offset compensation of the analog front-end of all channels of the card (see register `SPC_ACDC_OFFS_COMPENSATION` in the manual)
    
        Parameters
        ----------
        value : int
            The coupling offset compensation of all channels
        """

        for channel in self.channels:
            channel.coupling_offset_compensation(value)
    
    def filter(self, value : int) -> None:
        """
        Sets the filter of the analog front-end of all channels of the card (see register `SPC_FILTER` in the manual)
    
        Parameters
        ----------
        value : int
            The filter of all channels
        """

        for channel in self.channels:
            channel.filter(value)
    
    def stop_level(self, value : int) -> None:
        """
        Usually the used outputs of the analog generation boards are set to zero level after replay. 
        This is in most cases adequate. In some cases it can be necessary to hold the last sample,
        to output the maximum positive level or maximum negative level after replay. The stoplevel will 
        stay on the defined level until the next output has been made. With this function
        you can define the behavior after replay (see register `SPC_CH0_STOPLEVEL` in the manual)
    
        Parameters
        ----------
        value : int
            The wanted stop behaviour:
        """

        for channel in self.channels:
            channel.stop_level(value)

    def custom_stop(self, value : int) -> None:
        """
        Allows to define a 16bit wide custom level per channel for the analog output to enter in pauses. The sample format is 
        exactly the same as during replay, as described in the „sample format“ section.
        When synchronous digital bits are replayed along, the custom level must include these as well and therefore allows to 
        set a custom level for each multi-purpose line separately. (see register `SPC_CH0_CUSTOM_STOP` in the manual)
    
        Parameters
        ----------
        value : int
            The custom stop value
        """
    
        for channel in self.channels:
            channel.custom_stop(value)