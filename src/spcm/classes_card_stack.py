# -*- coding: utf-8 -*-

from contextlib import ExitStack

import spcm_core
from spcm_core.constants import *

from .classes_card import Card
from .classes_sync import Sync
from .classes_error_exception import SpcmException

class CardStack(ExitStack):
    """
    A context manager object for handling multiple Card objects with or without a Sync object
 
    Parameters
    ----------
    cards : list[Card]
        a list of card objects that is managed by the context manager
    sync : Sync
        an object for handling the synchronization of cards
    sync_card : Card
        a card object that is used for synchronization
    sync_id : int
        the index of the sync card in the list of cards
    is_synced : bool
        a boolean that indicates if the cards are synchronized
    """

    cards : list[Card] = []
    sync : Sync = None
    sync_card : Card = None
    sync_id : int = -1
    is_synced : bool = False

    def __init__(self, card_identifiers : list[str] = [], sync_identifier : str = "", find_sync : bool = False) -> None:
        """
        Initialize the CardStack object with a list of card identifiers and a sync identifier

        Parameters
        ----------
        card_identifiers : list[str] = []
            a list of strings that represent the VISA strings of the cards
        sync_identifier : str = ""
            a string that represents the VISA string of the sync card
        find_sync : bool = False
            a boolean that indicates if the sync card should be found automatically
        """

        super().__init__()
        # Handle card objects
        self.cards = [self.enter_context(Card(identifier)) for identifier in card_identifiers]
        if find_sync:
            for id, card in enumerate(self.cards):
                if card.starhub_card():
                    self.sync_card = card
                    self.sync_id = id
                    self.is_synced = True
                    break
        if sync_identifier and (not find_sync or self.is_synced):
            self.sync = self.enter_context(Sync(sync_identifier))
            self.is_synced = bool(self.sync)
    
    def __bool__(self) -> bool:
        """Checks if all defined cards are connected"""
        connected = True
        for card in self.cards:
            connected &= bool(card)
        return connected
    
    def synched(self):
        """Checks if the sync card is connected
        """
        return bool(self.is_synced)
    
    def start(self, *args) -> None:
        """
        Start all cards

        Parameters
        ----------
        args : list
            a list of arguments that will be passed to the start method of the cards
        """
        
        if self.sync:
            self.sync.start(*args)
        else:
            for card in self.cards:
                card.start(*args)
    
    def stop(self, *args) -> None:
        """
        Stop all cards

        Parameters
        ----------
        args : list
            a list of arguments that will be passed to the stop method of the cards
        """
        
        if self.sync:
            self.sync.stop(*args)
        else:
            for card in self.cards:
                card.stop(*args)
    
    def reset(self, *args) -> None:
        """
        Reset all cards

        Parameters
        ----------
        args : list
            a list of arguments that will be passed to the reset method of the cards
        """
        
        if self.sync:
            self.sync.reset(*args)
        else:
            for card in self.cards:
                card.reset(*args)
    
    def force_trigger(self, *args) -> None:
        """
        Force trigger on all cards

        Parameters
        ----------
        args : list
            a list of arguments that will be passed with the force trigger command for the cards
        """
        
        # TODO: the force trigger needs to be correctly implemented in the driver
        if self.sync_card:
            self.sync_card.cmd(M2CMD_CARD_FORCETRIGGER, *args)
        elif self.sync:
            # self.sync.cmd(M2CMD_CARD_FORCETRIGGER, *args)
            self.cards[0].cmd(M2CMD_CARD_FORCETRIGGER, *args)
        else:
            for card in self.cards:
                card.cmd(M2CMD_CARD_FORCETRIGGER, *args)

    def sync_enable(self, enable : int = True) -> int:
        """
        Enable synchronization on all cards

        Parameters
        ----------
        enable : int or bool
            a boolean or integer mask to enable or disable the synchronization of different channels
        
        Returns
        -------
        int
            the mask of the enabled channels
        
        Raises
        ------
        ValueError
            The enable parameter must be a boolean or an integer
        SpcmException
            No sync card avaliable to enable synchronization on the cards
        """
        
        if self.sync:
            return self.sync.enable(enable)
        else:
            raise SpcmException("No sync card avaliable to enable synchronization on the cards")
    
    
    @staticmethod
    def id_to_ip(device_identifier : str) -> str:
        """
        Returns the IP address of the Netbox using the device identifier

        Parameters
        ----------
        device_identifier : str
            The device identifier of the Netbox

        Returns
        -------
        str
            The IP address of the Netbox
        """
        ip = device_identifier
        ip = ip[ip.find('::') + 2:]
        ip = ip[:ip.find ('::')]
        return ip
    
    @staticmethod
    def discover(max_num_remote_cards : int = 50, max_visa_string_len : int = 256, max_idn_string_len : int = 256, timeout_ms : int = 5000) -> dict[list[str]]:
        """
        Do a discovery of the cards connected through a network

        Parameters
        ----------
        max_num_remote_cards : int = 50
            the maximum number of remote cards that can be discovered
        max_visa_string_len : int = 256
            the maximum length of the VISA string
        max_idn_string_len : int = 256
            the maximum length of the IDN string
        timeout_ms : int = 5000
            the timeout in milliseconds for the discovery process

        Returns
        -------
        CardStack
            a stack object with all the discovered cards
        
        Raises
        ------
        SpcmException
            No Spectrum devices found
        """

        visa = (spcm_core.c_char_p * max_num_remote_cards)()
        for i in range(max_num_remote_cards):
            visa[i] = spcm_core.cast(spcm_core.create_string_buffer(max_visa_string_len), spcm_core.c_char_p)
        spcm_core.spcm_dwDiscovery (visa, spcm_core.uint32(max_num_remote_cards), spcm_core.uint32(max_visa_string_len), spcm_core.uint32(timeout_ms))

        # ----- check from which manufacturer the devices are -----
        idn = (spcm_core.c_char_p * max_num_remote_cards)()
        for i in range(max_num_remote_cards):
            idn[i] = spcm_core.cast(spcm_core.create_string_buffer(max_idn_string_len), spcm_core.c_char_p)
        spcm_core.spcm_dwSendIDNRequest (idn, spcm_core.uint32(max_num_remote_cards), spcm_core.uint32(max_idn_string_len))

        # ----- store VISA strings for all discovered cards and open them afterwards -----
        list_spectrum_devices = {}
        for (id, visa) in zip(idn, visa):
            if not id:
                break

            if id.decode('utf-8').startswith("Spectrum GmbH,"):
                ip = __class__.id_to_ip(visa.decode("utf-8"))
                if ip in list_spectrum_devices:
                    list_spectrum_devices[ip].append(visa.decode("utf-8"))
                else:
                    list_spectrum_devices[ip] = [visa.decode("utf-8")]
        
        if not list_spectrum_devices:
            raise SpcmException("No Spectrum devices found")
        
        return list_spectrum_devices