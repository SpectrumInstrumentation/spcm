# -*- coding: utf-8 -*-

from spcm_core.constants import *

from .classes_card_stack import CardStack
from .classes_card import Card

class Netbox(CardStack):
    """
    A hardware class that controls a Netbox device
 
    Parameters
    ----------
    netbox_card : Card
        a card object that is the main card in the Netbox
    netbox_number : int
        the index of the netbox card in the list of cards
    is_netbox : bool
        a boolean that indicates if the card is a Netbox
    
    """
    netbox_card : Card = None
    netbox_number : int = -1
    is_netbox : bool = False

    def __init__(self, card_identifiers : list[str] = [], sync_identifier : str = "", find_sync : bool = False, **kwargs) -> None:
        """
        Initialize the Netbox object with a list of card identifiers and a sync identifier

        Parameters
        ----------
        card_identifiers : list[str] = []
            a list of strings that represent the VISA strings of the cards
        sync_identifier : str = ""
            a string that represents the VISA string of the sync card
        find_sync : bool = False
            a boolean that indicates if the sync card should be found automatically
        """

        super().__init__(card_identifiers, sync_identifier, find_sync, **kwargs)

        for id, card in enumerate(self.cards):
            netbox_type = card.get_i(SPC_NETBOX_TYPE)
            if netbox_type != 0:
                self.netbox_card = card
                self.netbox_number = id
                self.is_netbox = True
                break

    def __bool__(self) -> bool:
        """
        Checks if the Netbox is connected and returns true if the connection is alive

        Returns
        -------
        bool
            True if the Netbox is connected
        """

        return self.is_netbox
    
    def __str__(self) -> str:
        """
        Returns the string representation of the Netbox

        Returns
        -------
        str
            The string representation of the Netbox
        """

        netbox_type = self.type()
        netbox_str = "DN{series:x}.{family:x}{speed:x}-{channel:d}".format(**netbox_type)
        return f"Netbox: {netbox_str} at {self.ip()} sn {self.sn():05d}"
    __repr__ = __str__
        
    def type(self) -> dict[int, int, int, int]:
        """
        Returns the type of the Netbox (see register 'SPC_NETBOX_TYPE' in chapter `Netbox` in the manual)

        Returns
        -------
        dict[int, int, int, int]
            A dictionary with the series, family, speed and number of channels of the Netbox
        """

        netbox_type = self.netbox_card.get_i(SPC_NETBOX_TYPE)
        netbox_series = (netbox_type & NETBOX_SERIES_MASK) >> 24
        netbox_family = (netbox_type & NETBOX_FAMILY_MASK) >> 16
        netbox_speed = (netbox_type & NETBOX_SPEED_MASK) >> 8
        netbox_channel = (netbox_type & NETBOX_CHANNEL_MASK)
        return {"series" : netbox_series, "family" : netbox_family, "speed" : netbox_speed, "channel" : netbox_channel}

    def ip(self) -> str:
        """
        Returns the IP address of the Netbox using the device identifier of the netbox_card

        Returns
        -------
        str
            The IP address of the Netbox
        """
        
        return self.id_to_ip(self.netbox_card.device_identifier)

    def sn(self) -> int:
        """
        Returns the serial number of the Netbox (see register 'SPC_NETBOX_SERIALNO' in chapter `Netbox` in the manual)

        Returns
        -------
        int
            The serial number of the Netbox
        """

        return self.netbox_card.get_i(SPC_NETBOX_SERIALNO)
    
    def production_date(self) -> int:
        """
        Returns the production date of the Netbox (see register 'SPC_NETBOX_PRODUCTIONDATE' in chapter `Netbox` in the manual)

        Returns
        -------
        int
            The production date of the Netbox
        """

        return self.netbox_card.get_i(SPC_NETBOX_PRODUCTIONDATE)
    
    def hw_version(self) -> int:
        """
        Returns the hardware version of the Netbox (see register 'SPC_NETBOX_HWVERSION' in chapter `Netbox` in the manual)

        Returns
        -------
        int
            The hardware version of the Netbox
        """

        return self.netbox_card.get_i(SPC_NETBOX_HWVERSION)
    
    def sw_version(self) -> int:
        """
        Returns the software version of the Netbox (see register 'SPC_NETBOX_SWVERSION' in chapter `Netbox` in the manual)

        Returns
        -------
        int
            The software version of the Netbox
        """

        return self.netbox_card.get_i(SPC_NETBOX_SWVERSION)

    def features(self) -> int:
        """
        Returns the features of the Netbox (see register 'SPC_NETBOX_FEATURES' in chapter `Netbox` in the manual)

        Returns
        -------
        int
            The features of the Netbox
        """
        
        return self.netbox_card.get_i(SPC_NETBOX_FEATURES)
    
    def custom(self) -> int:
        """
        Returns the custom code of the Netbox (see register 'SPC_NETBOX_CUSTOM' in chapter `Netbox` in the manual)

        Returns
        -------
        int
            The custom of the Netbox
        """
        return self.netbox_card.get_i(SPC_NETBOX_CUSTOM)

    # def wake_on_lan(self, mac : int):
    #     """
    #     Set the wake on lan for the Netbox (see register 'SPC_NETBOX_WAKEONLAN' in chapter `Netbox` in the manual)

    #     Parameters
    #     ----------
    #     mac : int
    #         The mac addresse of the Netbox to wake on lan
    #     """
    #     self.netbox_card.set_i(SPC_NETBOX_WAKEONLAN, mac)

    def mac_address(self) -> int:
        """
        Returns the mac address of the Netbox (see register 'SPC_NETBOX_MACADDRESS' in chapter `Netbox` in the manual)

        Returns
        -------
        int
            The mac address of the Netbox
        """
        return self.netbox_card.get_i(SPC_NETBOX_MACADDRESS)
    
    def temperature(self) -> int:
        """
        Returns the temperature of the Netbox (see register 'SPC_NETBOX_TEMPERATURE' in chapter `Netbox` in the manual)

        Returns
        -------
        int
            The temperature of the Netbox
        """
        return self.netbox_card.get_i(SPC_NETBOX_TEMPERATURE)
    
    def shutdown(self):
        """
        Shutdown the Netbox (see register 'SPC_NETBOX_SHUTDOWN' in chapter `Netbox` in the manual)
        """
        self.netbox_card.set_i(SPC_NETBOX_SHUTDOWN, 0)

    def restart(self):
        """
        Restart the Netbox (see register 'SPC_NETBOX_RESTART' in chapter `Netbox` in the manual)
        """
        self.netbox_card.set_i(SPC_NETBOX_RESTART, 0)
    
    def fan_speed(self, id : int) -> int:
        """
        Returns the fan speed of the Netbox (see register 'SPC_NETBOX_FANSPEED' in chapter `Netbox` in the manual)

        Returns
        -------
        int
            The fan speed of the Netbox
        """
        return self.netbox_card.get_i(SPC_NETBOX_FANSPEED0 + id)