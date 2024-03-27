# -*- coding: utf-8 -*-

from .classes_card import Card 

class CardFunctionality:
    """
    A prototype class for card specific functionality that needs it's own namespace
    """
    card : Card
    function_type = 0

    def __init__(self, card : Card, *args, **kwargs) -> None:
        """
        Takes a Card object that is used by the functionality

        Parameters
        ----------
        card : Card
            a Card object on which the functionality works
        """
        self.card = card
        self.function_type = self.card.function_type()
    
    
    # Check if a card was found
    def __bool__(self) -> bool:
        """
        Check for a connection to the active card
    
        Returns
        -------
        bool
            True for an active connection and false otherwise
        
        """
        
        return bool(self.card)
    
