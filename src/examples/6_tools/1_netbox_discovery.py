"""
Spectrum Instrumentation GmbH (c)

1_netbox_discovery.py

This example will send a LXI discovery request to the network and check the
answers for Spectrum products.

Example for Netboxes for the the M2p, M4i, M4x and M5i card-families.

See the README file in the parent folder of this examples directory for information about how to use this example.

See the LICENSE file for the conditions under which this software may be used and distributed.
"""

import spcm

device_list = spcm.Netbox.discover()

found_netbox = False

sync_identifier = "sync{}"
sync_id = 0

for devices_identifiers in device_list.values():
    with spcm.Netbox(card_identifiers=devices_identifiers, sync_identifier=sync_identifier.format(sync_id), find_sync=True) as netbox:
        if netbox:
            found_netbox = True
            netbox_type = netbox.type()
            netbox_str = "DN{series:x}.{family:x}{speed:x}-{channel:d}".format(**netbox_type)
            netbox_ip = netbox.ip()
            netbox_sn = netbox.sn()
            print("Netbox found: {} at {} with SN {}".format(netbox_str, netbox_ip, netbox_sn))

            for card in netbox.cards:
                print(f'\t{card.product_name()} SN: {card.sn()}')

            if netbox.is_synced:
                print('\t{} at SN: {}'.format(sync_identifier.format(sync_id), netbox.netbox_card.sn()))
                sync_id += 1
            print("----------------------------------------------")

if not found_netbox:
    print("No Netbox found")
