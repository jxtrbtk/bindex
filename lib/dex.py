## from https://github.com/sammchardy/python-binance-chain
## I've broke the object model from it to understand how it works...

import math
import json
import requests
import binascii

from decimal import Decimal
from collections import OrderedDict

from secp256k1 import PrivateKey

BROADCAST_SOURCE = 0 

from . import api
from . import dex_pb2      # NewOrder, CancelOrder, StdTx, StdSignature
from . import segwit_addr  # decode_address


def encode_number(num):
    if type(num) == Decimal:
        return int(num * (Decimal(10) ** 8))
    else:
        return int(num * math.pow(10, 8))

def varint_encode(num):
    """Convert number into varint bytes
    :param num: number to encode
    """
    buf = b''
    while True:
        towrite = num & 0x7f
        num >>= 7
        if num:
            buf += bytes(((towrite | 0x80), ))
        else:
            buf += bytes((towrite, ))
            break
    return buf

def to_amino(pb, amino_type, include_prefix):
    if type(pb) != bytes:
        pb = pb.SerializeToString()

    type_bytes = b""
    if amino_type:
        type_bytes = binascii.unhexlify(amino_type)
        varint_length = varint_encode(len(pb) + len(type_bytes))
    else:
        varint_length = varint_encode(len(pb))

    msg = b""
    if include_prefix:
        msg += varint_length
    msg += type_bytes + pb    

    return msg

def to_amino_pub_key(pb, amino_type):
    type_bytes = binascii.unhexlify(amino_type)
    varint_length = varint_encode(len(pb))
    msg = type_bytes + varint_length + pb

    return msg

def create_tx_hash(msg, signature_amino):
    ##Transaction
    broadcast_source = BROADCAST_SOURCE
    data = ""
    memo = ""
    stdtx = dex_pb2.StdTx()
    stdtx.msgs.extend([msg])
    stdtx.signatures.extend([signature_amino])
    stdtx.data = data.encode()
    stdtx.memo = memo
    stdtx.source = broadcast_source
    stdtx_amino = to_amino(stdtx, b"F0625DEE", True)
    TxHash = binascii.hexlify(stdtx_amino)

    # print("wallet_sequence", wallet_sequence)
    return TxHash    

def generate_signature(msg_to_dict, wallet_pk, meta_data):

    wallet_sequence, account_number, chain_id = meta_data
    wallet_public_key  = wallet_pk.pubkey.serialize(compressed=True)
    pub_key_msg = to_amino_pub_key(wallet_public_key, b"EB5AE987")
    
    ##Signature
    broadcast_source = BROADCAST_SOURCE
    std_sig = dex_pb2.StdSignature()
    std_sig.sequence       = wallet_sequence
    std_sig.account_number = account_number
    std_sig.pub_key        = pub_key_msg

    msg_to_json = json.dumps(OrderedDict([
                ("account_number",str(account_number)),
                ("chain_id",chain_id),
                ("data",None),
                ("memo",""),
                ("msgs",[msg_to_dict]),
                ("sequence",str(wallet_sequence)),
                ("source",str(broadcast_source))
            ]), ensure_ascii=False)
    msg_to_json = msg_to_json.replace(" ", "")
    json_bytes = msg_to_json.encode()
    sig = wallet_pk.ecdsa_sign(json_bytes)
    signature = wallet_pk.ecdsa_serialize_compact(sig)
    signature = signature[-64:]
    std_sig.signature = signature

    signature_amino = to_amino(std_sig, None, False)
    
    return signature_amino

def prepare_order(wallet_address, mode, quantity, price, symbol, meta_data):

    wallet_sequence, account_number, chain_id = meta_data    
    if mode.lower() == "buy": side = 1
    if mode.lower() == "sell": side = 2                  
    time_in_force   = 1 #constant / other not implemented
    order_type      = 2 #limit order
    price           = Decimal(price)
    quantity        = Decimal(quantity)
    
    order_id           = "{}-{}".format(binascii.hexlify(
        segwit_addr.decode_address(wallet_address)).decode().upper(), (wallet_sequence + 1))

    msg_to_dict = OrderedDict([
                ('id', order_id),
                ('ordertype', order_type),
                ('price', encode_number(price)),
                ('quantity', encode_number(quantity)),
                ('sender', wallet_address),
                ('side', side),
                ('symbol', symbol),
                ('timeinforce', time_in_force),
            ])

    pb = dex_pb2.NewOrder()
    pb.sender      = segwit_addr.decode_address(wallet_address)
    pb.id          = order_id
    pb.symbol      = symbol.encode()
    pb.timeinforce = time_in_force
    pb.ordertype   = order_type
    pb.side        = side
    pb.price       = encode_number(price)
    pb.quantity    = encode_number(quantity)
    msg = to_amino(pb, b'CE6DC043', False)
    
    return msg, msg_to_dict

def prepare_cancel_order(wallet_address, order_id, symbol, meta_data):

    wallet_sequence, account_number, chain_id = meta_data    

    msg_to_dict = OrderedDict([
            ('refid', order_id),
            ('sender', wallet_address),
            ('symbol', symbol),
        ])

    pb = dex_pb2.CancelOrder()
    pb.sender = segwit_addr.decode_address(wallet_address)
    pb.refid = order_id
    pb.symbol = symbol.encode()
    msg = to_amino(pb, b"166E681B", False)
    
    return msg, msg_to_dict

def get_meta_data(wallet_address):
    account          = api.get_rj("account/{}".format(wallet_address))
    wallet_sequence  = account["sequence"]
    account_number   = account["account_number"]
    node_info        = api.get_rj("node-info")
    chain_id         = node_info["node_info"]["network"]
    meta_data        = (wallet_sequence, account_number, chain_id)
    return meta_data

def send_order(wallet_address, wallet_private_key, mode, quantity, price, symbol):
    wallet_pk      = PrivateKey(bytes(bytearray.fromhex(wallet_private_key)))
    meta_data = get_meta_data(wallet_address)

    msg, msg_to_dict = prepare_order(wallet_address, mode, quantity, price, symbol, meta_data)
    signature_amino = generate_signature(msg_to_dict, wallet_pk, meta_data)
    TxHash = create_tx_hash(msg, signature_amino)
    
    res = api.broadcast(TxHash)
    
    return res

def cancel_order(wallet_address, wallet_private_key, symbol, order_id):
    wallet_pk          = PrivateKey(bytes(bytearray.fromhex(wallet_private_key)))

    meta_data = get_meta_data(wallet_address)
    msg, msg_to_dict = prepare_cancel_order(wallet_address, order_id, symbol, meta_data)
    signature_amino = generate_signature(msg_to_dict, wallet_pk, meta_data)
    
    TxHash = create_tx_hash(msg, signature_amino)
    
    res = api.broadcast(TxHash)
    
    return res