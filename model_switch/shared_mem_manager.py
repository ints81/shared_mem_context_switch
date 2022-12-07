import torch

import socket
import pickle
import threading
import selectors
import logging
import sys

from ipc_utils import get_tensor_ipc_info

logging.basicConfig(
    format='%(asctime)s: %(message)s',
    datefmt='%m/%d/%Y %I:%M:%S %p',
)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class SharedMemManager(threading.Thread):
    def __init__(self):
        super().__init__()

        self.selector = selectors.DefaultSelector()
        self.server_ip_addr = '127.0.0.1'
        self.server_port = 9999

        self.client_to_gpu = {}

        self.flags = {}

    def register(self, ip_addr, device_name):
        self.client_to_gpu[ip_addr] = device_name

    def get_flag_ipc_info(self, client_addr):
        device_name = self.client_to_gpu[client_addr]
        if device_name not in self.flags.keys():
            LOG.info(f'create new flag for {device_name}')
            device = torch.device(device_name)
            flag = torch.tensor([0], dtype=torch.int, device=device)
            self.flags[device_name] = flag
        else:
            LOG.info(f'use already created flag...')
        
        flag_ipc_info = get_tensor_ipc_info(self.flags[device_name])

        return flag_ipc_info

    def accept_callback(self, socket):
        client_sock, client_addr = socket.accept()
        
        self.selector.register(client_sock, selectors.EVENT_READ, self.msg_callback) 

    def msg_callback(self, client_socket):
        client_addr = ":".join([str(value) for value in client_socket.getpeername()])
        msg_len = 4096
        msg = client_socket.recv(msg_len).decode('utf-8')
        LOG.info(f'message from client {client_addr} - {msg}')

        command = msg.split('/')[0]
        if command == 'register':
            _, device_name = msg.split('/')
            self.register(client_addr, device_name)

            client_socket.send('done'.encode('utf-8'))
        elif command == 'get_flag':
            send_msg = pickle.dumps(self.get_flag_ipc_info(client_addr))
            client_socket.send(len(send_msg).to_bytes(4, byteorder='little'))
            client_socket.send(send_msg)

            client_socket.send('done'.encode('utf-8'))
        else:
            if msg == '':
                LOG.info('client socket killed') 
            elif msg == 'exit':
                LOG.info('client socket closed') 
            else:
                LOG.info('ERROR: Wrong command')

            device_name = self.client_to_gpu[client_addr]
            del self.client_to_gpu[client_addr]
            if device_name not in self.client_to_gpu.values():
                del self.flags[device_name]

            self.selector.unregister(client_socket)
            client_socket.close()
 
    def run(self):
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((self.server_ip_addr, self.server_port))
        server_socket.listen()

        self.selector.register(server_socket, selectors.EVENT_READ, self.accept_callback)

        while True:
            events = self.selector.select()
            for key, mask in events:
                callback = key.data
                callback(key.fileobj)

        server_socket.close()


if __name__ == "__main__":
    shared_mem_manager = SharedMemManager()

    shared_mem_manager.start()
    shared_mem_manager.join()

