import torch

import socket
import pickle
import threading
import selectors

from ipc_utils import get_tensor_ipc_info


class SharedMemManager(threading.Thread):
    def __init__(self):
        super().__init__()

        self.seletor = selectors.DefaultSelector()
        self.server_ip_addr = '127.0.0.1'
        self.server_port = 9999

        self.client_to_gpu = {}

        self.flags = {}

    def register(self, ip_addr, device_name):
        self.client_to_gpu[ip_addr] = device_name

    def get_flag_ipc_info(self, ip_addr):
        device_name = self.client_to_gpu[ip_addr]
        if device_name not in self.flags.keys():
            device = torch.device(device_name)
            flag = torch.tensor([0], dtype=torch.int, device=device)
            self.flags[device_name] = flag
        
        flag_ipc_info = get_tensor_ipc_info(flag)

        return flag_ipc_info

    def accept_callback(self, socket):
        client_sock, client_addr = socket.accept()
        
        self.selector.register(client_sock, selectors.EVENT_READ, self.msg_callback) 

    def msg_callback(self, client_socket):
        msg_len = 4096
        msg = client_socket.recv(msg_len).decode('utf-8')

        command = msg.split('/')[0]
        if command == "register":
            _, ip_addr, device_name = msg.split('/')
            self.register(ip_addr, device_name)
        elif command == "get_flag":
            _, ip_addr = msg.split('/')
            send_msg = pickle.dumps(get_flag_ipc_info(ip_addr))
            client_socket.send(len(send_msg).to_byte(4, byteorder='little'))
            client_socket.send(send_msg)
        else:
            print("ERROR: Wrong command")
            socket.close()
 
    def run(self):
        socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        socket.bind((self.server_ip_addr, self.server_port))
        socket.listen()

        self.selector.register(socket, selectors.EVENT_READ, self.accept_callback)

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

