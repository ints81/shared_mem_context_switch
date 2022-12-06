import torch

import socket
import pickle
import threading
import selectors
from enum import Enum


def get_cuda_ipc_info(tensor):
    cuda_ipc_info = tensor.storage()._share_cuda_()

    tensor_info = (type(tensor), tensor.size(), tensor.stride(),
                   tensor.storage_offset(), tensor.dtype,
                   type(tensor.storage()))

    cuda_ipc_info = tensor_info + cuda_ipc_info

    return cuda_ipc_info


def rebuild_tensor_from_ipc(ipc_info):
    (tensor_cls, tensor_size, tensor_stride,
     tensor_offset, tensor_dtype,
     storage_cls, device, handle,
     storage_size_bytes, storage_offset_bytes,
     ref_counter_handle, ref_counter_offset,
     event_handle, event_sync_required) = ipc_info

    storage = storage_cls._new_shared_cuda(
        device, handle,
        storage_size_bytes, storage_offset_bytes,
        ref_counter_handle, ref_counter_offset,
        event_handle, event_sync_required
    )

    rebuilded_tensor = torch._utils._rebuild_tensor(
        torch.storage._TypedStorage(wrap_storage=storage._untyped(), dtype=tensor_dtype),
        tensor_offset, tensor_size, tensor_stride
    )

    return rebuilded_tensor


class TaskType(Enum):
    TRAIN = 1
    INFER = 2


class DLJob:
    def __init__(self, task_type, ip_addr):
        self.task_type = task_type
        self.ip_addr = ip_addr


class SharedMemManager(threading.Thread):
    def __init__(self):
        super().__init__()

        self.seletor = selectors.DefaultSelector()
        self.server_ip_addr = '127.0.0.1'
        self.server_port = 9999

        self.client_to_gpu = {}

        self.flags = {}

    def get_flag_ipc_info(self, device_name):
        if device_name not in self.flags.keys():
            device = torch.device(device_name)
            flag = torch.tensor([0], dtype=torch.int, device=device)
            self.flags[device_name] = flag
        
        flag_ipc_info = get_cuda_ipc_info(flag)

        return flag_ipc_info

    def accept_callback(self, socket):
        client_sock, client_addr = socket.accept()
        
        self.selector.register(client_sock, selectors.EVENT_READ, self.msg_callback) 

    def msg_callback(self, socket):
        msg_len = 4096
        msg = socket.recv(msg_len).decode('utf-8')

        command, ip_addr, device_name = msg.split('/')
        if command == "register":
            # self.register(ip_addr, device_name)
            pass
        elif command == "get_flag":
            # flag_ipc_info,  = get_flag_ipc_info(ip_addr)
            # 
            pass
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


