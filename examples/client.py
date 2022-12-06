import torch

import socket
import pickle
import time

import custom_wait


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
        torch.storage.TypedStorage(wrap_storage=storage.untyped(), dtype=tensor_dtype),
        tensor_offset, tensor_size, tensor_stride
    )

    return rebuilded_tensor


if __name__ == '__main__':
    HOST = '127.0.0.1'  
    PORT = 9999       
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST, PORT))

    len_bytestr = client_socket.recv(4)    
    cuda_ipc_info_bytestr = client_socket.recv(int.from_bytes(len_bytestr, byteorder="little"))
    cuda_ipc_info = pickle.loads(cuda_ipc_info_bytestr)

    flag_tensor = rebuild_tensor_from_ipc(cuda_ipc_info)

    client_socket.send("rebuild done.".encode())

    wait_op = custom_wait.WaitCuda()
    wait_op(flag_tensor)
    torch.cuda.synchronize()
    print(f"wait_time : {time.time_ns()}")
    print("wait done!!")

    del flag_tensor
    client_socket.close()
    
