import torch

import socket
import pickle
import time


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


if __name__ == '__main__':
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', 9999))
    server_socket.listen()

    client_socket, addr = server_socket.accept()

    flag_tensor = torch.tensor([0], dtype=torch.int).cuda()
    cuda_ipc_info = get_cuda_ipc_info(flag_tensor)

    cuda_ipc_info_bytestr = pickle.dumps(cuda_ipc_info)
    len_cuda_ipc_info_bytestr = len(cuda_ipc_info_bytestr)

    client_socket.send(len_cuda_ipc_info_bytestr.to_bytes(4, byteorder="little"))
    client_socket.send(cuda_ipc_info_bytestr)

    msg = client_socket.recv(4096)

    time.sleep(10)
    print(f"start_time : {time.time_ns()}")
    flag_tensor[0] += 1
    print("flag on!!")
    time.sleep(10)

    del flag_tensor
    server_socket.close()
