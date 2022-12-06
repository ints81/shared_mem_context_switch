import torch


def get_tensor_ipc_info(tensor):
    tensor_ipc_info = tensor.storage()._share_cuda_()

    tensor_info = (type(tensor), tensor.size(), tensor.stride(),
                   tensor.storage_offset(), tensor.dtype,
                   type(tensor.storage()))

    tensor_ipc_info = tensor_info + tensor_ipc_info

    return tensor_ipc_info


def rebuild_tensor_from_ipc_info(ipc_info):
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
