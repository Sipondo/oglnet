from compute.tensor import Tensor
from gl.context import Context


def arange(length):
    tensor = Tensor(shape=(length,))
    compute_shader = Context.shader("arange", tensor.temp)

    tensor.buffer.bind_to_storage_buffer(0)

    compute_shader.run(group_x=tensor.bs[0], group_y=1, group_z=1)

    return tensor


def from_array(array):
    tensor = Tensor(array.shape, array)
    return tensor
