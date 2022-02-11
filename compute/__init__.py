from compute.tensor import Tensor

with open('gl/arange.glsl', 'r') as fp:
    shader_raw = fp.read()

def source(consts):
    ''' read gl code '''
    content = shader_raw

    # feed constant values
    for key, value in consts.items():
        content = content.replace(f"C_{key}_", str(value))
    return content

def arange(length):
    tensor = Tensor((length,))
    compute_shader = tensor.context.compute_shader(source(tensor.temp))

    tensor.buffer.bind_to_storage_buffer(0)

    compute_shader.run(group_x=tensor.bs[0], group_y=1, group_z=1)

    return tensor

def from_array(array):
    pass
