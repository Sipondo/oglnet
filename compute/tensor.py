import numpy as np
from gl.context import Context


class Tensor:
    def __init__(self, source=None, shape=None):
        if shape is not None:
            assert all(isinstance(v, int) for v in shape)
            self.s = shape
        else:
            # TODO: skip converting to Numpy
            if not isinstance(source, np.ndarray):
                source = np.array(source)
            self.s = source.shape

        arlen = np.prod(self.shape)
        MAX_X = 1024
        X = min(MAX_X, arlen)
        Y = 1
        Z = 1
        consts = {
            "X": X,
            "Y": Y,
            "Z": Z,
        }
        b_X = arlen // MAX_X + 1

        if source is None:
            self.buffer = Context.get().buffer(
                reserve=X * b_X * 4
            )  # p.zeros((X, b_X)).astype("f4"))
        else:
            if isinstance(source, np.ndarray):
                self.buffer = Context.get().buffer(source.astype("f4"))
            else:
                self.buffer = Context.get().buffer(source)

        self.bs = (b_X,)
        self.temp = consts

    def __str__(self):
        return str(self.array)

    def __repr__(self):
        return f"tensor({self.array})"

    def __del__(self):
        self.buffer.release()

    def copy(self):
        return Tensor(self.buffer.read(), self.shape)

    def reshape(self, *shape):
        tensor = self.copy()
        tensor.s = shape
        return tensor

    def __mul__(self, other):
        tensor = self.copy()
        consts = tensor.temp.copy()

        consts["S"] = other
        compute_shader = Context.shader("multiply_scalar", consts)
        tensor.buffer.bind_to_storage_buffer(0)
        compute_shader.run(group_x=tensor.bs[0], group_y=1, group_z=1)

        return tensor

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (1 / other)

    @property
    def shape(self):
        return self.s

    @property
    def length(self):
        return self.s[0]

    @property
    def size(self):
        return np.prod(self.s)

    @property
    def array(self):
        # TODO: prevent computing useless stuff
        return np.frombuffer(self.buffer.read(), dtype=np.float32)[: self.size].reshape(
            self.shape
        )
