import numpy as np
import moderngl

context = moderngl.create_standalone_context(require=430)


class Tensor:
    def __init__(self, shape, source=None):
        assert all(isinstance(v, int) for v in shape)
        self.context = context
        self.s = shape

        MAX_X = 1024
        X = min(MAX_X, shape[0])
        Y = 1
        Z = 1
        consts = {
            "X": X,
            "Y": Y,
            "Z": Z,
        }
        b_X = shape[0] // MAX_X + 1

        if source is None:
            self.buffer = context.buffer(
                reserve=X * b_X * 4
            )  # p.zeros((X, b_X)).astype("f4"))
        else:
            self.buffer = context.buffer(source.astype("f4"))

        self.bs = (b_X,)
        self.temp = consts

    @property
    def shape(self):
        return self.s

    @property
    def length(self):
        return self.s[0]

    @property
    def data_length(self):
        return np.prod(self.s)

    def __str__(self):
        return str(self.array)

    def __repr__(self):
        return f"tensor({self.array})"

    @property
    def array(self):
        # TODO: prevent computing useless stuff
        return np.frombuffer(self.buffer.read(), dtype=np.float32)[: self.data_length]

    def __del__(self):
        self.buffer.release()
