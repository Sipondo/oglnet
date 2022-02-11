import numpy as np
import moderngl

context = moderngl.create_standalone_context(require=430)


class Tensor():
    def __init__(self, shape):
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

        self.buffer = context.buffer(np.zeros((X, b_X)).astype('f4'))

        self.bs = (b_X,)
        self.temp = consts
    
    @property
    def shape(self):
        return self.s
    
    @property
    def length(self):
        return self.s[0]

    @property
    def __repr__(self):
        return np.frombuffer(self.buffer.read(), dtype=np.float32)
