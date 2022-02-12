import moderngl
from pathlib import Path


class Context(object):
    context = moderngl.create_standalone_context(require=430)
    shaders = {}
    for p in Path("gl/shaders").glob("*.glsl"):
        with open(p) as fp:
            shaders[p.stem] = fp.read()

    @staticmethod
    def get():
        return Context.context

    @staticmethod
    def shader(name, consts=None):
        shader = Context.shaders[name]
        if consts is not None:
            for key, value in consts.items():
                shader = shader.replace(f"C_{key}_", str(value))
        return Context.context.compute_shader(shader)

