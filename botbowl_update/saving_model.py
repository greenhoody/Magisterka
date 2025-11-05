from torch import jit


def save_model(model, path):
    scripted = jit.script(model.eval())
    scripted.save(path)


def load_model(path):
    return jit.load(path)
