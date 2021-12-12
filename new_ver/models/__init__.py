from .ABMIL import ABMIL
from .DNDF import DNDF


def get_model(name):
    if name == "abmil":
        return ABMIL()
    elif name == "dndf":
        return DNDF()
    else:
        raise Exception(f"model {name} not defined")
