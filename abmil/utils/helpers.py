def get_device():
    import torch as t

    if t.cuda.is_available():
        return t.device("cuda:1") if t.cuda.device_count() > 1 else t.device("cuda")
    else:
        return t.device("cpu")
