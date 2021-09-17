def null(*args, **kwargs):
    pass


class FakeLogger:
    __init__ = null
    log_metrics = null
