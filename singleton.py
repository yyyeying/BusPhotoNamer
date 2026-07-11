import threading


def singleton(cls, *args, **kwargs):
    """线程安全的单例装饰器，使用双重检查锁确保只创建一个实例。"""
    _instance = {}
    _lock = threading.Lock()

    def inner():
        if cls not in _instance:
            with _lock:
                if cls not in _instance:
                    _instance[cls] = cls(*args, **kwargs)
        return _instance[cls]

    return inner
