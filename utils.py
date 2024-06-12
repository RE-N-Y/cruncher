import time
from loguru import logger

def timer(name=None, verbosity=1):
    def _timer(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            if verbosity > 0:
                logger.info(f"{name or func.__name__} took {end - start} seconds")
            return result
        return wrapper

    return _timer