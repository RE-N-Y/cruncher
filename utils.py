import time
from loguru import logger

def timer(verbosity=1):
    def _timer(func):
        def wrapper(self, *args, **kwargs):
            start = time.time()
            result = func(self, *args, **kwargs)
            end = time.time()
            if verbosity > 0:
                name = self.__class__.__name__ + "." + func.__name__
                logger.info(f"{name} took {end - start} seconds")
            return result
        return wrapper

    return _timer