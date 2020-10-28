from core.logger import logger
import time


def timing(func):
    """
    函数执行时间记录修饰函数
    """
    def wrapper(*args, **kwargs):
        logger.info('function started: {}'.format(func.__name__))
        t0 = time.time()
        result = func(*args, **kwargs)
        t1 = time.time()
        logger.info('function completed: {} ({:.3f}s)'.format(func.__name__, t1-t0))
        return result
    return wrapper
