import time


def TimeProf(func, *args):
    start = time.perf_counter()
    res = func(*args)
    end = time.perf_counter()
    print(f'Time elapsed {end-start}s')
    return res
