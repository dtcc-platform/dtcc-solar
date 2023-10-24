import os
import time
import numpy as np
import math
from multiprocessing import Pool
from pathos.multiprocessing import ProcessingPool as PPool


def etl(name: str):
    start = 10
    steps = 1000
    xs = np.linspace(start, start + 100, steps)
    d = 0

    for i in range(0, len(xs)):
        for j in range(0, len(xs)):
            d += math.sqrt(math.pow(xs[i] - xs[j], 2))

    return d, name


def etl2(name: str, start: int):
    steps = 1000
    xs = np.linspace(start, start + 100, steps)
    d = 0

    for i in range(0, len(xs)):
        for j in range(0, len(xs)):
            d += start + math.sqrt(math.pow(xs[i] - xs[j], 2))

    return d, name


if __name__ == "__main__":
    os.system("clear")

    names = []
    start = []
    n = 30
    for i in range(0, n):
        names.append(str(i))
        start.append(i * 10)

    t_start = time.perf_counter()

    for name in names:
        d, name = etl(name)
        print(name, d)

    t_mid = time.perf_counter()
    duration = t_mid - t_start
    print(f"Linear duration: {duration}")

    with Pool() as p:
        res = p.imap_unordered(etl, names)
        for d, name in res:
            print(name, d)

    t_mid2 = time.perf_counter()
    duration = t_mid2 - t_mid
    print(f"Pool duration: {duration}")

    with Pool() as p:
        res = p.starmap(etl2, zip(names, start))
        for d, name in res:
            print(name, d)

    t_mid3 = time.perf_counter()
    duration = t_mid3 - t_mid2
    print(f"Pool duration: {duration}")

    with PPool() as p:
        res = p.map(etl2, names, start)
        for d, name in res:
            print(name, d)

    t_end = time.perf_counter()
    duration = t_end - t_mid3
    print(f"Pool duration pathos: {duration}")
