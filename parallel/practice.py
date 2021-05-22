from multiprocessing import Pool

def f(x):
    return x**3

if __name__ == '__main__':
    pool = Pool(processes=4)
    results = [pool.apply(f, args=(x,)) for x in range(1, 7)]
    print(results)

    results2 = pool.map(f, range(1, 7))
    print(results2)
