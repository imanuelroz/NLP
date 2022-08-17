import matplotlib.pyplot as plt
import numpy as np
from time import time as tic
from scipy.interpolate import lagrange


a, b, c = 2, 3, 4


def CountRecursive234(N):
    if N < a:
        return 0
    elif N == a or N == b:
        return 1
    elif N == c:
        return 2
    else:
        return CountRecursive234(N - a) + CountRecursive234(N - b) + CountRecursive234(N - c)


def fac(n):
    return np.math.factorial(n)


def UpdatePatternsGivenNumber4s(n_a, n_b, n_c, fac_a, fac_b, fac_c, fac_abc, old_n_patterns=0):
    # n_a, n_b, fac_a, fac_b, fac_abc all modified locally within this method,
    # but the "original" values remain in the method Count234
    n_patterns = old_n_patterns + fac_abc / fac_a / fac_b / fac_c  # first configuration
    while n_a >= b:  # vary n_a e n_b with n_c fixed
        fac_a /= n_a * (n_a - 1) * (n_a - 2)
        fac_b *= (n_b + 1) * (n_b + 2)
        fac_abc /= n_a + n_b + n_c
        n_a -= b
        n_b += a
        n_patterns += fac_abc / fac_a / fac_b / fac_c
    return int(n_patterns)


def Count234(N):
    # complxity N^2 (the factorial is computed in a smart way, otherwise it would be N^3)
    n_patterns = 0
    if N <= 1:
        return n_patterns
    if N % a == 0:
        n_a = int(N / a)
        n_b = 0
        n_c = 0
    else:
        n_a = int((N - b) / a)
        n_b = 1
        n_c = 0
    # compute the factorials of n_a, n_b e n_c
    fac_a = fac(n_a)
    fac_b = 1
    fac_c = 1
    # compute factorials of n_a + n_b + n_c
    fac_abc = fac_a  # case N even, n_a + n_b + n_c = n_a
    if n_b == 1:  # case N odd, n_a + n_b + n_c = n_a + n_b = n_a + 1
        fac_abc *= n_a + 1
    n_patterns = UpdatePatternsGivenNumber4s(n_a, n_b, n_c, fac_a, fac_b, fac_c, fac_abc, old_n_patterns=n_patterns)  #varying n_a e n_b con n_c = 0
    while n_a >= a:
        # update the factorials
        fac_a /= n_a * (n_a - 1)
        fac_c *= n_c + 1
        fac_abc /= n_a + n_b + n_c
        n_a -= a  # decrease n_a
        n_c += 1  # increase n_c
        n_patterns = UpdatePatternsGivenNumber4s(n_a, n_b, n_c, fac_a, fac_b, fac_c, fac_abc, old_n_patterns=n_patterns)  # vario n_a e n_b con n_c determinato nel while
    return n_patterns


def RunRicorsivo():
    # Could be very slow for big N (N > 40)
    N = range(0, 40)
    times234 = []
    for n in N:
        start = tic()
        p2 = CountRecursive234(n)
        times234.append(tic() - start)
        start = tic()
        print(f"N: {p2} (2, 3 and 4) ")

    plt.figure()
    plt.plot(N, times234)
    plt.title('Computational time')
    plt.xlabel('N')
    plt.ylabel('time')
    plt.legend([ '2, 3 and 4'])
    plt.grid()
    plt.show()

def RunCount234():
    # If N big (N > 40), exclude CountRecursive234 takes too long
    N = range(0, 21)
    times_exp = []
    times = []
    p1, p2 = 0, 0
    n_rep = 1
    for n in N:
        start = tic()
        p1 = CountRecursive234(n)
        times_exp.append(tic() - start)
        start = tic()
        for i in range(n_rep):
            p2 = Count234(n)
        times.append((tic() - start) / n_rep)
        print(f"N: {n} --> {p2} (veloce) vs {p1} (ricorsivo)")
    times = np.array(times)

    plt.figure()
    plt.plot(N, times_exp)
    plt.plot(N, times)
    plt.plot(N, lagrange(N[1::int(len(N) / 2 - 1)], times[1::int(len(N) / 2 - 1)])(N))
    plt.title('Computational time')
    plt.xlabel('N')
    plt.ylabel('time')
    plt.legend(['ricorsivo', 'veloce', 'parabola'])
    plt.grid()
    plt.figure()
    plt.plot(N, times / N / N)
    plt.show()


if __name__ == "__main__":
    #RunRicorsivo()
    RunCount234()
