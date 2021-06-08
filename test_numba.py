from numba import jit
@jit(nopython=True)
def f(x,y):
    return x+y
print(f(1,2))
