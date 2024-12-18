import math

def f(x):
    return math.exp(x) - x*x

def bisection_method(a, b):
    tol = 1e-7
    iterations = 7

    if f(a)*f(b) >= 0:
        raise ValueError("f(a) and f(b) must be negative numbers")
    results = []
    for i in range(1,iterations+1):
        c = (a + b) / 2
        fC = f(c)
        results.append((i,a,b,c,fC))

        if abs(fC) < tol:
            break
        if f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return results
a, b = -2 , 0
results = bisection_method(a, b)
print("Stage | a          | b          | c          | f(c)")
print("--------------------------------------------------")
for stage, a, b, c, f_c in results:
    print(f"{stage:<5} | {a:<10.6f} | {b:<10.6f} | {c:<10.6f} | {f_c:<10.6e}")


