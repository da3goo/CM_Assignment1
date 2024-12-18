import numpy as np
import matplotlib.pyplot as plt

def bisection_algorithm(func, start, end, tolerance=1e-6, max_iterations=100):
    steps = [(start + end) / 2]
    if func(start) * func(end) >= 0:
        print("Error: f(start) and f(end) must have opposite signs.")
        return None, steps

    for _ in range(max_iterations):
        midpoint = (start + end) / 2
        steps.append(midpoint)
        if abs(end - start) < tolerance:
            return midpoint, steps
        if func(midpoint) == 0:
            return midpoint, steps
        if func(start) * func(midpoint) < 0:
            end = midpoint
        else:
            start = midpoint

    return None, steps


def fixed_point_iteration(g, initial_guess, tolerance=1e-6, max_iterations=100):
    steps = [initial_guess]
    current = initial_guess
    for _ in range(max_iterations):
        next_value = g(current)
        steps.append(next_value)
        if abs(next_value - current) < tolerance:
            return next_value, steps
        current = next_value
    return None, steps


def muller_algorithm(func, x0, x1, x2, tolerance=1e-6, max_iterations=100):
    for _ in range(max_iterations):
        f0 = func(x0)
        f1 = func(x1)
        f2 = func(x2)
        h1 = x1 - x0
        h2 = x2 - x1
        delta1 = (f1 - f0) / h1
        delta2 = (f2 - f1) / h2
        a = (delta2 - delta1) / (h2 + h1)
        b = a * h2 + delta2
        c = f2
        discriminant = b**2 - 4 * a * c
        if discriminant >= 0:
            if b > 0:
                x3 = x2 + (-2 * c) / (b + np.sqrt(discriminant))
            else:
                x3 = x2 + (-2 * c) / (b - np.sqrt(discriminant))
        else:
            if b > 0:
                x3 = x2 + (-2 * c) / (b + np.sqrt(discriminant))
            else:
                x3 = x2 + (-2 * c) / (b - np.sqrt(discriminant))

        if abs(x3 - x2) < tolerance:
            return x3
        x0 = x1
        x1 = x2
        x2 = x3

    return None


def newton_raphson_method(func, derivative, initial_guess, tolerance=1e-6, max_iterations=100):
    steps = [initial_guess]
    current = initial_guess
    for _ in range(max_iterations):
        f_value = func(current)
        df_value = derivative(current)
        if abs(df_value) < 1e-12:
            return None, steps
        next_value = current - f_value / df_value
        steps.append(next_value)
        if abs(next_value - current) < tolerance:
            return next_value, steps
        current = next_value
    return None, steps


def secant_method(func, x0, x1, tolerance=1e-6, max_iterations=100):
    steps = [x0, x1]
    for _ in range(max_iterations):
        f_x0 = func(x0)
        f_x1 = func(x1)
        if abs(f_x1 - f_x0) < 1e-12:
            return None, steps
        x2 = x1 - f_x1 * (x1 - x0) / (f_x1 - f_x0)
        steps.append(x2)
        if abs(x2 - x1) < tolerance:
            return x2, steps
        x0 = x1
        x1 = x2
    return None, steps


def plot_function_and_iterations(func, iterations_dict, x_start, x_end, title="Root-Finding Methods"):
    """Plots the function and the iterations of different methods."""
    x_values = np.linspace(x_start, x_end, 400)
    y_values = func(x_values)

    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='f(x)', color='blue', lw=2)

    markers = ['o', 'x', '+', '*']
    index = 0

    for method_name, iterations in iterations_dict.items():
        if iterations:
            plt.plot(iterations, [func(xi) for xi in iterations], marker=markers[index % 4], linestyle='--', label=method_name)
            index += 1

    plt.axhline(0, color='black', lw=0.5)
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


def calculate_absolute_error(current, previous):
    """Calculates the absolute error."""
    return abs(current - previous)


def calculate_relative_error(current, previous):
    """Calculates the relative error."""
    if abs(current) < 1e-12:
        return float('inf')
    return abs((current - previous) / current)


def example_function(x):
    return x - np.cos(x)


def example_derivative(x):
    return 1 + np.sin(x)


def g_function(x):
    return 0.5 * x + 0.5 * np.cos(x)


a = 1
b = -2

root_bisection, iterations_bisection = bisection_algorithm(example_function, a, b)
if root_bisection is not None:
    print("Bisection method: Root:", root_bisection)
    print("Function value at the root:", example_function(root_bisection))
else:
    print("Bisection method: Did not converge or invalid interval")

print("========================")

root_secant, iterations_secant = secant_method(example_function, a, b)
if root_secant is not None:
    print("Secant method: Root:", root_secant)
    print("Function value at the root:", example_function(root_secant))
else:
    print("Secant method: Did not converge or invalid interval")

print("========================")

root_newton, iterations_newton = newton_raphson_method(example_function, example_derivative, a)
if root_newton is not None:
    print("Newton-Raphson method: Root:", root_newton)
    print("Function value at the root:", example_function(root_newton))
else:
    print("Newton-Raphson method: Did not converge or invalid interval")

print("========================")

root_iteration, iterations_iteration = fixed_point_iteration(g_function, a)
if root_iteration is not None:
    print("Fixed point iteration method: Root:", root_iteration)
    print("Function value at the root:", example_function(root_iteration))
else:
    print("Fixed point iteration method: Did not converge or invalid interval")

print("========================")

if root_secant is not None:
    print("Secant method: Root:", root_secant)
if root_iteration is not None:
    print("Iteration method: Root:", root_iteration)
if root_newton is not None:
    print("Newton-Raphson method: Root:", root_newton)
if root_bisection is not None:
    print("Bisection method: Root:", root_bisection)

print("========================")

# Absolute error tests
print("Absolute Error tests")
root_secant_abs, iterations_secant_abs = secant_method(example_function, a, b)
root_iteration_abs, iterations_iteration_abs = fixed_point_iteration(g_function, a)
root_newton_abs, iterations_newton_abs = newton_raphson_method(example_function, example_derivative, a)
root_bisection_abs, iterations_bisection_abs = bisection_algorithm(example_function, a, b)

if root_secant_abs is not None:
   print(f"Secant method (abs): Root {root_secant_abs}, iterations = {len(iterations_secant_abs)}")
if root_iteration_abs is not None:
    print(f"Iteration method (abs): Root {root_iteration_abs}, iterations = {len(iterations_iteration_abs)}")
if root_newton_abs is not None:
    print(f"Newton-Raphson (abs): Root {root_newton_abs}, iterations = {len(iterations_newton_abs)}")
if root_bisection_abs is not None:
  print(f"Bisection Method (abs): Root {root_bisection_abs}, iterations = {len(iterations_bisection_abs)}")

print("========================")

# Relative error tests
print("Relative Error tests")
root_secant_rel, iterations_secant_rel = secant_method(example_function, a, b)
root_iteration_rel, iterations_iteration_rel = fixed_point_iteration(g_function, a)
root_newton_rel, iterations_newton_rel = newton_raphson_method(example_function, example_derivative, a)
root_bisection_rel, iterations_bisection_rel = bisection_algorithm(example_function, a, b)

if root_secant_rel is not None:
   print(f"Secant method (rel): Root {root_secant_rel}, iterations = {len(iterations_secant_rel)}")
if root_iteration_rel is not None:
    print(f"Iteration method (rel): Root {root_iteration_rel}, iterations = {len(iterations_iteration_rel)}")
if root_newton_rel is not None:
    print(f"Newton-Raphson (rel): Root {root_newton_rel}, iterations = {len(iterations_newton_rel)}")
if root_bisection_rel is not None:
  print(f"Bisection Method (rel): Root {root_bisection_rel}, iterations = {len(iterations_bisection_rel)}")


iterations_dict = {
    "Secant": iterations_secant,
    "Fixed Point Iteration": iterations_iteration,
    "Newton-Raphson": iterations_newton,
    "Bisection": iterations_bisection
}

plot_function_and_iterations(example_function, iterations_dict, -1, 3, title="Root-Finding Methods for f(x) = x - cos(x)")
