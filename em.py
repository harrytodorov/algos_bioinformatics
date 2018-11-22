## Haralambi Todorov (harrytodorov@gmail.com)
## University of Freiburg, Germany
## November 2018

# The Expectation Maximization algorithm
# Input:    Observed data X=(X_1, ..., X_N),
#           number of iterations,
#           initial values for theta_A, theta_B
#           parameter m
# Output:   theta_A, theta_B

def e_step(x, n, theta_a, theta_b, m):
    ti_a = [0.0] * n
    ti_b = [0.0] * n

    for i in range(n):
        a_part = (theta_a ** x[i]) * ((1.0 - theta_a) ** (m - x[i]))
        b_part = (theta_b ** x[i]) * ((1.0 - theta_b) ** (m - x[i]))

        ti_a[i] = a_part / (a_part + b_part)
        ti_b[i] = 1.0 - ti_a[i]
    return [ti_a, ti_b]

def m_step(ti_a, ti_b, n, x):
    numerator_a = sum([ti_a[i] * x[i] for i in range(n)])
    denominator_a = n * sum([ti_a[i] for i in range(n)])
    theta_a = numerator_a / denominator_a

    numerator_b = sum([ti_b[i] * x[i] for i in range(n)])
    denominator_b = n * sum([ti_b[i] for i in range(n)])
    theta_b = numerator_b / denominator_b

    return [theta_a, theta_b]

if __name__ == '__main__':
    X = (4, 9, 8, 3, 7)
    N = len(X)
    theta_A = 0.3
    theta_B = 0.4
    M = 10
    ni = 4

    for i in range(ni):
        TI_A, TI_B = e_step(X, N, theta_A, theta_B, M)
        theta_A, theta_B = m_step(TI_A, TI_B, N, X)
    print(f"theta_a: {theta_A}\ntheta_b: {theta_B}")