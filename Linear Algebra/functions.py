import numpy as np


def pivot(m, b):
    """
    Swaps rows to ensure diagonal elements contain the largest values. This returns a diagonally dominant matrix for matrices which can be made diagonally dominant and if not, returns the original matrix with warning.
    """

    n = len(b)
    A = m.copy()

    for row_swaps in range(n):  # ensures the function runs enough times to carry out all potential row swaps
        for i in range(n):
            for j in range(n):
                if abs(A[i][i]) < abs(A[i][j]): # condition that the diagonal is largest, if it's not, perform row switches
                    A[[i, i+1]] = A[[i+1, i]] # row switches
                    b[[i, i+1]] = b[[i+1, i]]

    for rows in range(n):
        if abs(A[i][i]) < abs(A[i][j]):  # if the diagonal is still lower than off diagonals, then the matrix isn't diagonally dominant
            print('Warning: Matrix not fully diagonally dominant')

    return A, b



## Gaussian elimination method

def gaussian_elimination(m, b):
    """
    Solves a system of linear equations by gaussian elimination for input matrices
    A and b, returning solution vector x.
    """

    n = len(b)
    x = np.zeros(n)
    A, w = pivot(m.copy(), b.copy())  # pivot

    for i in range(n):  # sets coeff = a_ji/a_ii, subtracts this val from jth row for each k element, reducing matrix
        for j in range(i+1, n):
            if A[j][i] == 0:  # if element is already 0, continue
                continue

            coeff = A[j][i]/A[i][i]  # sets the coefficient term to be subtracted in producing augmented A
            w[j] = (w[j] - coeff * w[i])  # calculates corresponding rhs vals

            for k in range(i, n):
                A[j][k] = A[j][k] - coeff * A[i][k]  # multiplies each row by coeff and subtracts it from each element

    x[n-1] = w[n-1] / A[n-1][n-1]  # solves the last row of the upper triangle matrix for X

    for i in range(n-1, -1, -1):  # moves backwards to -1 in steps of -1
        sum = 0  # starts counter for the sum
        for j in range(i+1, n):
            sum += A[i][j] * x[j]  # computes sum

        x[i] = (w[i] - sum)/A[i][i]  # solves for given x by subtracting unwanted coefficient

    return x



## Jacobi method

def jacobi(m, b, iterations=100, guess=None, convergence=1e-13):
    """
    Systems of linear equations solver using jacobian iteration method for input matrices A and b, initial guess parameter for x and a desired convergence threshold returning solution vector x.
    """

    n = len(b)
    D = np.zeros((n, n))  # empty 0 array to be used in creating diagonal matrix
    A, b = pivot(m.copy(), b.copy())  # not strictly diagonally dominant but performs a partial pivot

    if guess is None:
        x = np.zeros(n)  # if no guess vector entered, creates 1
    else:
        x = guess

    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = A[i][j] # conserves the diagonal elements of A, leaving off diags as 0
            else:
                continue

    LU = A - D  # removes the diagonal elements leaving a lower upper matrix
    j = np.dot(np.linalg.inv(D), LU)  # computes the iteration matrix j
    eigvals, eigvec = np.linalg.eig(j)  # computes eigenvalues of the iteration matrix j
    sr = np.absolute(eigvals).max()  # calculates spectral radius, must be < 1 for convergence

    # If spectral radius < 1, iterates through input number of iterations and for each solution, checks to see if the solution has converged.
    z = [x]
    if sr < 1:  # if SR meets convergence requirements, continues to compute solution vector
        for j in range(n):
            for i in range(1, iterations):
                x = np.dot(np.linalg.inv(D), (b - np.dot(LU, x)))  # calculates solution vector element x
                z.append(np.array(x))
                if np.absolute(z[i][j] - z[i-1][j]) < convergence: # convergence checker with threshhold 1e-13
                    print('Solution vector component {} converges at N={}'.format(j+1, i))
                    break
                else:
                    continue

            if np.absolute(z[i][j] - z[i-1][j]) > convergence: # checks the last iteration for convergence, if not converged yet, low count
                print('Low iteration count')
            else:
                continue
    else:
        print('No solutions')  # no solutions if SR > 1

    return x



# Gauss-Seidel method

def gauss_seidel(m, b, iterations=100, x=None, convergence=1e-13):
    """
    Gauss-Seidel iteration method for input matrices A and b. Returns solution vector x
    & iterations required for convergence, respectively.

    The method sets up the D, L & U matrices first. Then calculates the spectral radius, sr. By finding the largest
    absolute eigenvalue of the Jacobi Iteration Matrix.

    With sr found, this function iterates over each element of the solution vector (if sr < 1) using
    the equation given in the lecture slides. And it does this for a specified amount of iterations,
    defined in the function call.
    """

    # Sets up initial 0 matrices for D, L & U. Also sets up n.
    n = len(b)
    D = np.zeros((n, n))
    L, U = D.copy(), D.copy()
    A, b = pivot(m.copy(), b.copy()) # not strictly diagonally dominant but performs a partial pivot

    if x is None:
        x = np.zeros(n)

    # Find diagonal, lower and upper matrices D, L, U respectively
    for i in range(n):
        for j in range(n):
            if j == i:
                D[i, j] = A[i, j]
            elif j-i > 0:
                L[i, j] = A[i, j]
            else:
                U[i, j] = A[i, j]

    # Finds spectral radius, using eigenvalues of Jacobi iteration matrix
    j = np.dot(np.linalg.inv(D), (L + U))
    eigvals, eigvec = np.linalg.eig(j)
    sr = np.absolute(eigvals).max()

    # If spectral radius < 1, iterates through input number of iterations and for
    # each solution, checks to see if the solution has converged. If the convergence
    # threshhold of 1e-13 has been met, the loop ends.
    z = [x]
    convergence_counter = 0
    updated_counter = 0
    if sr < 1:
        for j in range(n):
            for i in range(1, iterations):
                x = np.dot(np.linalg.inv(D + L), (b - np.dot(U, x)))
                z.append(np.array(x))
                if np.absolute(z[i][j] - z[i-1][j]) < convergence:
                    print('Solution vector component {} converges at N={}'.format(j+1, i))
                    updated_counter = i
                    break
                else:
                    continue

            # Updates the counter that tracks largest amount of iterations, when the function ends it gives
            # iterations required for convergence
            if updated_counter > convergence_counter:
                convergence_counter = updated_counter

            if np.absolute(z[i][j] - z[i-1][j]) > convergence: # checks the very last iteration for convergence to check for non-convergence due to low iteration count
                print('Low iteration count')
            else:
                continue
    else:
        print('No solutions')

    return x, convergence_counter



## Successive Over Relaxation method

def SOR(A,b, iterations=10000, x=None, convergence=1e-13):
    """
    Successive Over Relaxation iteration method for input matrices A and b. Returns solution vector x,
    optimal relaxation parameter & iterations required for convergence, respectively.

    The method sets up the D, L & U matrices first. Then calculates the optimal relaxation parameter
    by finding the spectral radius of the Jacobi iteration matrix. i.e. the largest absolute eigenvalue
    of the Jacobi iteration matrix. Then uses the equation given in the lecture slides to calculate
    the optimal relaxation parameter, w.

    With w found this function iterates over each x value if 1 < w < 2, there should be n of them. And it does this
    for a specified amount of iterations, defined in the function call.
    """

    # Sets up initial 0 matrices for D, L & U. Also sets up n.
    n = len(b)
    if x is None:
        x = np.zeros(len(b))

    D = np.zeros((n, n))
    L = D.copy()
    U = D.copy()

    # Find diagonal and lower and upper matrices D, L, U respectively
    for i in range(n):
        for j in range(n):
            if j == i:
                D[i, j] = A[i, j]
            elif j-i > 0:
                U[i, j] = A[i, j]
            else:
                L[i, j] = A[i, j]

    # Relaxation parameter W calculation

    J = np.dot(np.linalg.inv(D), (L + U))
    eigvals, v = np.linalg.eig(J)
    eigvals_abs = np.absolute(eigvals)
    sr = np.max(eigvals_abs)
    print('Spectral Radius: ', sr)
    w = 2 / (1 + (1-sr**2)**0.5)

    # Calculates terms 1, 2 and 3 as required for the final equation for x(k+1)
    # because the equation is long
    t1 = np.linalg.inv(D + (w * L))
    t2 = (w * b)
    t3 = (w * U + (w - 1) * D)

   # If optimal relaxation parameter: 1 < w < 2; iterates through input number of iterations and for
    # each solution, checks to see if the solution has converged. If the convergence
    # threshhold of 1e-13 has been met, the loop ends.
    if 1 < w < 2:
        z = [x]
        convergence_counter = 0
        for j in range(n):
            for i in range(iterations):
                x = np.dot(t1, (t2 - np.dot(t3, x)))
                z.append(np.array(x))
                if np.absolute(z[i][j] - z[i-1][j]) < convergence:
                    updated_counter = i
                    break
                else:
                    continue

            # Updates the counter that tracks largest amount of iterations, when the function ends it gives
            # iterations required for convergence
            if updated_counter > convergence_counter:
                convergence_counter = updated_counter

            # checks the very last iteration for convergence to check for non-convergence due to low iteration count
            if np.absolute(z[i][j] - z[i-1][j]) > convergence:
                print('Low iteration count')

            else:
                continue

    else:
        print('No Solutions.')

    return x, w, convergence_counter
