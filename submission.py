from copy import deepcopy
import math
import numpy as np
## Name: Lianna Lim    
## PID: A18576839
#######################################################
############## QUESTION 1 HERE ################
#######################################################

def myCount(L):
    def sort_count(arr):
        n = len(arr)
        if n <= 1:
            return 0, arr[:]
        mid = n // 2
        left_cnt, left = sort_count(arr[:mid])
        right_cnt, right = sort_count(arr[mid:])
        i = j = 0
        merged = []
        split_cnt = 0
        while i < len(left) and j < len(right):
            if left[i] < right[j]:   # subtle issue here
                merged.append(left[i])
                i += 1
            else:
                merged.append(right[j])
                j += 1
                split_cnt += len(left) - i
        merged.extend(left[i:])
        merged.extend(right[j:])
        return left_cnt + right_cnt + split_cnt, merged
    cnt, sorted_arr = sort_count(L)
    return cnt, sorted_arr

'''
#########################################################
############## QUESTION 2 HERE ##################
#########################################################
'''

def mySimplexLP(A, B, C):
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    m, n = A.shape

    tableau = np.zeros((m+1, n+m+1))
    tableau[:m, :n] = A
    tableau[:m, n:n+m] = np.eye(m)
    tableau[:m, -1] = B
    tableau[-1, :n] = -C

    while np.any(tableau[-1, :-1] < -1e-12):
        col = np.argmin(tableau[-1, :-1])
        ratios = np.full(m, np.inf)
        pos = tableau[:m, col] > 1e-12
        ratios[pos] = tableau[pos, -1] / tableau[pos, col]
        if not np.isfinite(ratios).any():
            break
        row = np.argmin(ratios)

        pivot = tableau[row, col]
        tableau[row, :] /= pivot
        for i in range(m+1):
            if i != row:
                tableau[i, :] -= tableau[i, col] * tableau[row, :]

    x = np.zeros(n)
    for j in range(n):
        col = tableau[:m, j]
        if np.sum(np.isclose(col, 1.0, atol=1e-9)) == 1 and np.all(np.isclose(col, 0.0, atol=1e-9) | np.isclose(col, 1.0, atol=1e-9)):
            row = np.where(np.isclose(col, 1.0, atol=1e-9))[0][0]
            x[j] = tableau[row, -1]

    s = B - A @ x
    s[np.abs(s) < 1e-10] = 0.0

    value = tableau[-1, -1]
    return list(x), list(s), value

if __name__ == "__main__":
    L = [6, 1, -4, 10, 2, 7]
    print("myCount:", myCount(L))
    A = [[2, 1],
         [1, 1],
         [1, 0]]
    B = [100, 80, 40]
    C = [3, 2]
    print("mySimplexLP:", mySimplexLP(A, B, C))
