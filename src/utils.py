def matmul_diag_left(D_diagonal, A):
    # Multiply D * A where D_diagonal is the main diagonal of a diagonal matrix D.
    return (D_diagonal * A.T).T


def matmul_diag_right(A, D_diagonal):
    return A * D_diagonal
