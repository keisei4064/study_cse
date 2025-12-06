import numpy as np
import scipy.linalg
from fractions import Fraction
import sympy

# 1. データの準備
col = np.array([2, -1, 0, 0, 0])
K = scipy.linalg.toeplitz(col)
P, L, U = scipy.linalg.lu(K)
L_inv = np.linalg.inv(L)

# ==========================================
# Python版 format rat
# ==========================================
np.set_printoptions(formatter={"all": lambda x: str(Fraction(x).limit_denominator())})

print("=== L_inv (numpy分数表示) ===")
print("--- numpy分数表示 ---")
print(L_inv)

# NumPy行列をSymPyのMatrixオブジェクトに変換
L_inv_sym = sympy.Matrix(L_inv)

# 分数として表示（有理数化）
# nsimplify=True にすると、0.3333... を勝手に 1/3 にしてくれる
print("--- SymPy: そのまま ---")
sympy.pprint(sympy.nsimplify(L_inv_sym))

# nsimplify を使って、浮動小数点誤差(1e-10以下)を無視して分数化する
# rational=True: 完全に有理数として扱う
L_inv_clean = sympy.nsimplify(L_inv_sym, tolerance=1e-10, rational=True)

print("--- SymPy: nsimplify ---")
sympy.pprint(L_inv_clean)
