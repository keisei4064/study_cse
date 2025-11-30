K = toeplitz([2 -1 0])

T = K;
T(1, 1) = 1

B = T;
B(end, end) = 1

fprintf("eps=%d " ,eps)
B_plus_eps = B + eps * eye(3)

fprintf("------------------------------\n")

chol_K = chol(K)
chol_T = chol(T)
% chol_B = chol(B)  % エラー発生「正定値じゃない」
chol_B_plus_eps = chol(B_plus_eps)
