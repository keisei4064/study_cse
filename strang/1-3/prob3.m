
%% Problem 3: Toeplitz行列の構造と逆行列の秘密
clear; clc; close all;
format rat % 分数表示モードON

% 1
K = toeplitz([2 -1 zeros(1, 3)])

% 2
det_K = det(K)
inv_K = inv(K)

det_K_inv_K = det_K * inv_K

% 3
[L, U] = lu(K)
D = U * inv(L')
LDL = L * D * L'


inv_L = inv(L)



format short % フォーマット元に戻す

% --- スパース性の可視化 ---
figure('Name', 'Sparsity Pattern');
subplot(1,3,1); spy(K); title('K (Sparse: Connection)');
subplot(1,3,2); spy(L); title('L (Causal Flow)');
subplot(1,3,3); spy(inv_K); title('K^{-1} (Dense: Correlation)');