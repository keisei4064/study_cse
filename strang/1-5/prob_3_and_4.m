%% 問題 3 & 4: K行列 (Fixed-Fixed) と DST
clear; clc;

N = 5;
h = 1/(N+1);

% 1. K5行列の生成
K = toeplitz([2 -1 zeros(1, N-2)]);

% 2. 固有値分解
[Q, E] = eig(K);
lambda_num = diag(E); % 数値解

% 3. 理論値との比較 (2 - 2cos(k*pi*h))
k = (1:N)';
lambda_theory = 2 - 2*cos(k * pi * h);

fprintf('--- Problem 3: Eigenvalues of K5 ---\n');
disp(table(k, lambda_num, lambda_theory, 'VariableNames', {'Mode_k', 'Numerical', 'Theory'}));
fprintf('--- ---------------------------- ---\n');


fprintf('--- Problem 4: Check DST match ---\n');

% 4. 固有ベクトル(Q) と DST (離散正弦変換) の関係
Q
DST_num = Q * diag([-1, -1, 1, -1, 1])
 
% 理論的な正弦波 (DST基底)
j = (1:N)';
k_row = 1:N;
DST_theory = sin(j * k_row * pi * h) * sqrt(2*h) % 正規化係数:sqrt(2h)

diff_norm = norm(DST_num - DST_theory);
fprintf('Norm difference between Q and Theory: %e\n', diff_norm);

% 5. 転置が逆行列になることの確認
fprintf('--- Problem 5: Check if transpose is the inverse ---\n');
DST_inv = inv(DST_num);
using_inv = DST_num * DST_inv
using_transpose = DST_num * DST_num'