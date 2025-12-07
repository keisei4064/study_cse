%% 問題 6: 行列 T6 の解析
clear; clc;

N = 6;

% 1. 行列 T の生成
%    左上(1,1)が 1 (自由端)、右下(N,N)は 2 (固定端)
T = toeplitz([2 -1 zeros(1, N-2)]);
T(1,1) = 1;
fprintf("T=\n")
disp(T)

% 2. 数値解 (eig)
[Q_num, E_num] = eig(T);
lambda_num = diag(E_num);

% ソート (小さい順)
[lambda_num, ind] = sort(lambda_num);
Q_num = Q_num(:, ind);

fprintf("Q=\n")
disp(Q_num)
fprintf("E=\n")
disp(E_num)

% 3. 理論値との比較
%    問題文の式: 2 - 2cos((k - 1/2) * pi / (N + 1/2))
%    分母が 6.5 (N+0.5) になるのがポイントです
k_vec = (1:N)';
lambda_theory = 2 - 2 * cos((k_vec - 0.5) * pi / (N + 0.5));

fprintf('--- Problem 6: Eigenvalues of T ---\n');
disp(table(k_vec, lambda_num, lambda_theory, 'VariableNames',{'k','Numerical','Theory'}));

% 4. 固有ベクトルの検証
indices = (0 : N-1)' + 0.5; % [0.5; 1.5; ...; 5.5]

% 理論的な行列 (正規化係数 sqrt(3.25) = sqrt((N+0.5)/2) に注意)
Q_theory = cos(indices * indices' * pi / (N + 0.5)) / sqrt((N + 0.5)/2);

% --- 符号合わせ ---
for k = 1:N
    if sign(Q_num(1,k)) ~= sign(Q_theory(1,k))
        Q_num(:,k) = -Q_num(:,k);
    end
end

% 誤差確認
diff_norm = norm(Q_num - Q_theory);
fprintf('固有ベクトルの誤差 (Numerical vs Theory): %e \n', diff_norm);

% 5. Qを使って計算
QtQ = Q_num' * Q_num
QtTQ = Q_num' * T * Q_num

% 6. 可視化 (片持ち梁の振動モード)
figure(6); clf;
plot(1:N, Q_num(:,1), '-o', 'LineWidth', 2, 'DisplayName', 'Mode 1 (Low Freq)'); hold on;
yline(0, 'k--');
title('Eigenvectors of T (Free-Fixed)');
legend; grid on;
% 左端(1)は傾きが緩やか(自由)、右端(6)は0に向かって急激に落ちる(固定)様子を確認