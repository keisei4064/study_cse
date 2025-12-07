%% 問題 5: 行列 B と DCT基底
clear; clc;

N = 6;

h = 1/N;

% 1. 自由端行列 B の生成
B = toeplitz([2 -1 zeros(1, N-2)]);
B(1,1) = 1; B(N,N) = 1;
fprintf("B=\n")
disp(B)

% 2. 固有値分解
%    Q: 固有ベクトル行列, E: 固有値対角行列
[Q, E] = eig(B);
fprintf("Q=\n")
disp(Q)
fprintf("E=\n")
disp(E)

% --- 固有値の並び替え ---
% MATLABのeigは必ずしも大きさ順とは限らないため、昇順にソート
[lambda_sorted, ind] = sort(diag(E));
Q = Q(:, ind); % 固有ベクトルも同じ順序で並べ替え

fprintf('最小固有値 (理論値は0): %.4e \n', lambda_sorted(1));

% 3. DCT行列と比較
%DCT_basis = dctmtx(N)'

% 【修正】第1列 (k=0) だけ sqrt(2) で割って、長さを1にする
DCT_basis = cos([.5 : 5.5]' * [0 : 5] * pi / N) * sqrt(2*h);
DCT_basis(:, 1) = DCT_basis(:, 1) / sqrt(2);

% --- 符号の整列 ---
for k = 1:N
    if Q(1,k) < 0
        Q(:,k) = -Q(:,k);
    end
end
for k = 1:N
    if DCT_basis(1,k) < 0
        DCT_basis(:,k) = -DCT_basis(:,k);
    end
end

% 4. 誤差の確認
difference = norm(Q - DCT_basis);

fprintf('行列Bの固有ベクトルとDCT基底の誤差: %.4e \n', difference);

if difference < 1e-10
    disp('>> 成功: 行列 B の固有ベクトルは DCT基底 と一致しました！');
else
    disp('>> 失敗: 何かがおかしいようです...');
end

% 5. 可視化

% 定数モード
figure(1); clf;
subplot(2,1,1);
plot(0:N-1, Q(:,1), '-o', 'LineWidth', 2, 'MarkerFaceColor', 'b');
title(['Mode 0 (Eigenvalue \approx ' num2str(lambda_sorted(1), '%.2f') '): 定数']);
grid on; ylabel('Amplitude');

% 低周波
subplot(2,1,2);
plot(0:N-1, Q(:,2), '-r^', 'LineWidth', 2, 'MarkerFaceColor', 'r');
title(['Mode 1 (Eigenvalue \approx ' num2str(lambda_sorted(2), '%.2f') '): 半波長のCosine']);
xlabel('n'); grid on; ylabel('Amplitude');