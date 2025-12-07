%% 問題 7: 巡回行列 C4 と フーリエ行列 F4
clear; clc;

N = 4;

% 1. 巡回行列 C4 の生成
C = toeplitz([2 -1 0 -1])

% 2. 数値解 (eig)
[Q_eig, E_eig] = eig(C);
lambda_eig = diag(E_eig);

fprintf("Q=\n")
disp(Q_eig)

fprintf('Eigenvalues of C4=\n');
disp(lambda_eig');

% 3. フーリエ行列 F4 (理論上の固有ベクトル)
%    F_jk = w^((j-1)(k-1)) / sqrt(N)
F = zeros(N);
w = exp(1i * 2 * pi / N); % 1のN乗根
for j = 1:N
    for k = 1:N
        F(j,k) = w^((j-1)*(k-1));
    end
end
F = F / sqrt(N) % 正規化

% 4. F が C の固有ベクトルであることの確認
lambda_theory = fft(C(1,:)).'  % 理論的な固有値（複素数の場合もあるが実対称なら実数）

% 対角行列化
Lambda_F = diag(lambda_theory);

% C*F = F*Lambda の確認
check_diff = norm(C * F - F * Lambda_F);
fprintf('FがCの固有ベクトル行列であるかの誤差: %e \n', check_diff);


% 5. 実数固有ベクトル(Q_eig) と 複素フーリエ(F) の関係
%    Q_eig は実数 (cos, sin)
%    F は複素数 (exp)
%    重複固有値に対応する Q_eig の列を混ぜ合わせると F になる
fprintf('\n--- 重複固有値の空間での変換 ---\n');
% 固有値2に対応する空間
%   Q_eig の中で固有値が 2 になっている列を探す
indices_2 = find(abs(lambda_eig - 2) < 1e-10);
if length(indices_2) == 2  % 重複が2つ見つかるはず
    v1 = Q_eig(:, indices_2(1));
    v2 = Q_eig(:, indices_2(2));
    
    disp('固有値2に対応する実固有ベクトル(eig):');
    disp([v1, v2]);
    
    % これらを「i (虚数単位)」を使って混ぜる
    %   Eulerの公式: exp(ix) = cos(x) + i*sin(x)
    v_complex = (v1 + 1i * v2) / sqrt(2); 
    
    disp('線形結合で作った複素ベクトル:');
    disp(v_complex);
    
    disp('対応するフーリエ行列 F の列 (理論値):');
    disp(F(:,2)); % ※順序は実装依存なので対応するものを目視確認
else
    disp('固有値の並び順が想定と異なります。');
end