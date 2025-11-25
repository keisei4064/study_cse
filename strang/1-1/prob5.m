% フォーマットを分数表示にする（これがこの問題のキモです）
format rat

disp('--- n=3 の場合 (K3) ---')
n = 3;
% K行列を作る（対角成分が2, 隣が-1）
e = ones(n,1);
K3 = spdiags([-e 2*e -e], -1:1, n, n);
K3 = full(K3); % スパース行列から通常の行列へ変換

disp('行列 K3:')
disp(K3)

disp('行列式 det(K3):')
d3 = det(K3);
disp(d3)

disp('逆行列 inv(K3):')
invK3 = inv(K3);
disp(invK3)

disp('逆行列 * 行列式 (整数の構造を見る):')
disp(invK3 * d3)

disp('--------------------------')

disp('--- n=4 の場合 (K4) ---')
n = 4;
e = ones(n,1);
K4 = spdiags([-e 2*e -e], -1:1, n, n);
K4 = full(K4);

disp('行列 K4:')
disp(K4)

disp('行列式 det(K4):')
d4 = det(K4);
disp(d4)

disp('逆行列 inv(K4):')
invK4 = inv(K4);
disp(invK4)

disp('逆行列 * 行列式 (整数の構造を見る):')
disp(invK4 * d4)

% 分数表示にする（パターンが見やすくなります）
format rat

% --- 設定 ---
n = 5;

% --- K5 行列の生成 ---
% spdiagsを使う方法（一般的）
e = ones(n,1);
K5 = spdiags([-e 2*e -e], -1:1, n, n);
K5 = full(K5);

% --- 表示 ---
disp(['--- n = ', num2str(n), ' の場合 (K5) ---'])

disp('行列 K5 (2と-1が並ぶ):')
disp(K5)

disp('行列式 det(K5) (予想: 5+1 = 6 になるはず):')
d5 = det(K5);
disp(d5)

disp('逆行列 inv(K5) (分母が6の分数になるはず):')
invK5 = inv(K5);
disp(invK5)

% --- 構造の確認 ---
disp('逆行列の整数部分 (det * inv):')
disp('この数字の並びの「美しさ」を見てください')
M = d5 * invK5;
disp(M)