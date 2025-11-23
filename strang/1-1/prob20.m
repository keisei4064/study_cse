A = [2 3; 4 5]
B = [1 2; 2 4]

% 普通の掛け算
disp('普通の AB:')
disp(A * B)

% 列×行 の分解（Outer Product Expansion）
disp('第1項 (Aの1列目 * Bの1行目):')
Term1 = A(:,1) * B(1,:)
% 結果: [2 4; 4 8]

disp('第2項 (Aの2列目 * Bの2行目):')
Term2 = A(:,2) * B(2,:)
% 結果: [6 12; 10 20]

disp('足し合わせ:')
disp(Term1 + Term2)