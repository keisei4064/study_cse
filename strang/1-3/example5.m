% 1.3節 例5: LU分解とピボッティングの実験
clear; clc; format compact;

% --- ケース1: 対角成分がゼロの場合 (必須の入れ替え) ---
disp('=== Case A1: a11 = 0 (Pivot cannot be zero) ===');
A1 = [ 0 -1  1;
      -1  2 -1;
       1 -1  0];
   
% LU分解を実行 (P: 置換行列, L: 下三角, U: 上三角)
[L1, U1, P1] = lu(A1);

disp('A1 ='); disp(A1);
disp('P1 (Permutation) ='); disp(P1);
disp('L1 (Lower) ='); disp(L1);
disp('U1 (Upper) ='); disp(U1);
disp('Check (P*A - L*U should be 0):');
disp(norm(P1*A1 - L1*U1)); % 誤差がほぼ0ならOK

disp('--------------------------------------------------');

% --- ケース2: 数値的安定性のための入れ替え (Partial Pivoting) ---
disp('=== Case A2: Looking for largest pivot ===');
A2 = [ 1  0  0;
       2  3  0;
       0  4  5];

% 1行目は「1」ですが、2行目の「2」の方が絶対値が大きいので、
% MATLABは精度を保つためにあえて入れ替えます。
[L2, U2, P2] = lu(A2);

disp('A2 ='); disp(A2);
disp('P2 (Notice the swap) ='); disp(P2);
disp('L2 ='); disp(L2);
disp('U2 ='); disp(U2);

disp('--------------------------------------------------');

% --- ケース3: 正定値でない行列 (ランク落ちの可能性) ---
disp('=== Case A3: Singular Matrix (Rank deficient) ===');
A3 = [ 1  2  3;
       2  3  4;
       3  4  5];

% この行列は特異(singular)です。1行目-2行目+3行目 = 0 になるはず。
[L3, U3, P3] = lu(A3);

disp('A3 ='); disp(A3);
disp('P3 = '); disp(P3);
disp('L3 ='); disp(L3);
disp('U3 (Check the last diagonal element) ='); disp(U3);
disp('Check (P*A - L*U should be 0):');
disp(norm(P3*A3 - L3*U3)); % 誤差がほぼ0ならOK