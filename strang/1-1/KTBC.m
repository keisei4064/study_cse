function [K, T, B, C] = KTBC_func(n)
% KTBC  非対称トゥプリッツ行列 K と、そこから派生する 4 種類の行列 T, B, C を生成する。
% n > 1 を前提とする。

    % 基本の対称 Toeplitz 行列 K を作る
    K = toeplitz([2 -1 zeros(1, n-2)]);

    % T: (1,1) 要素を 2 → 1
    T = K;
    T(1,1) = 1;

    % B: (1,1) = 1, (n,n) = 1 に変更
    B = K;
    B(1,1) = 1;
    B(n,n) = 1;

    % C: (1,n) = -1, (n,1) = -1 に変更する
    C = K;
    C(1,n) = -1;
    C(n,1) = -1;
end


n = 4;
[K, T, B, C] = KTBC_func(n)
det(K)
det(T)
det(B)
det(C)
