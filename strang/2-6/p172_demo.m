%% p172_minimal.m
clear; clc;
format long

%% 1. fzero の例：u^2 - 1 = 0
g = @(u) u^2 - 1;

fprintf("u0=10");
u = fzero(g, 10)

fprintf("u0=11")
u = fzero(g, 11)

fprintf("\n---\n")
%% 2. roots の例：多項式 u^2 - 1 = 0
% u^2 - 1 の係数は [1 0 -1]
r = roots([1 0 -1])

% 参考：コンパニオン行列の固有値としても同じ根が出る
C = [0 1;
     1 0];

eig_C = eig(C)


%% 3. Newton 法の例：sin(u) = 0
g = @(u) sin(u);
J = @(u) cos(u);

u = 1;  % u0 = 1 から出発

for i = 1:10
    u = u - J(u) \ g(u);  % Newton 反復
end

[u, g(u)]

%% 色んな初期値を試す
u0_list = [1, 5, 6, 1.5];

for u0 = u0_list
    [u, residual] = newton_sin(u0, 10);

    fprintf("u0 = %.4f\n", u0);
    fprintf("u  = %.16f\n", u);
    fprintf("sin(u) = %.3e\n", residual);
    fprintf("u/pi = %.16f\n", u / pi);
    fprintf("\n");
end


function [u, residual] = newton_sin(u0, iter_num)
g = @(u) sin(u);
J = @(u) cos(u);

u = u0;

for i = 1:iter_num
    u = u - J(u) \ g(u); % ニュートン反復
end

residual = g(u);
end