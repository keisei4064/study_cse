% CSE公式サイトにあるサンプルを整理したもの

%2.9  fiedlercode.m ------------------------------------------------

% クラスタ数: 2
% 各クラスタのノード数: N=10, 総ノード数: 20
% 隣接行列: W
N=10; W=zeros(2*N,2*N);

rand('state',100);               % Rand repeats to give same graph

% 全てのノード対{i,j} (1<=i<j<=2N) を走査
for i=1:2*N-1
    for j=i+1:2*N
        % j-iが偶数→同じクラスタに所属
        % p: エッジを張る確率
        p=0.7 - 0.6*mod(j-i,2);  % j-i が奇数のときp=0.1, 偶数のとき0.7

        % 演習問題2.9 - 7: p=0.5,0.6
        %p=0.6 - 0.1*mod(j-i,2);

        W(i,j)=rand < p;         % 閾値p未満ならエッジを張る
        % rand は 0以上 1未満の一様乱数
    end
end

% 隣接行列 W と次数行列 D の生成
W=W+W'; D=diag(sum(W));

% グラフラプラシアンの生成
G=D-W; 

% 一般化固有値問題: Gv=λDv を解く
%  E: 固有値をもつ対角行列
%  V: 固有ベクトルを縦ベクトルとしてもつ行列 
[V,E]=eig(G,D);

% 固有値をソート
%  a：昇順に並べた固有値
%  b：元の固有値がどの順番に並んだかを表すインデックス
[a,b]=sort(diag(E));

% フィードラーベクトル（2番目に小さい固有値に対応する固有ベクトル）を取得
z=V(:,b(2));  
z

%2.9  fiedlerplotcode.m ----------------------------------------------

% 2つのクラスタをもつグラフを描画（座標は視覚的な見やすさ以外の意味はない）
%  第1クラスタ（奇数): 中心(-1,-1)
%  第2クラスタ（偶数): 中心(1,1)
theta=[1:N]*2*pi/N;
x=zeros(2*N,1);
y=x;
x(1:2:2*N-1)=cos(theta)-1;
y(1:2:2*N-1)=sin(theta)-1;
x(2:2:2*N)=cos(theta)+1;
y(2:2:2*N)=sin(theta)+1;
subplot(2,2,1);
gplot(W,[x,y]);
title('Graph'); % gplot: graph plot

% 元の隣接行列を描画
%  spy: 非ゼロ要素の位置に点を打つ
subplot(2,2,2); 
spy(W);
title('Adjacency matrix W');

% フィードラーベクトル要素の値を見る
subplot(2,2,3);
plot(z(1:2:2*N-1),'ko'); hold on; % 奇数(第1クラスタ)
plot(z(2:2:2*N),'r*'); hold off; % 偶数(第2クラスタ)
z_max = max(abs(z));
ylim(1.1 * [-z_max, z_max]);
title('Fiedler components');

% 隣接行列をフィードラー成分順に並び替え
%   c：フィードラー成分を小さい順に並べた値
%   d：その並べ替え順
%   W(d,d): 隣接行列の行と列を同じ順序 d で並べ替え
[c,d]=sort(z);
subplot(2,2,4);
spy(W(d,d));
title('Reordered Matrix W');
