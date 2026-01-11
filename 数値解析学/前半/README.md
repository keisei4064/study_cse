# 数値解析学：前半レポート

## 問題2：ホイン法（Heun 法）

### 数値解と厳密解（h=1.0）

<img src="report_prob2_heun_method_solution_h1.png" alt="数値解と厳密解" width="600">

### 絶対誤差の比較（h=1.0 / 0.5）

<img src="report_prob2_heun_method_error_compare.png" alt="絶対誤差の比較" width="600">

## 問題3：境界値問題（重ね合わせ + RK4）

### 数値解

<img src="report_prob3_solution.png" alt="境界値問題の数値解" width="600">

### 感度：最大誤差の分布

<img src="report_prob3_assumption_max_error.png" alt="最大誤差の分布" width="600">

### 感度：係数 C の分布

<img src="report_prob3_assumption_c_grid.png" alt="係数 C の分布" width="600">

## 問題4：べき乗法（Power Method）

### 残差ノルム比の推移

<img src="report_prob4_power_method_residual_ratio.png" alt="残差ノルム比" width="600">

### 固有値・固有ベクトル誤差の推移

![固有値・固有ベクトル誤差](report_prob4_power_method_steps.png)

### 反復の様子（アニメーション）

![べき乗法アニメーション](report_prob4_power_method_anim.gif)

## 問題5：ヤコビ法（Jacobi 法）

### オフ対角成分ノルムの収束

<img src="report_prob5_jacobi_method_offdiag_convergence.png" alt="オフ対角成分ノルムの収束" width="600">

### 固有値の収束（最大ピボット）

<img src="report_prob5_jacobi_method_eigen_convergence.png" alt="固有値の収束（最大ピボット）" width="600">

### 固有値の収束（サイクリック）

<img src="report_prob5_jacobi_method_eigen_convergence_cyclic.png" alt="固有値の収束（サイクリック）" width="600">

### 行列のヒートマップ（最大ピボット）

<img src="report_prob5_jacobi_method_B_heatmap.gif" alt="行列のヒートマップ（最大ピボット）" width="600">

### 行列のヒートマップ（サイクリック）

<img src="report_prob5_jacobi_method_B_heatmap_cyclic.gif" alt="行列のヒートマップ（サイクリック）" width="600">

### B・G・X の並列表示

![B・G・X の並列表示](report_prob5_jacobi_method_B_G_X_grid.png)
