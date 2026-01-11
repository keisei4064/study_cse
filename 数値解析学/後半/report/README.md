# 数値解析学：後半レポート課題

ポアソン方程式を用いた経路計画アルゴリズムの実装  
  
> 参考論文：  
> C. I. Connolly, J. B. Burns and R. Weiss, "Path planning using Laplace's equation," Proceedings., IEEE International Conference on Robotics and Automation, Cincinnati, OH, USA, 1990, pp. 2102-2106 vol.3, doi: 10.1109/ROBOT.1990.126315. 

## ディレクトリ構成

```tree
core/
  laplace_path_planning_solver.py    数値計算，経路生成
  path_planning_utils.py             補助ユーティリティ
problem_gen/
  drawio_to_layout.py                drawio から layout.yaml を生成
  occupancy_grid.py                  layout + config の読込と格子化
  maps/layout_a.drawio               drawio レイアウト
  maps/layout_a.yaml                 レイアウト定義（world/障害物/start/goal）
  maps/layout_b.drawio               drawio レイアウト
  maps/layout_b.yaml                 レイアウト定義（world/障害物/start/goal）
  numerical_config.yaml              数値計算設定（nx, ny など）
viz/
  plot_laplace.py                    可視化ユーティリティ
experiments/
  exp_sor_only.py                    SOR の数値実験スクリプト
  exp_sor_only_map_b.py              layout_b 用の SOR 数値実験
```

## 成果物プロット

### SOR（layout_a）

![potential_field](experiments/exp_sor_only/potential_field.png)
![gradient_descent_flow](experiments/exp_sor_only/gradient_descent_flow.png)
![potential_field_3d](experiments/exp_sor_only/potential_field_3d.png)
![residual_history](experiments/exp_sor_only/residual_history.png)

### SOR（layout_b）

![potential_field_map_b](experiments/exp_sor_only_map_b/potential_field.png)
![gradient_descent_flow_map_b](experiments/exp_sor_only_map_b/gradient_descent_flow.png)
![potential_field_3d_map_b](experiments/exp_sor_only_map_b/potential_field_3d.png)
![potential_field_3d_log_map_b](experiments/exp_sor_only_map_b/potential_field_3d_log.gif)
![residual_history_map_b](experiments/exp_sor_only_map_b/residual_history.png)

### 複数スタート点

![multi_start_paths](experiments/exp_multi_start_points/multi_start_paths.png)

### 手法比較（Jacobi / Gauss-Seidel / SOR）

<img src="experiments/exp_method_compare/residual_histories.png" alt="method_compare_residuals" height="300">
<img src="experiments/exp_method_compare/iterations_by_method.png" alt="method_compare_iterations" height="300">
<img src="experiments/exp_method_compare/cpu_time_by_method.png" alt="method_compare_cpu_time" height="300">

![method_compare_potential_log](experiments/exp_method_compare/potential_field_log.png)
![method_compare_velocity_log](experiments/exp_method_compare/gradient_descent_flow_log.png)

### 解像度スイープ

<img src="experiments/exp_resolution_sweep/iterations_vs_grid.png" alt="iterations_vs_grid" width="400">
<img src="experiments/exp_resolution_sweep/cpu_time_vs_grid.png" alt="cpu_time_vs_grid" width="400">

![potential_field_log_grid](experiments/exp_resolution_sweep/potential_field_log.png)
![gradient_descent_flow_log_grid](experiments/exp_resolution_sweep/gradient_descent_flow_log.png)
![potential_field_3d_log_grid](experiments/exp_resolution_sweep/potential_field_3d_log.png)

### 緩和係数比較（SOR）

<img src="experiments/exp_omega_sweep/iterations_vs_omega.png" alt="iterations_vs_omega" width="400">
<img src="experiments/exp_omega_sweep/residual_histories.png" alt="residual_histories_omega" width="400">

## 実行方法

### 実験例

```shell
python 数値解析学/後半/report/experiments/exp_sor_only.py
```

### drawio からレイアウト生成

```shell
python 数値解析学/後半/report/problem_gen/drawio_to_layout.py
```

### layout_b.yaml の再生成

```shell
python 数値解析学/後半/report/problem_gen/drawio_to_layout.py 数値解析学/後半/report/problem_gen/maps/layout_b.drawio --output 数値解析学/後半/report/problem_gen/maps/layout_b.yaml
```
