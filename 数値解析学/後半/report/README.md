# 数値解析学：後半レポート課題

## ディレクトリ構成

core/
  laplace_path_planning_solver.py    数値計算，経路生成
  path_planning_utils.py             補助ユーティリティ
problem_gen/
  drawio_to_layout.py                drawio から layout.yaml を生成
  occupancy_grid.py                  layout + config の読込と格子化
  layout.drawio                      drawio レイアウト
  layout.yaml                        レイアウト定義（world/障害物/start/goal）
  numerical_config.yaml              数値計算設定（nx, ny など）
viz/
  plot_laplace.py                    可視化ユーティリティ
experiments/
  exp_sor_only.py                    SOR の数値実験スクリプト

## 使い方

- 実験の実行: python 数値解析学/後半/report/experiments/exp_sor_only.py
- drawio からレイアウト生成:
  python 数値解析学/後半/report/problem_gen/drawio_to_layout.py
