# 数値解析学：後半レポート課題

## ディレクトリ構成

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

## 使い方

- 実験の実行: python 数値解析学/後半/report/experiments/exp_sor_only.py
- drawio からレイアウト生成:
  python 数値解析学/後半/report/problem_gen/drawio_to_layout.py
- layout_b.yaml の再生成:
  /home/njz/study_cse/.venv/bin/python /home/njz/study_cse/数値解析学/後半/report/problem_gen/drawio_to_layout.py /home/njz/study_cse/数値解析学/後半/report/problem_gen/maps/layout_b.drawio --output /home/njz/study_cse/数値解析学/後半/report/problem_gen/maps/layout_b.yaml
