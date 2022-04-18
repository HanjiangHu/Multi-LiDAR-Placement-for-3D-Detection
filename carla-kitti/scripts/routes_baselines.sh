#!/bin/bash

set -e

cd ../scenario_runner

 while IFS= read -r line; do
 	python scenario_runner.py --route srunner/data/routes_training.xml srunner/data/all_towns_traffic_scenarios.json "$line" --agent srunner/autoagents/npc_agent.py --lidar-params ../hyperparams/lidar_square.toml --hyperparams ../hyperparams/hyper_square.toml --split training --full_round &>> ../logs/routes_hyper_square.out
 done < ../scripts/all_routes.txt

# uncomment the following commands to collect point cloud data with the different LiDAR configurations.

# while IFS= read -r line; do
# 	python scenario_runner.py --route srunner/data/routes_training.xml srunner/data/all_towns_traffic_scenarios.json "$line" --agent srunner/autoagents/npc_agent.py --lidar-params ../hyperparams/lidar_center.toml --hyperparams ../hyperparams/hyper_center.toml --split training --full_round &>> ../logs/routes_hyper_center.out
# done < ../scripts/all_routes.txt
#
# while IFS= read -r line; do
# 	python scenario_runner.py --route srunner/data/routes_training.xml srunner/data/all_towns_traffic_scenarios.json "$line" --agent srunner/autoagents/npc_agent.py --lidar-params ../hyperparams/lidar_line.toml --hyperparams ../hyperparams/hyper_line.toml --split training --full_round &>> ../logs/routes_hyper_line.out
# done < ../scripts/all_routes.txt
#
# while IFS= read -r line; do
# 	python scenario_runner.py --route srunner/data/routes_training.xml srunner/data/all_towns_traffic_scenarios.json "$line" --agent srunner/autoagents/npc_agent.py --lidar-params ../hyperparams/lidar_pyramid.toml --hyperparams ../hyperparams/hyper_pyramid.toml --split training --full_round &>> ../logs/routes_hyper_pyramid.out
# done < ../scripts/all_routes.txt
#
# while IFS= read -r line; do
# 	python scenario_runner.py --route srunner/data/routes_training.xml srunner/data/all_towns_traffic_scenarios.json "$line" --agent srunner/autoagents/npc_agent.py --lidar-params ../hyperparams/lidar_trapezoid.toml --hyperparams ../hyperparams/hyper_trapezoid.toml --split training --full_round &>> ../logs/routes_hyper_trapezoid.out
# done < ../scripts/all_routes.txt

cd ../scripts
