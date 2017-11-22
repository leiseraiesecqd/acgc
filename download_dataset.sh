#！/bin/bash

git reset --hard
rm -rf results
rm -rf logs
rm -rf *outputs
rm -rf data
rm -rf checkpoints
ggpull
mkdir inputs
cd inputs/
wget http://static1.challenger.ai/ai_challenger_stock_train_20171117.zip
wget http://static1.challenger.ai/ai_challenger_stock_test_20171117.zip
unzip ai_challenger_stock_test_20171117.zip
unzip ai_challenger_stock_train_20171117.zip
mv data/20171117/ai_challenger_stock_test_20171117/stock_test_data_20171117.csv ./
mv data/20171117/ai_challenger_stock_train_20171117/stock_train_data_20171117.csv ./
rm *.zip
rm -rf data/
cd ..