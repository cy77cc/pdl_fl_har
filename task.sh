#!/bin/bash
sh run.sh cnn avg 0 1 mix3
sh run.sh cnn avg 1 1 mix3+l2
sh run.sh cnn fedprox 0 1 mix3
sh run.sh resnet avg 0 1 mix3
sh run.sh resnet avg 1 1 mix3+l2
sh run.sh resnet fedprox 0 1 mix3
sh run.sh cnn avg 1 1 mix2+l2+dropout
sh run.sh resnet avg 1 1 mix2+l2+dropout
sh run.sh cnn avg 0 0.5 mix3+dp+0.5
sh run.sh cnn avg 0 1 mix3+dp+1
sh run.sh cnn avg 0 5 mix3+dp+5
# exit 0
# sh run.sh cnn avg 0 1 mix2
# sh run.sh cnn avg 1 1 mix2+l2
# sh run.sh cnn fedprox 0 1 mix2
# sh run.sh resnet avg 0 1 mix2
# sh run.sh resnet avg 1 1 mix2+l2
# sh run.sh resnet fedprox 0 1 mix2
# sh run.sh cnn avg 1 1 mix2+l2+dropout
# sh run.sh resnet avg 1 1 mix2+l2+dropout


