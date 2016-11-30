#!/bin/bash
TOPLEVEL="$PWD"

for learning in advantage qlearning
do
    for stochastic in "deterministic" "stochastic"
    do
        for problem in "gridworld" "pogridworld" "polargridworld"
        do
            for model in gru #lstm #kerasnnet #  lstm gru mut1
            do
                dirname="$stochastic/$model/$problem/$learning"

		mkdir -p plots/$dirname && cd plots/$dirname && echo $dirname && time python "$TOPLEVEL/main.py" $stochastic $problem $model $learning softmax oneofn >> "log" 2>&1 && cd $TOPLEVEL
            done
        done
    done
done

