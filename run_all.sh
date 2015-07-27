#!/bin/bash
TOPLEVEL="$PWD"

for learning in advantage qlearning
do
    for stochastic in "deterministic" "stochastic"
    do
        for problem in "gridworld" "pogridworld" "polargridworld"
        do
            for model in kerasnnet lstm gru mut1
            do
                dirname="$stochastic/$model/$problem/$learning"

                parallel -j4 --semaphore "mkdir -p \"plots/$dirname\" && cd \"plots/$dirname\" && echo \"$dirname\" && time python3 \"$TOPLEVEL/main.py\" $stochastic $problem $model $learning softmax oneofn > \"log\" 2>&1"
            done
        done
    done
done

parallel --wait
