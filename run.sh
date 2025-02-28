#! /bin/bash
#You should use the next command to run the code.

# print all message:
# nohup ./run.sh >log 2>&1 &

# print the wrong message:
# nohup ./run.sh >/dev/null 2>log &

# print no message:
# nohup ./run.sh >/dev/null 2>&1 &

ARG=4
python3 main_H.py $ARG
