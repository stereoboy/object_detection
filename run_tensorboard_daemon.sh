#
# reference: https://stackoverflow.com/questions/19233529/run-bash-script-as-daemon
setsid tensorboard --logdir ./ssd_logs  >/dev/null 2>&1 < /dev/null &

