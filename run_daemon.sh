#
# reference: https://stackoverflow.com/questions/19233529/run-bash-script-as-daemon
setsid python ssd.py --log_name ssd7 >./ssd_log  2>&1 < /dev/null &
