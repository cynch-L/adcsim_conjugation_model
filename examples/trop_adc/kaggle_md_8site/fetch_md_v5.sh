#!/bin/bash
# 明早一键收 MD v5 结果：Autodl -> 本地 md_v5/ -> 直接跑后处理
#
#   用法:  MD_PW=你的服务器密码 ./fetch_md_v5.sh
#
# 做了三件事：
#   1) 远端把 md_v5/*.npz 打包（并打印还差几个副本没跑完）
#   2) scp 拉回本地 kaggle_md_8site/md_v5/
#   3) 跑 process_md_v4.py，结果写进 results_hinge/
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
PW="${MD_PW:-$(tr -d '\n' < ~/.autodl_ssh_pw 2>/dev/null)}"
[ -n "$PW" ] || { echo "需要服务器密码：MD_PW=xxx 或写入 ~/.autodl_ssh_pw"; exit 1; }
HOST="root@connect.westb.seetacloud.com"
PORT=20359
# Portable interpreter: honour $ADCSIM_PYTHON, else fall back to python3 on PATH.
PY="${ADCSIM_PYTHON:-$(command -v python3 2>/dev/null)}"
if [ -z "$PY" ]; then
  echo "error: no python3 on PATH. Set ADCSIM_PYTHON=/path/to/python" >&2
  exit 1
fi

ssh_do() {   # $1 = 远端命令
  expect -c "
    set timeout 900; set send_slow {1 .001}
    spawn ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -p $PORT $HOST {bash -c {$(printf '%s' "$1" | sed "s/'/'\\\\''/g")}}
    expect { \"*assword:*\" { sleep 0.2; send -s \"$PW\r\"; exp_continue } eof }
  " 2>&1 | grep -v "^spawn \|Permanently added\|assword:"
}

echo "== 1/3 远端进度 =="
ssh_do 'ls /root/autodl-tmp/md_v5/*.npz 2>/dev/null | wc -l | xargs echo "已完成副本数:";
        tail -3 /root/autodl-tmp/md_v5.log;
        tail -3 /root/autodl-tmp/md_v5_gpu2.log;
        cd /root/autodl-tmp && tar czf md_v5.tgz md_v5 2>/dev/null && echo PACKED'

echo "== 2/3 拉回本地 =="
mkdir -p "$HERE/md_v5"
expect -c "
  set timeout 1800; set send_slow {1 .001}
  spawn scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -P $PORT $HOST:/root/autodl-tmp/md_v5.tgz $HERE/md_v5.tgz
  expect { \"*assword:*\" { sleep 0.2; send -s \"$PW\r\"; exp_continue } eof }
" 2>&1 | grep -v "^spawn \|Permanently added\|assword:"
tar xzf "$HERE/md_v5.tgz" -C "$HERE" && rm -f "$HERE/md_v5.tgz"
ls -la "$HERE/md_v5/"

echo "== 3/3 后处理 =="
cd "$HERE" && "$PY" process_md_v4.py
