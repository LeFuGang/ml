#!/bin/bash

APP_HOME=`cd $(dirname $0)/..;pwd`
LOG_DIR="$APP_HOME"/logs
VIRTUAL_ENVIRONMENT_NAME="AIScheduleMonitor"
THREAD_NAME='startscheduledetect'

#结束服务
PID_FILE="${LOG_DIR}"/run.pid
echo $PID_FILE

if [ ! -f $PID_FILE ]; then
	echo "run.pid not exist！！！"
fi


pid=`cat $PID_FILE`


procinfo=`ps -ef|grep -w $pid|grep -v grep`
count=0
while [ $count -lt 10 ]; do
	if [ -z "$procinfo" ]; then
		echo "kill  $pid successful"
		rm -rf $PID_FILE
		sleep 1
		break
	fi
	kill $pid
	
	sleep 1
	
	procinfo=`ps -ef|grep -w ${THREAD_NAME} | grep -v grep`
	count=`expr $count + 1`
done


#环境初始化
bash python_virtual_environment_initialization.sh

# 激活虚拟环境
source activate ${VIRTUAL_ENVIRONMENT_NAME}

#安装模块
cd $APP_HOME

pip --version

pip install -r requirements.txt


#开启服务
nohup gunicorn -w 4 -b 0.0.0.0:6006 ${THREAD_NAME}:app -p ${PID_FILE} >> ${LOG_DIR}/server.log 2>&1 &
