#1、检查Python编译版本
#2、需要上传的文件
#3、上传哪个服务器


#!/bin/bash

PYTHON_COMPILE="/Users/yzj/anaconda3/bin/python"
PYTHON_SCRIPT="pyc_process.py"

ZIPFILE="ai-schedule_monitor.zip"
ZIPFOLDER="ai-schedule_monitor"
CPFOLDER="ai-schedule_monitor_CP"
DEPLOYFLODER="service_deployment"

CP_FILES_DIRS="./ai-schedule_monitor/monitor
               ./ai-schedule_monitor/service_deployment
               "

ZIP_FILES_DIRS="./ai-schedule_monitor_CP/monitor/
                "

# 注意这个路径是复制目录的路径，千万别搞错了，不然会把你原项目的源代码给编译成pyc的
PYC_COMPILE_PATH="/Users/yzj/lefugang/"${CPFOLDER}"/monitor"

# dv_test
#USER="***"
#IP="***"
#PASSWORD="***"
#SCP_PATH="/***/***"
#PROJECT_FOLDER="***"
#RUN="***"

# kd_test
#USER="***"
#IP="***"
#PASSWORD="***"
#SCP_PATH="/***/***"
#PROJECT_FOLDER="***"
#RUN="***"

# product
#USER="***"
#IP="***"
#PASSWORD="***"
#SCP_PATH="/***/***"
#PROJECT_FOLDER="***"
#RUN="***"


cd ../..
echo 'copying ......'

mkdir ${CPFOLDER}
cp -R ${CP_FILES_DIRS} ${CPFOLDER}

cd ${CPFOLDER}/${DEPLOYFLODER}
${PYTHON_COMPILE} ${PYTHON_SCRIPT} ${PYC_COMPILE_PATH}

cd ../..
echo 'zipping ......'
zip -r ${ZIPFILE} ${ZIP_FILES_DIRS}

_CURRENTPATH=$(pwd)

# 传输项目至服务器
expect -c "
spawn scp -r ${_CURRENTPATH}/${ZIPFILE} ${USER}@${IP}:${SCP_PATH}
expect {
\"*assword\" {set timeout 300; send \"${PASSWORD}\n\";}
\"yes/no\" {send \"yes\r\"; exp_continue;}
}
expect eof"


# 服务器端处理
expect -c "
spawn ssh ${USER}@${IP}
expect {
\"yes/no\" {send \"yes\r\";exp_continue }
\"password:\" {set timeout 60; send \"${PASSWORD}\r\";}
}
expect \"]# \"
send \"cd  ${SCP_PATH} \r\"
send \"unzip ${ZIPFILE} \r\"
send \"cp -rf ./${CPFOLDER}/. ./${PROJECT_FOLDER} \r\"
send \"rm -r ${CPFOLDER} \r\"
send \"rm -r ${ZIPFILE} \r\"
send \"cd ./${PROJECT_FOLDER} \r\"
send \"${RUN} \r\"
send \"exit \r\"

expect eof"

rm -rf ${ZIPFILE}
rm -rf ${CPFOLDER}

echo "finish!!!"
