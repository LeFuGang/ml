 # -*- coding: utf-8 -*-
'''
tornado.web的请求处理类, http调用接口处理类

post request sample

{
    "reqType": "command/question", # command是命令/意图识别方式, question是自然语言问答
    "orgId": "<string>",  # 组织标识
    "userId": "<string>", # 用户唯一标识
    "sessionId": "<string>", # 会话标识

    "themeId": "<string>", # 主题/领域标识
    "payload": "<string>", # 用户内容

    # 用户上下文属性, context原则上是一个json对象, 可以定义任意的json结构对象.
    "context": {
        "key1": "value1",
        "key2": "value2"
    }
}

response sample

{
    "rspType": "intention/...", 意图识别结果
    "request": {
        "orgId": "<string>",  # 组织标识
        "userId": "<string>", # 用户唯一标识
        "sessionId": "<string>", # 会话标识
        "payload": "<string>", # 用户内容
    },
    "response": {
        "intent": "<string>", 意图识别结果
        "themeId": "<string>", # 主题/领域标识
        "themeresult": {}, # 领域识别的结果对象, 词槽抽取的结果
        "matchedQuestion": "<string>",
        "answer": "<string>",
        "relatedQuestion": [],
    },
}

'''

import uuid
from tornado.web import RequestHandler
import json
from monitor.meetingschedule import meetingSchedulePredict

from monitor.utils.logger import BusinessLogger
from monitor.rule import *


def _predict(test):
    timeBase = datetime.datetime.now()
    current_Time = datetime.datetime.now()
    result = rule(test, timeBase, current_Time)
    return result


class ScheduledetectHandler(RequestHandler):
    def data_received(self, chunk):
        pass

    def __init__(self, application, request, **kwargs):
        RequestHandler.__init__(self, application, request, **kwargs)

    def get(self):
        RequestHandler.get(self)

    def post(self):
        # python3中post过来的数据被转成bytes类型，需要解码成str
        req_json = json.loads(self.request.body.decode())
        sentence = req_json.get("payload")
        tmp = _predict(sentence)
 
        BusinessLogger.set_global(session_id=req_json.get("sessionId"), req_id=str(uuid.uuid1()))
        BusinessLogger.info(orig_tenant_id=req_json.get("orgId"), question=req_json.get("payload"), result=tmp)
        response = result
        self.set_header('Content-Type', 'application/json;charset=utf-8')
        self.write(response)
