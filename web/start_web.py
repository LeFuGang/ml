# -*- coding: utf-8 -*-
'''
启动类
'''

import sys
import os
import time

_ROOTPATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_ROOTPATH.replace('/monitor', ''))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import tornado.httpserver
from config.config import LISTEN_PORT, CPU_MULTIPLER
from monitor.handler.urlsmap import urls
from monitor.utils.logger import TraceLogger

settings = {
    "static_url_prefix": "/static/",
    "static_path": os.path.join(os.path.dirname(__file__), "static"),
    "template_path": os.path.join(os.path.dirname(__file__), "templates"),
    "cookie_secret": "f2dc5c26204e11e88bf48c859066059b",
    "autoescape": None,
    "gzip": True,
    "debug": False,
}


def _writepid(procid):
    pid = os.getpid()
    f = open("proc%s.pid" % procid, 'w')
    f.write(str(pid))
    f.close()


def _killproc(procid):
    pidfile = "proc%s.pid" % procid
    f = open(pidfile, "r")
    pid = int(f.read())
    f.close()
    os.kill(pid, signal.SIGINT)
    os.remove(pidfile)


def main_sig_handle(sig, frame):
    TraceLogger.info("main proc, caught signal:%s." % sig)
    for i in range(proc_num):
        _killproc(i)


def sig_handler(sig, frame):
    TraceLogger.info("caught signal:%s, %s", sig, frame)
    tornado.ioloop.IOLoop.instance().add_callback(shutdown)


def shutdown():
    TraceLogger.info("tornado stoping...")
    http_server.stop()

    io_loop = tornado.ioloop.IOLoop.instance()
    deadline = time.time() + 10

    def stop_loop():
        now = time.time()
        if now < deadline and (io_loop._callbacks or io_loop._timeouts):
            io_loop.add_timeout(now + 1, stop_loop)
        else:
            io_loop.stop()  # 处理完现有的 callback 和 timeout 后，可以跳出 io_loop.start() 里的循环
            TraceLogger.info('Shutdown')

    stop_loop()
    pass


if __name__ == '__main__':
    import signal

    proc_num = tornado.process.cpu_count() * CPU_MULTIPLER
    _mainpid = os.getpid()

    signal.signal(signal.SIGTERM, main_sig_handle)
    signal.signal(signal.SIGINT, main_sig_handle)

    app = tornado.web.Application(handlers=urls, **settings)
    sockets = tornado.netutil.bind_sockets(LISTEN_PORT)
    try:
        child_id = tornado.process.fork_processes(proc_num)
        if child_id != None:
            _writepid(child_id)
        http_server = tornado.httpserver.HTTPServer(app, xheaders=True)
        http_server.add_sockets(sockets)
        TraceLogger.info("proc:%s web start http://0.0.0.0:%s/" % (child_id, LISTEN_PORT))

        signal.signal(signal.SIGTERM, sig_handler)
        signal.signal(signal.SIGINT, sig_handler)

        tornado.ioloop.IOLoop.instance().start()
        TraceLogger.info("proc:%s exit." % child_id)
    except Exception:
        TraceLogger.info("main proc:%s exit." % _mainpid)
