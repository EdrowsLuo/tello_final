# coding=utf-8
import time
import threading
from utils import fps
import sl4p

_services = {}  # type: dict[Service]
_start_order = []

_logger = sl4p.Sl4p("service_center")
_logger.enable = True


class Service:
    def __init__(self):
        self.started = False
        self.run_flag = True

    def flag(self):
        return self.run_flag

    def available(self):
        return True

    def call_start(self):
        self.start()
        self.started = True

    def start(self):
        pass

    def on_request_exit(self):
        self.run_flag = False


class ServiceProxy:
    def __init__(self, key):
        self.key = key

    def available(self):
        return get_service(self.key) is not None and get_service(self.key).available()

    def __getattr__(self, item):
        service = get_service(self.key)
        if service is None:
            return None
        return service.__getattribute__(item)


class FpsRecoder(fps.FpsRecoder):

    def __init__(self, name):
        fps.FpsRecoder.__init__(self, name)

    @staticmethod
    def key(name):
        return "fps::%s" % name

    def on_loop(self):
        if get_config(FpsRecoder.key(self.name), fallback=True):
            fps.FpsRecoder.on_loop(self)


def service_proxy(name):
    return ServiceProxy(name)


def service_proxy_by_class(klass):
    return ServiceProxy(klass.name)


def get_service(name):
    if name in _services:
        return _services[name]
    else:
        return None


def get_service_by_class(klass):
    return get_service(klass.name)


def register_service(model):
    _services[model.name] = model
    _start_order.append(model)
    _logger.info("register service:%s"%model.name)


def start_all_service():
    _logger.info("start all services")
    for s in _start_order:
        _logger.info("start %s"%s.name)
        s.call_start()
        _logger.info("%s started"%s.name)


def lock_loop():
    input_exit_thread(daemon=False)


def input_func(info):
    pass


def input_exit_thread(daemon=True):
    def exit_func():
        while True:
            line = input()
            if 'exit' in line:
                call_request_exit()
                return
            else:
                if len(line) > 0:
                    input_func(line)
    t = threading.Thread(target=exit_func)
    t.daemon = daemon
    t.start()


def call_request_exit():
    for s in _services:
        _services[s].on_request_exit()


def get_config(key, fallback=None):
    config = get_service_by_class(ConfigService)  # type: ConfigService
    if config is not None:
        if key in config.config:
            return config[key]
        else:
            return fallback
    else:
        return fallback


def get_preloaded(key):
    preload = get_service_by_class(PreLoadService)  # type: PreLoadService
    if preload is not None:
        return preload.get_loaded(key)
    else:
        return None


def wait_until_proxy_available(proxy):
    while not proxy.available():
        time.sleep(0.01)


def async_wait_until_proxy_available(proxy: ServiceProxy, target=None):
    def _t():
        wait_until_proxy_available(proxy)
        if target is not None:
            target()
    t = threading.Thread(target=_t)
    t.daemon = True
    t.start()


class ConfigService(Service):
    name = 'ConfigService'

    def __init__(self, config=None):
        Service.__init__(self)
        self.config = {
            'enableConfig': True
        }
        if config is not None:
            for key in config:
                self.config[key] = config[key]

    def get_config(self, key):
        return self.config[key] if key in self.config else None

    def __getitem__(self, item):
        return self.get_config(item)


class PreLoadService(Service):
    name = 'preload_service'

    def __init__(self, tasks=None):
        super().__init__()
        if tasks is None:
            tasks = []
        self.tasks = tasks
        self.loaded = {}

    def put_loaded(self, key, obj):
        self.loaded[key] = obj

    def get_loaded(self, key):
        if key in self.loaded:
            return self.loaded[key]
        else:
            return None

    def start(self):
        for t in self.tasks:
            t(self)