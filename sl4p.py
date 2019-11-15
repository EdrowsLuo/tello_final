LOG_LEVEL_ALL = 0
LOG_LEVEL_INFO = 1
LOG_LEVEL_ERROR = 2


class Sl4p:
    class MessageToStr:
        def __init__(self, fun):
            self.fun = fun

        def to_str(self, msg, style=False):
            return self.fun(msg, style)

    def __init__(self, name, style=None):
        self.name = name
        self.style = style
        self.log_level = LOG_LEVEL_ALL
        self.LEVEL_NAME_DIC = {
            LOG_LEVEL_ALL: "ALL",
            LOG_LEVEL_INFO: "INFO",
            LOG_LEVEL_ERROR: "ERROR"
        }

        self.LOG_LEVEL_STYLE = {
            LOG_LEVEL_ALL: "0",
            LOG_LEVEL_INFO: "32",
            LOG_LEVEL_ERROR: "1;7;31"
        }

        self.LOG_LEVEL_MSG_STYLE = {
            LOG_LEVEL_ALL: "0",
            LOG_LEVEL_INFO: "0",
            LOG_LEVEL_ERROR: "1;31"
        }

        self.msg_to_str = None

    def info(self, msg):
        return self._print(LOG_LEVEL_INFO, msg)

    def error(self, msg):
        return self._print(LOG_LEVEL_ERROR, msg)

    def _print(self, level, msg):
        if self.log_level <= level:
            if self.msg_to_str is None:
                _msg = str(msg)
            else:
                _msg = self.msg_to_str.to_str(msg, style=True)
            print("\033[0;%sm[%s]\033[0;%sm[%s]\033[0;%sm %s\033[0m" %
                  (self.style or "0", self.name, self.LOG_LEVEL_STYLE[level],
                   self.LEVEL_NAME_DIC[level], self.LOG_LEVEL_MSG_STYLE[level], _msg))
            return "[%s][%s] %s" % \
                   (self.name,
                    self.LEVEL_NAME_DIC[level], self.msg_to_str.to_str(msg, style=False))


if __name__ == '__main__':
    log = Sl4p("main", "1;36")
    log.info("test")
    log.info("test")
    log.info("test")
    log.error("error!")
