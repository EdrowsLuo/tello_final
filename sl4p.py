LOG_LEVEL_ALL = 0
LOG_LEVEL_INFO = 1
LOG_LEVEL_ERROR = 2


class Sl4p:

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

    def info(self, msg):
        self._print(LOG_LEVEL_INFO, msg)

    def error(self, msg):
        self._print(LOG_LEVEL_ERROR, msg)

    def _print(self, level, msg):
        if self.log_level <= level:
            msg = str(msg)
            print("\033[0;%sm[%s]\033[0;%sm[%s]\033[0;%sm %s\033[0m" %
                  (self.style or "0", self.name, self.LOG_LEVEL_STYLE[level],
                   self.LEVEL_NAME_DIC[level], self.LOG_LEVEL_MSG_STYLE[level], msg))


if __name__ == '__main__':
    log = Sl4p("main", "1;36")
    log.info("test")
    log.info("test")
    log.info("test")
    log.error("error!")
