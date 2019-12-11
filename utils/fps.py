import sl4p
import time


class FpsRecoder:

    def __init__(self, name):
        self.name = name
        self.logger = sl4p.Sl4p('fps_%s' % name)
        self.call_list = []
        self.max_recode_count = 120
        self.latest_print_time = None
        self.print_duration = 1
        self.fps = None

    def on_loop(self):
        t = time.time()
        while len(self.call_list) >= self.max_recode_count:
            self.call_list.pop(0)
        self.call_list.append(t)
        count = len(self.call_list)
        if count < 30:
            return
        self.fps = float(count) / (self.call_list[-1] - self.call_list[0])
        if self.latest_print_time is None or time.time() - self.latest_print_time > self.print_duration:
            self.latest_print_time = time.time()
            self.logger.info('fps: %.2f' % self.fps)
