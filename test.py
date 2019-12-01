
info_idx = 0

def print_info(stage, msg):
    global info_idx
    if info_idx == 0:
        info_idx = 1
        print("\033[7m\033[32m[tello]\033[7m\033[33m[stage:%s]\033[37;40m %s\033[0m" % (str(stage).ljust(8), msg.ljust(max(30, len(msg) + 2))))
    else:
        info_idx = 0
        print("\033[7m\033[31m[tello]\033[7m\033[37m[stage:%s]\033[37;40m %s\033[0m" % (str(stage).ljust(8), msg.ljust(max(30, len(msg) + 2))))


if __name__ == "__main__":
    print_info(1, "aaa")
    print_info(1, "bbb")
    print_info("1 => 2", "sja ssj")
    print_info(2, "ss da k!")

