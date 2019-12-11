# coding=utf-8
from control import tello_center
import requests
import copy
import sl4p
import time

CODE_ERROR_TARGET = 0
CODE_CONTINUE = 1
CODE_TASK_DONE = 2

code2name = {
    CODE_ERROR_TARGET: "CODE_ERROR_TARGET",
    CODE_CONTINUE: "CODE_CONTINUE",
    CODE_TASK_DONE: "CODE_TASK_DONE"
}

NAME_BABY = 'baby'
NAME_PAINTING = 'painting'
NAME_CAT = 'cat'
NAME_FILES = 'files'
NAME_GAS_TANK = 'gas tank'

IDX_CAT = 1
IDX_PAINTING = 2
IDX_FILES = 3
IDX_BABY = 4
IDX_GAS_TANK = 5

name2id = {
    'cat': IDX_CAT,
    'painting': IDX_PAINTING,
    'files': IDX_FILES,
    'baby': IDX_BABY,
    'gas tank': IDX_GAS_TANK
}

id2name = {
    IDX_CAT: NAME_CAT,
    IDX_PAINTING: NAME_PAINTING,
    IDX_FILES: NAME_FILES,
    IDX_BABY: NAME_BABY,
    IDX_GAS_TANK: NAME_GAS_TANK
}


class JudgeServerInterface(tello_center.Service):
    name = 'judge_server'

    def __init__(self):
        tello_center.Service.__init__(self)

    def takeoff(self):
        """
        客户端请求起飞
        @return: 返回时表示允许起飞
        """
        raise NotImplemented()

    def seen_fire(self):
        """
        客服端表示通过着火点
        @return: 无返回信息
        """
        raise NotImplemented()

    def send_target_chest(self, target_idx, chest):
        """
        客户端表示找到了对应target
        @param target_idx:
        @param chest:
        @return: CODE_ERROR_TARGET = 0, CODE_CONTINUE = 1
        """
        raise NotImplemented()

    def send_task_done(self):
        """
        客户端表示完成所有任务
        @return: 无
        """
        raise NotImplemented()

    def get_targets(self):
        """
        在takeoff后客户端读取targets信息
        @return: (target1, target2, target3)
        """
        raise NotImplemented()


class JudgeServerOverHttp(JudgeServerInterface):

    def __init__(self):
        JudgeServerInterface.__init__(self)
        self.logger = sl4p.Sl4p('judge_http')
        self.base = 'http://127.0.0.1:5000'

    def takeoff(self):
        self.logger.info('takeoff')
        requests.get(self.base + '/takeoff')
        start_time = time.time()
        while True:
            can_take_off = int(requests.get(self.base + '/can/takeoff').content)
            if can_take_off == 1:
                return
            if time.time() - start_time > 20:
                raise BaseException('Timeout!')

    def seen_fire(self):
        self.logger.info('seen_fire')
        requests.get(self.base + '/seen/fire')

    def send_target_chest(self, target_idx, chest):
        return int(requests.get(self.base + '/send/target/chest?id=%d&chest=%d'%(target_idx, chest)).content)

    def send_task_done(self):
        requests.get(self.base + '/task/done')

    def get_targets(self):
        ts = requests.get(self.base + '/get/targets').content
        ts = ts.split(' ')
        rqs = []
        for s in ts:
            rqs.append(int(s))
        return rqs


class JudgeServerLocal(JudgeServerInterface):

    def __init__(self):
        JudgeServerInterface.__init__(self)
        self.logger = sl4p.Sl4p('judge_local')
        self.targets = [IDX_BABY, IDX_CAT, IDX_FILES]
        self.results = [1, 3, 4]
        self.next_receive_idx = 1

    def takeoff(self):
        self.logger.info('takeoff')
        return

    def seen_fire(self):
        self.logger.info('seen_fire')
        return

    def send_target_chest(self, target_idx, chest):
        if target_idx != self.next_receive_idx:
            self.logger.info('%d error order %d'%(self.next_receive_idx, target_idx))
            return CODE_ERROR_TARGET
        if self.results[target_idx - 1] != chest:
            self.logger.info('%d error chest %s %d (correct %d)'
                             %(target_idx, id2name[self.targets[target_idx - 1]], chest, self.results[target_idx]))
            return CODE_ERROR_TARGET
        else:
            self.next_receive_idx += 1
            return CODE_CONTINUE

    def get_targets(self):
        self.logger.info("%s %s %s"%(id2name[self.targets[0]], id2name[self.targets[1]], id2name[self.targets[2]]))
        return copy.copy(self.targets)


class JudgeClientService(tello_center.Service):
    name = 'judge_client'

    class ChestInfo:
        def __init__(self, idx, obj_name=None):
            self.idx = idx
            self.obj_name = obj_name

        def __str__(self):
            return "(%d, %s)"%(self.idx, self.obj_name)

    def __init__(self):
        tello_center.Service.__init__(self)
        self.server = tello_center.service_proxy_by_class(JudgeServerInterface)  # type: JudgeServerInterface
        self.targets = None
        self.fail = False
        self.chest_info = {
            1: JudgeClientService.ChestInfo(1),
            2: JudgeClientService.ChestInfo(2),
            3: JudgeClientService.ChestInfo(3),
            4: JudgeClientService.ChestInfo(4),
            5: JudgeClientService.ChestInfo(5)
        }

    def update_chest_info(self):
        if self.fail:
            return CODE_ERROR_TARGET
        if len(self.targets) == 0:
            return CODE_TASK_DONE
        target_idx = 4 - len(self.targets)
        target_object_idx = self.targets[0]
        for c in self.chest_info:
            chest = self.chest_info[c]
            if chest.obj_name is None:
                continue
            object_idx = name2id[chest.obj_name]
            if target_object_idx == object_idx:
                r = self.server.send_target_chest(target_idx, chest.idx)
                if r == CODE_ERROR_TARGET:
                    self.fail = True
                    return CODE_ERROR_TARGET
                else:
                    self.targets.pop(0)
                    return self.update_chest_info()
        return CODE_CONTINUE

    def put_chest_info(self, chest_idx, obj_name):
        """
        帮助处理target顺序
        @param chest_idx:
        @param obj_name:
        @return: CODE_ERROR_TARGET = 0, CODE_CONTINUE = 1, CODE_TASK_DONE = 2
        """
        if self.targets is None:
            self.targets = self.server.get_targets()

        self.chest_info[chest_idx].obj_name = obj_name
        return self.update_chest_info()


if __name__ == '__main__':
    logger = sl4p.Sl4p('__main__')
    tello_center.register_service(JudgeClientService())
    tello_center.register_service(JudgeServerLocal())
    tello_center.start_all_service()

    client = tello_center.service_proxy_by_class(JudgeClientService)  # type: JudgeClientService

    client.server.takeoff()
    client.server.seen_fire()
    logger.info(code2name[client.put_chest_info(1, NAME_BABY)])
    logger.info(code2name[client.put_chest_info(3, NAME_CAT)])
    logger.info(code2name[client.put_chest_info(2, NAME_GAS_TANK)])
    logger.info(code2name[client.put_chest_info(4, NAME_FILES)])
    logger.info(code2name[client.put_chest_info(5, NAME_PAINTING)])
