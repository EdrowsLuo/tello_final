from control.tello_judge_client import *

if __name__ == '__main__':
    logger = sl4p.Sl4p('__main__')
    tello_center.register_service(JudgeClientService())
    tello_center.register_service(JudgeServerOverHttp())
    tello_center.start_all_service()

    client = tello_center.service_proxy_by_class(JudgeClientService)  # type: JudgeClientService

    client.server.takeoff()
    client.server.seen_fire()
    logger.info(code2name[client.put_chest_info(1, NAME_BABY)])
    logger.info(code2name[client.put_chest_info(3, NAME_CAT)])
    logger.info(code2name[client.put_chest_info(2, NAME_GAS_TANK)])
    logger.info(code2name[client.put_chest_info(4, NAME_FILES)])
    logger.info(code2name[client.put_chest_info(5, NAME_PAINTING)])
