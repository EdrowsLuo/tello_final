from flask import Flask, request
import time
import rospy
import threading
from std_msgs.msg import String, Int16, Bool

GROUP_INDEX = 41

CODE_ERROR_TARGET = '0'
CODE_CONTINUE = '1'
CODE_TASK_DONE = '2'

takeoff_pub, seenfire_pub, tgt1_pub, tgt2_pub, tgt3_pub, done_pub = None, None, None, None, None, None
state_fail = 0
state_received = 0
state_receivedtarget1 = 0
state_receivedtarget2 = 0
state_receivedtarget3 = 0
target_id = [-1, -1, -1]

app = Flask(__name__)


@app.route('/reset')
def reset():
    state_fail = 0
    state_received = 0
    state_receivedtarget1 = 0
    state_receivedtarget2 = 0
    state_receivedtarget3 = 0
    target_id = [-1, -1, -1]
    return ''


@app.route('/takeoff')
def sent_takeoff():
    print('takeoff')
    takeoff_pub.publish(1)
    return ''

@app.route('/can/takeoff')
def can_takeoff():
    return str(state_received)

@app.route('/seen/fire')
def seen_fire():
    seenfire_pub.publish(1)
    return ''

@app.route('/send/target/chest')
def send_target_chest():
    target_idx = int(request.args.get('id'))
    chest = int(request.args.get('chest'))
    if target_id == 1:
        tgt1_pub.publish(chest)
        if state_fail:
            return CODE_ERROR_TARGET
        while not state_receivedtarget1:
            if state_fail:
                return CODE_ERROR_TARGET
            time.sleep(0.01)
        return CODE_CONTINUE
    if target_id == 2:
        tgt2_pub.publish(chest)
        if state_fail:
            return CODE_ERROR_TARGET
        while not state_receivedtarget2:
            if state_fail:
                return CODE_ERROR_TARGET
            time.sleep(0.01)
        return CODE_CONTINUE
    if target_id == 3:
        tgt3_pub.publish(chest)
        if state_fail:
            return CODE_ERROR_TARGET
        while not state_receivedtarget3:
            if state_fail:
                return CODE_ERROR_TARGET
            time.sleep(0.01)
        return CODE_CONTINUE
    return CODE_ERROR_TARGET


@app.route('/get/targets')
def get_targets():
    while not all_target_got():
        time.sleep(0.01)
    return '%d %d %d' % (target_id[0], target_id[1], target_id[2])


@app.route('/task/done')
def task_done():
    done_pub.publish(1)
    return ''

def all_target_got():
    for s in target_id:
        if s == -1:
            return False
    return True

def failure_handle(data):
    global state_fail
    if state_fail == 0:
        state_fail = data.data

def received_handle(data):
    global state_received
    if state_received == 0:
        state_received = data.data
        print ("state_received = {state_received}".format(state_received=state_received))

def receivedtarget1_handle(data):
    global state_receivedtarget1
    if (state_receivedtarget1 == 0):
        state_receivedtarget1 = data.data
        print ("state_receivedtarget1 = {state_receivedtarget1}".format(state_receivedtarget1=state_receivedtarget1))

def receivedtarget2_handle(data):
    global state_receivedtarget2
    if (state_receivedtarget2 == 0):
        state_receivedtarget2 = data.data
        print ("state_receivedtarget2 = {state_receivedtarget2}".format(state_receivedtarget2=state_receivedtarget2))

def receivedtarget3_handle(data):
    global state_receivedtarget3
    if state_receivedtarget3 == 0:
        state_receivedtarget3 = data.data
        print ("state_receivedtarget3 = {state_receivedtarget3}".format(state_receivedtarget3=state_receivedtarget3))

def target1_handle(data):
    global target_id
    target_id[0] = data.data

def target2_handle(data):
    global target_id
    target_id[1] = data.data

def target3_handle(data):
    global target_id
    target_id[2] = data.data


if __name__ == '__main__':
    rospy.init_node('control', anonymous=False)
    groupid = '/group' + str(GROUP_INDEX)

    takeoff_pub = rospy.Publisher(groupid + '/takeoff', Int16, queue_size=3)
    seenfire_pub = rospy.Publisher(groupid + '/seenfire', Int16, queue_size=3)
    tgt1_pub = rospy.Publisher(groupid + '/seentarget1', Int16, queue_size=3)
    tgt2_pub = rospy.Publisher(groupid + '/seentarget2', Int16, queue_size=3)
    tgt3_pub = rospy.Publisher(groupid + '/seentarget3', Int16, queue_size=3)
    done_pub = rospy.Publisher(groupid + '/done', Int16, queue_size=3)

    rospy.Subscriber(groupid + '/failure', Int16, failure_handle)
    rospy.Subscriber(groupid + '/received', Int16, received_handle)
    rospy.Subscriber(groupid + '/receivedtarget1', Int16, receivedtarget1_handle)
    rospy.Subscriber(groupid + '/receivedtarget2', Int16, receivedtarget2_handle)
    rospy.Subscriber(groupid + '/receivedtarget3', Int16, receivedtarget3_handle)
    rospy.Subscriber(groupid + '/target1', Int16, target1_handle)
    rospy.Subscriber(groupid + '/target2', Int16, target2_handle)
    rospy.Subscriber(groupid + '/target3', Int16, target3_handle)
    app.run(port=5000)
