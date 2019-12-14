# tello_final
## 大致结构
* image_detect
  * [yolo_v3](https://github.com/ultralytics/yolov3)，权重文件过大，并没有上传到repo里

* mydetect
  * [efficientdet](https://github.com/toandaominh1997/EfficientDet.Pytorch)，
  已经将训练好的权重随repo发布了，可以直接使用

* control  
  这个模块里包含了主要的功能，模仿分布式服务来写灵活的代码！
  * tello_center
    * 模仿分布式服务的一个服务中心，
    通过这种模式来方便配置的处理以及模仿的简单的服务注册功能，
    来实现一种灵活的代码结构
    * 通过代理模式来方便服务的发现和调用
  * tello_abs
    * 内含修改自Tello官方SDK的无人机控制类，
    以及处理无人机初始化的一个服务。
    * ~~一个响应式的处理图像和state变换的服务~~（并没有起到特别大的作用）
  * tello_image_process
    * 设计之初用于托管处理所有的图像识别的服务，
    最后实际只是用来显示无人机视角和状态（只需要注册了服务就会开启这个功能）
    * ~~检测着火点~~（为了节省写代码的时间检测过程都被直接写进了控制代码里）
  * tello_imshow
    * 注册这个服务后就会把cv2的imshow功能做一个替换（实际上是替换了locks.imshow),
    在一个单独的线程里执行显示图像的功能，防止在Linux上多线程使用imshow导致的锁死。
  * tello_judge_client
    * 用于处理和上位机上报比赛流程的服务，
    包含了client以及一个对和上位机交互的抽象服务，通过改变注册的这个抽象服务类别
    来做到本地运行测试和在上位机上运行测试的特定环境部署特定功能。
    * ~~我一开始也没想到可以用这种方法来做到差异部署~~
  * ~~tello_panda（弃用）~~
    * 本来是一个显示当前无人机位置和视角碰撞的模块，
    但是定位毯实在太不靠谱了 :(
    * 虽然很帅但是没啥实际作用（花里胡哨）
  * ~~tello_world（弃用）~~
    * 存储世界模型以及处理射线检测和碰撞的模块，弃用理由同上。
  * tello_save_video
    * 保存无人机摄像头视频的服务（用于debug）
    * 注册即可使用，退出时自动保存
  * tello_yolo
    * 提供图像识别的功能，并对原yolo检测和新的efficientdet统一接口。
    * ~~改一个设置就可以随便选识别算法，它不香吗~~
    * ~~为了处理pytorch只能在主线程跑的问题加了个把检测转发到主线程的功能~~（过于丢人的修bug）
  * tello_main
    * 测试时使用的主入口，算是一个修改配置和服务注册的例子  
  * tello_control
    * 控制整个比赛流程的服务（~~实际上为了方便把大部分流程写进一个函数了~~）
    * 控制无人机移动的功能其实就写了两个简单的函数
    （utils.drone_util里的goto和look_at)
  * 几乎所有模块都有单元测试，方便debug
  
  
