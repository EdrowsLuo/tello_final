# Tello_Control
相关信息见[主repo](https://github.com/zoeyuchao/tello_control/)  
这是初赛的一个完整实现

## 运行
基本的运行环境要求参考[主repo](https://github.com/zoeyuchao/tello_control/)  
因为彻底脱离了ros，直接运行
```cmd
python tello_state.py （直接从原版进行修改的，比较乱，而且图像刷新只会在指令传输完成后进行，但是基础功能已经完善了）
```
或者
```cmd
python tello_wrap.py （多线程版本，但是有一点点问题）
```
即可  
__Ctrl+C__ 终止后会自动降落

## 调试
通过指定:
```python
drone.stop = True
```
就可以让所有传到drone里的指令都不会实际生效，可以通过这个来手动移动无人机来进行测试。

所有无人机运行过程中关键点的摄像头画面以及信息都会保存在./save文件夹里，通过运行：
```cmd
python view_saved_img.py
```
就可以逐帧的进行检查



