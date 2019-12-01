import cv2

def display_saved_imgs(time=50):
    idx = 0
    while True:
        img = cv2.imread("./save/p%d.png" % idx)
        if img is None:
            break
        cv2.imshow("saved img", img)
        cv2.waitKey(time)
        idx += 1

if __name__ == '__main__':
    display_saved_imgs(10000)
