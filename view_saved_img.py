import cv2

def display_saved_imgs():
    idx = 0
    while True:
        img = cv2.imread("./save/p%d.png" % idx)
        if img is None:
            break
        cv2.imshow("saved img", img)
        cv2.waitKey(0)
        idx += 1

if __name__ == '__main__':
    display_saved_imgs()
