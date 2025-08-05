import cv2
import numpy as np
s = 0  # Specify 0 for accessing the web camera.
source = cv2.VideoCapture(s)
frame_h = int(source.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(source.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_fps = int(source.get(cv2.CAP_PROP_FPS))
size = (frame_w, frame_h)
# Create a window to display the video stream.
video_filter = cv2.VideoWriter('videofilter.mp4', cv2.VideoWriter_fourcc(*'mp4v'), frame_fps, size)
# win_name = 'Filter Demo'
# cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)

def shadow(img,level=2):
    height, width = img.shape[:2]
    x_kernel = cv2.getGaussianKernel(width,width/level)
    y_kernel = cv2.getGaussianKernel(height,height/level)

    kernel = y_kernel*x_kernel.T
    mask = kernel/kernel.max()
    shadow_img = img.copy()

    for i in range(3):
        shadow_img[:,:,i] = shadow_img[:,:,i]*mask
    return shadow_img

def vintage(img):
    img_sepia = img.copy()
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB)
    img_sepia = np.array(img_sepia,dtype=np.uint64)
    img_sepia = cv2.transform(img_sepia,np.matrix([[0.393, 0.769, 0.189],
                                                    [0.349, 0.686, 0.168],
                                                    [0.272, 0.534, 0.131]]))

    img_sepia = np.clip(img_sepia, 0, 255)
    img_sepia = np.array(img_sepia,dtype=np.uint8)
    img_sepia - cv2.cvtColor(img_sepia,cv2.COLOR_RGB2BGR)
    return img_sepia

def edge(img):
    kernel = np.array([[0,-3,-3],[3,0,-3],[3,3,0]])

    image = cv2.filter2D(img,-1,kernel)
    return image

PREVIEW = 0   # Preview Mode
CANNY   = 1   # Canny Edge Detector
bright = 2
style = 3
gray = 4
pencil = 5
shadoww = 6
vin = 7
edegee = 8
image_filter = PREVIEW
result = None

while True:
    has_frame, frame = source.read()
    if not has_frame:
        break
    # Flip video frame for convenience.
    frame = cv2.flip(frame,1)
    if image_filter == PREVIEW:
        result = frame
    elif image_filter == CANNY:
        result = cv2.Canny(frame, 50, 120)
    elif image_filter == bright:
        result = cv2.convertScaleAbs(frame, beta=25)
    elif image_filter == style:
        img_blur = cv2.GaussianBlur(frame, (5, 5), 0, 0)
        result = cv2.stylization(img_blur, sigma_s=100, sigma_r=0.1)
    elif image_filter == gray:
        result = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif image_filter == pencil:
        result,_ = cv2.pencilSketch(frame)
    elif image_filter == shadoww:
        result = shadow(frame)
    elif image_filter == vin:
        result = vintage(frame)
    elif image_filter == edegee:
        result = edge(frame)

    video_filter.write(result)
    cv2.imshow('Different filters', result)


    key = cv2.waitKey(1)
    if key == ord('Q') or key == ord('q') or key == 27:
        break
    elif key == ord('C') or key == ord('c'):
        image_filter = CANNY
    elif key == ord('P') or key == ord('p'):
        image_filter = PREVIEW
    elif key == ord('B') or key == ord('b'):
        image_filter = bright
    elif key == ord('S') or key == ord('s'):
        image_filter = style
    elif key == ord('G') or key == ord('g'):
        image_filter = gray
    elif key == ord('L') or key == ord('l'):
        image_filter = pencil
    elif key == ord('R') or key == ord('r'):
        image_filter = shadoww
    elif key == ord('V') or key == ord('v'):
        image_filter = vin
    elif key == ord('E') or key == ord('e'):
        image_filter = edegee


source.release()
video_filter.release()
cv2.destroyAllWindows()
