import cv2
import numpy as np
import os
import glob
import copy

# ------------------ Parameters ------------------ #
images_folder = 'temp/annotation_example/images'  # input images
masks_folder = 'temp/annotation_example/masks'   # baseline masks
save_folder = 'temp/annotation_example/masks_refined'
os.makedirs(save_folder, exist_ok=True)

eraser_brush_size = 5
draw_brush_size = 2
mode = 'draw'  # default mode

# ------------------ Helper Functions ------------------ #
def overlay_mask(img, mask, color, alpha):
    """Overlay mask on image. mask: binary 0/255"""
    overlay = img.copy()
    overlay[mask>0] = color
    return cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

# ------------------ Mouse Callback ------------------ #
drawing = False
def draw(event, x, y, flags, param):
    global drawing, mask_display, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            if mode == 'draw':
                cv2.circle(mask_display, (x,y), draw_brush_size, 255, -1)
            elif mode == 'erase':
                cv2.circle(mask_display, (x,y), eraser_brush_size, 0, -1)

# ------------------ Main Loop ------------------ #
image_files = sorted(glob.glob(os.path.join(images_folder, '*.png')))

cv2.namedWindow('mask_editor', cv2.WINDOW_NORMAL)
cv2.setMouseCallback('mask_editor', draw)

for img_path in image_files:
    img_name = os.path.basename(img_path)
    mask_path = os.path.join(masks_folder, img_name)
    
    # Load image and mask
    img = cv2.imread(img_path)
    if not os.path.exists(mask_path):
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
    else:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    mask_display = mask.copy()
    mask_backup = mask.copy()  # for redo
    
    while True:
        overlay = overlay_mask(img, mask_display, color=(0,0,255), alpha=0.3)
        cv2.imshow('mask_editor', overlay)
        # Mask to get lower 8 bits for key code compatibility (OpenCV returns platform-dependent values)
        k = cv2.waitKey(1) & 0xFF
        
        if k == ord('d'):
            mode = 'draw'
            print("Mode: DRAW")
        elif k == ord('e'):
            mode = 'erase'
            print("Mode: ERASE")
        elif k == ord('r'):
            mask_display = mask_backup.copy()
            print("Reset: mask restored to original")
        elif k == ord('s'):
            save_path = os.path.join(save_folder, img_name)
            cv2.imwrite(save_path, mask_display)
            print(f"Saved refined mask: {save_path}")
            break
        elif k == ord('n'):
            print(f"Skipped image: {img_name}")
            break
        elif k == ord('q'):
            print("Quit editor")
            exit()

cv2.destroyAllWindows()
