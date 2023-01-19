import csv
import glob
import os

import cv2
import mss.tools
import nibabel as nib
from matplotlib import pyplot as plt
from pynput import keyboard
from pynput import mouse
import re

'''----------------code snippets of our boxpoints.py file----------------------------'''

def match_image_with_label(volumes_path, segmentation_path):
    image_dict = []
    for image_path in volumes_path:

        for label_path in segmentation_path:
            image_dict.append({"vol": image_path, "seg": label_path})
            break

    return image_dict

in_dir = r'C:\Users\angel\Documents\fhtw\inno\guitest_boundingbox\KiTS'

# path to data
# glob hilft uns die unterschiedlich benannten files zu selecten
volumes = sorted(glob.glob(in_dir + "\\imagesTr\\**\\*.nii"))
segmentation = sorted(glob.glob(in_dir + "\\labelsTr\\**\\*.nii"))
# "\\labelsTr\\*.nii.gz" funktioniert auch --> datensparender aber noch unsicher wegen umgang mit daten

# hier rufen wir dann Clemens funktion auf damit wir uns nicht darum scheren müssen ob alle images label haben
all_files = match_image_with_label(volumes, segmentation)
#print("all files: ", all_files)
print("all files: ", len(all_files))

'''----------------Load the NiFTI scan and extract the scan’s data array----------------------------'''

for i in range(0, 5):
    scanFilePath = all_files[i].get("vol")
    #scanFilePath = all_files[0].get("vol")
    print("scanfilepath: ", scanFilePath)

    # Load the scan and extract data using nibabel
    scan = nib.load(scanFilePath)
    scanArray = scan.get_fdata()

    # Get and print the scan's shape
    scanArrayShape = scanArray.shape
    print('The scan data array has the shape: ', scanArrayShape)

    # Display scan array's middle slices
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Scan Array (Middle Slices)')
    axs[0].imshow(scanArray[scanArrayShape[0] // 2, :, :], cmap='gray')
    axs[1].imshow(scanArray[:, scanArrayShape[1] // 2, :], cmap='gray')
    axs[2].imshow(scanArray[:, :, scanArrayShape[2] // 2], cmap='gray')
    fig.tight_layout()
    plt.show()

    '''-----------------------UI----------------------'''
    global myKey
    myKey = ''
    global xm, ym
    xm, ym = 0, 0


    def on_move(x, y):
        global xm, ym
        xm, ym = x, y
        print('Pointer moved to {0}'.format((xm, ym)))


    def on_click(x, y, button, pressed):
        print('{0} at {1}'.format(
            'Pressed' if pressed else 'Released',
            (x, y)))
        if not pressed:
            # Stop listener
            return False


    def on_scroll(x, y, dx, dy):
        print('Scrolled {0} at {1}'.format(
            'down' if dy < 0 else 'up',
            (x, y)))


    def on_press(key):
        global myKey

        try:
            print('alphanumeric key {0} pressed'.format(key.char))
            myKey = key
        except AttributeError:
            print('special key {0} pressed'.format(key))
            myKey = key


    def on_release(key):
        print('{0} released'.format(
            key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False


    listener = mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll)

    listenerk = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)

    mousePoints = []
    cleanScreen = False
    drawing = False
    writer = None
    with mss.mss() as sct:
        sct.shot()

    global rectDone
    rectDone = False
    drawing = False

    drawing = False
    global x1, y1, x2, y2
    x1, y1, x2, y2 = 0, 0, 0, 0


    def draw_rect(event, x, y, flags, param):
        global x1, y1, drawing, num, img, img2, x2, y2
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x1, y1 = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                a, b = x, y
                if a != x & b != y:
                    img = img2.copy()

                    cv2.rectangle(img, (x1, y1), (x, y), (255, 0, 0), 2)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            num += 1

            font = cv2.FONT_HERSHEY_SIMPLEX
            x2 = x
            y2 = y


    key = ord('a')

    # for i in scanArray_list:
    # use scanArrayShape from extracted NIfTI scan data by NiBabel
    img = scanArray[scanArrayShape[0] // 2, :, :]
    '''scanArrayShape = scanArray_list[i].shape
    img = scanArray_list[i][scanArrayShape[0]//2,:,:]'''

    img2 = img

    print(type(img))
    #img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2BGR)
    #img_gray_bgr = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    #img2 = cv2.cvtColor(img_gray, img_rgb, CV_GRAY2RGB)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.namedWindow("main", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("main", draw_rect)
    cv2.setWindowProperty("main", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    num = 0
    # PRESS w to confirm save selected bounded box
    while key != ord('w'):
        cv2.imshow("main", img)
        key = cv2.waitKey(1) & 0xFF
    print('Here are points:', x1, y1, x2, y2)
    if key == ord('w'):
        cv2.imwrite('snap' + repr(i) + '.png', img2[y1:y2, x1:x2])
        cv2.destroyAllWindows()
        print('Saved as snap.png')
        #os.remove('monitor-1.png')


    '''-----------------------save points to csv file----------------------'''
    # Define the structure of the data
    csv_header = ['case', 'x1', 'y1', 'x2', 'y2']

    # get case name
    case = re.search('case_(.+?).nii', scanFilePath).group(1)
    print(case)

    # Define the actual data
    box_data = [scanFilePath, x1, y1, x2, y2]

    # 1. Open a new CSV file
    file_exists = os.path.isfile('bounding_box_points.csv')

    with open('bounding_box_points.csv', 'a+') as file:
        # 2. Create a CSV writer
        writer = csv.writer(file,  lineterminator='\r')

        # 3. if file new created add header
        if not file_exists:
            writer.writerow(csv_header)

        # 4. Write data to the file
        writer.writerow(box_data)

        #5. close file
        file.close()


