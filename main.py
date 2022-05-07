# Import the tkinter library
import tkinter as tk
from tkinter import *
import numpy as np
import cv2
from PIL import Image, ImageTk, ImageChops
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import random
from imageai.Detection import ObjectDetection
import os
import pixellib
from pixellib.semantic import semantic_segmentation

input_path =""
# Create an instance of tkinter frame
class main_gui(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Editor")
        self.w = self.winfo_screenwidth()
        self.h = self.winfo_screenheight()
        self.geometry("%dx%d+0+0" % (self.w, self.h))
        self.state("zoomed")

        # Create a Label to display the image
        self.lbl_modified = Label(self, bg="green", text="MANIPULATED IMAGE AREA", font=("Allegro", 12, "bold")).place(
            relx=0.5, rely=0, relwidth=0.5, relheight=1)
        self.loaded_image = Label(self, text="LOADED IMAGE AREA", font=("Allegro", 12, "bold"))
        self.loaded_image.place(relx=0, rely=0, relwidth=0.5, relheight=1)

        # Filemenu
        menubar = Menu(self)
        self.config(menu=menubar)

        filemenu_file = Menu(menubar, tearoff=0)
        filemenu_edit = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=filemenu_file)
        filemenu_file.add_command(label="Load Image", command=self.loadImage, underline=0)
        filemenu_file.add_command(label="Save Image", command=self.saveImage, underline=0)
        menubar.add_cascade(label="Edit", menu=filemenu_edit)
        filemenu_edit.add_command(label="Blur Image", command=self.blurImage, underline=0)
        filemenu_edit.add_command(label="GrayScale Image", command=self.grayScale, underline=0)
        filemenu_edit.add_command(label="Crop Image", command=self.cropImage, underline=0)
        filemenu_edit.add_command(label="Flip Image", command=self.flipImage, underline=0)
        filemenu_edit.add_command(label="Mirror Image", command=self.mirrorImage, underline=0)
        filemenu_edit.add_command(label="Rotate Image", command=self.rotateImage, underline=0)
        filemenu_edit.add_command(label="Reverse Color", command=self.reverseColor, underline=0)
        filemenu_edit.add_command(label="Change Color Balance", command=self.changeColorBalance, underline=0)
        filemenu_edit.add_command(label="Adjust Brightness", command=self.adjustBrightness, underline=0)
        filemenu_edit.add_command(label="Adjust Contrast", command=self.adjustContrast, underline=0)
        filemenu_edit.add_command(label="Adjust Saturation", command=self.adjustSaturation, underline=0)
        filemenu_edit.add_command(label="Add Noise", command=self.addNoise, underline=0)
        filemenu_edit.add_command(label="Detect Edges", command=self.detectEdges, underline=0)
        filemenu_edit.add_command(label="Reset Image", command=self.reset_image, underline=0)
        filemenu_edit.add_command(label="Detect Object", command=self.objectDetection, underline=0)
        filemenu_edit.add_command(label="Segmentation", command=self.segmentation, underline=0)
        filemenu_edit.add_command(label="Remove Noise", command=self.median_filter, underline=0)
        filemenu_edit.add_command(label="Detect Face", command=self.face_recognition, underline=0)
        filemenu_edit.add_command(label="Change HSV", command=self.hsv_color, underline=0)
        filemenu_edit.add_command(label="Change LAB", command=self.lab_color, underline=0)

    def loadImage(self):
        global input_path
        path = self.select_file()
        print("image loaded")
        # Load the image
        self.img = cv2.imread(path)
        self.gray_scale = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        print(self.gray_scale.shape)
        input_path = path
        print(input_path)
        # Rearrange colors
        blue, green, red = cv2.split(self.img)
        self.img = cv2.merge((red, green, blue))
        self.im = Image.fromarray(self.img)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = self.im.resize((width, length), Image.ANTIALIAS)
        self.imgtk = ImageTk.PhotoImage(image=resized_image)
        self.loaded_image = Label(self, image=self.imgtk, bg="red")
        self.loaded_image.place(relx=0, rely=0, relwidth=0.5, relheight=1)

    def blurImage(self):
        print("image blurred")
        self.blurImg = cv2.blur(self.img, (10, 10))
        self.im_modified = Image.fromarray(self.blurImg)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.img_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.img_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    def grayScale(self):
        print("image grayscaled")
        self.gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.im_modified = Image.fromarray(self.gray_image)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.img_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.img_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    def flipImage(self):
        print("image flipped")
        flipped_image = cv2.flip(self.img, 0)
        self.im_modified = Image.fromarray(flipped_image)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.img_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.img_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    def mirrorImage(self):
        print("image mirrored")
        mirrored_image = np.flip(self.img, axis=1)
        self.im_modified = Image.fromarray(mirrored_image)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.img_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.img_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    def rotateImage(self):
        print("image rotated")

        def rotateNow(val):
            print(val)
            (rows, cols) = self.img.shape[:2]
            # getRotationMatrix2D creates a matrix needed for transformation.
            # We want matrix for rotation w.r.t center to 45 degree without scaling.
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), int(round(float(val), 0)), 1)
            res = cv2.warpAffine(self.img, M, (cols, rows))
            self.im_modified = Image.fromarray(res)
            width = int(round(self.w / 2, 0))
            length = int(round(self.h, 0))
            resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
            self.img_modified = ImageTk.PhotoImage(image=resized_image)
            self.lbl_modified = Label(self, bg="green", image=self.img_modified)
            self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

        rotate_window = Toplevel(self)
        rotate_window.title("Rotate Image")
        w = 250
        h = 75
        rotate_window.geometry("{}x{}".format(w, h))
        Label(rotate_window, text="Rotation Degree").grid(column=0, row=0, sticky=W)
        rotation = Scale(rotate_window, from_=-180, to=180, orient=HORIZONTAL, command=rotateNow)
        rotation.grid(column=2, row=0, sticky=W)

    def reverseColor(self):
        print("image color reversed")
        self.im_modified = ImageChops.invert(self.im)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.img_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.img_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    def cropImage(self):
        print("image cropped")
        crop_window = Toplevel(self)
        crop_window.title("Crop Coordinates")
        w = 320
        h = 150
        crop_window.geometry("{}x{}".format(w, h))
        # configure the grid
        crop_x_start = IntVar()
        crop_x_end = IntVar()
        crop_y_start = IntVar()
        crop_y_end = IntVar()

        def cropNow():
            print(self.img.shape)
            cropped_image = self.img[crop_y_start.get():crop_y_end.get(), crop_x_start.get():crop_x_end.get()]
            self.im_modified = Image.fromarray(cropped_image)
            width = int(round(self.w / 2, 0))
            length = int(round(self.h, 0))
            resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
            self.img_modified = ImageTk.PhotoImage(image=resized_image)
            self.lbl_modified = Label(self, bg="green", image=self.img_modified)
            self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

        Label(crop_window, text="Original Image Size Y,X: {}".format(self.img.shape)).grid(column=0, row=0, sticky=W)
        Label(crop_window, text="Crop Y End").grid(column=0, row=4, sticky=W)
        Entry(crop_window, textvariable=crop_y_end).grid(column=1, row=4, sticky=E)
        Label(crop_window, text="Crop Y Start").grid(column=0, row=3, sticky=W)
        Entry(crop_window, textvariable=crop_y_start).grid(column=1, row=3, sticky=E)
        Label(crop_window, text="Crop X End").grid(column=0, row=2, sticky=W)
        Entry(crop_window, textvariable=crop_x_end).grid(column=1, row=2, sticky=E)
        Label(crop_window, text="Crop X Start").grid(column=0, row=1, sticky=W)
        Entry(crop_window, textvariable=crop_x_start).grid(column=1, row=1, sticky=E)
        Button(crop_window, text="CROP", command=cropNow).grid(column=1, row=5, sticky=W)

    def changeColorBalance(self):
        image = self.img

        def color_balance(balance):
            print(balance)
            balance = int(round(float(balance), 0))
            image2 = np.zeros(image.shape)
            image2[:, :, 0] = ((1 + 2 * balance) * image[:, :, 0] + (1 - balance) * image[:, :, 1] + (
                    1 - balance) * image[:, :, 2]) / 3
            image2[:, :, 1] = ((1 + 2 * balance) * image[:, :, 1] + (1 - balance) * image[:, :, 0] + (
                    1 - balance) * image[:, :, 2]) / 3
            image2[:, :, 2] = ((1 + 2 * balance) * image[:, :, 2] + (1 - balance) * image[:, :, 0] + (
                    1 - balance) * image[:, :, 1]) / 3
            image2 = image2 / 255
            self.im_modified = Image.fromarray(image2.astype(np.uint8))
            width = int(round(self.w / 2, 0))
            length = int(round(self.h, 0))
            resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
            self.img_modified = ImageTk.PhotoImage(image=resized_image)
            self.lbl_modified = Label(self, bg="green", image=self.img_modified)
            self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

        balance_window = Toplevel(self)
        balance_window.title("Color Balance")
        w = 250
        h = 75
        balance_window.geometry("{}x{}".format(w, h))
        Label(balance_window, text="Color Balance").grid(column=0, row=0, sticky=W)
        balance = Scale(balance_window, from_=0, to=10, orient=HORIZONTAL, command=color_balance)
        balance.grid(column=2, row=0, sticky=W)

    def adjustBrightness(self):
        def scale_brightness(val):
            print(val)
            new_image = cv2.convertScaleAbs(self.img, alpha=1, beta=int(round(float(val), 0)))

            self.im_modified = Image.fromarray(new_image)
            width = int(round(self.w / 2, 0))
            length = int(round(self.h, 0))
            resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
            self.img_modified = ImageTk.PhotoImage(image=resized_image)
            self.lbl_modified = Label(self, bg="green", image=self.img_modified)
            self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

        brightness_window = Toplevel(self)
        brightness_window.title("Brightness")
        w = 250
        h = 75
        brightness_window.geometry("{}x{}".format(w, h))
        Label(brightness_window, text="Brightness").grid(column=0, row=0, sticky=W)
        brightness = Scale(brightness_window, from_=-255, to=255, orient=HORIZONTAL, command=scale_brightness)
        brightness.grid(column=2, row=0, sticky=W)

    def adjustContrast(self):
        def scale_contrast(val):
            print(val)
            new_image = cv2.convertScaleAbs(self.img, alpha=int(round(float(val), 0)), beta=0)

            self.im_modified = Image.fromarray(new_image)
            width = int(round(self.w / 2, 0))
            length = int(round(self.h, 0))
            resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
            self.img_modified = ImageTk.PhotoImage(image=resized_image)
            self.lbl_modified = Label(self, bg="green", image=self.img_modified)
            self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

        contrast_window = Toplevel(self)
        contrast_window.title("Contrast")
        w = 250
        h = 75
        contrast_window.geometry("{}x{}".format(w, h))
        Label(contrast_window, text="Contrast").grid(column=0, row=0, sticky=W)
        contrast = Scale(contrast_window, from_=-10, to=10, orient=HORIZONTAL, command=scale_contrast)
        contrast.grid(column=2, row=0, sticky=W)

    def adjustSaturation(self):
        hsv = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        greenMask = cv2.inRange(hsv, (26, 10, 30), (97, 100, 255))

        hsv[:, :, 1] = greenMask

        back = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        self.im_modified = Image.fromarray(back)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.img_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.img_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    def addNoise(self):
        row, col = self.gray_scale.shape
        print(row, col)
        number_of_pixels = random.randint(50000, 100000)
        print(number_of_pixels)
        for i in range(number_of_pixels):
            # Pick a random y coordinate
            y_coord = random.randint(0, row - 1)

            # Pick a random x coordinate
            x_coord = random.randint(0, col - 1)

            # Color that pixel to black
            self.gray_scale[y_coord][x_coord] = 0

        self.im_modified = Image.fromarray(self.gray_scale)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.img_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.img_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    def detectEdges(self):
        edges = cv2.Canny(self.img, 100, 200)
        self.im_modified = Image.fromarray(edges)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.img_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.img_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    def reset_image(self):
        im_resetted = Image.fromarray(self.img)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = im_resetted.resize((width, length), Image.ANTIALIAS)
        self.img_resetted = ImageTk.PhotoImage(image=resized_image)
        self.original_image = Label(self, image=self.img_resetted, bg="green")
        self.original_image.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    def select_file(self):
        filetypes = (
            ('png files', '*.png'),
            ('jpg files', '*.jpg'),
            ('All files', '*.*')
        )

        filename = fd.askopenfilename(
            title='Open a file',
            initialdir='/',
            filetypes=filetypes)

        return filename

    def saveImage(self):
        print("image saved")
        filename = fd.asksaveasfile(mode='w', defaultextension=".jpg")
        if not filename:
            return
        self.im_modified.save(filename)

    def objectDetection(self):
        execution_path = os.getcwd()
        self.im_modified = Image.fromarray(self.img)
        detector = ObjectDetection()
        detector.setModelTypeAsRetinaNet()
        detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.1.0.h5"))
        detector.loadModel()
        detected = detector.detectObjectsFromImage(input_image= input_path, output_image_path=os.path.join(execution_path , "out.jpg"))
        out_img = cv2.imread(os.path.join(execution_path , "out.jpg"))
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        self.im_modified = Image.fromarray(out_img)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.img_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.img_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
        # for eachObject in detections:
        #     print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
        #C:\Users\akrtt\PycharmProjects\BBM415_Project
    def segmentation(self):
        execution_path = os.getcwd()
        segment_image = semantic_segmentation()
        segment_image.load_pascalvoc_model("deeplabv3_xception_tf_dim_ordering_tf_kernels.h5") 
        segment_image.segmentAsPascalvoc(input_path, output_image_name = os.path.join(execution_path , "out.jpg"))
        out_img = cv2.imread(os.path.join(execution_path , "out.jpg"))
        out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
        self.im_modified = Image.fromarray(out_img)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.img_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.img_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
    def median_filter(self):
        median = cv2.medianBlur(self.img, 5)
        self.im_modified = Image.fromarray(median)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.img_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.img_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
    
    def face_recognition(self):
        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        self.gray_image = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        self.im_modified = self.img
        faces = faceCascade.detectMultiScale(
            self.gray_image,
            scaleFactor=1.1,
            minNeighbors=7,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in faces:
            cv2.rectangle(self.im_modified, (x, y), (x + w, y + h), (0, 255, 0), 2)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        self.im_modified = Image.fromarray(self.im_modified)
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.im_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.im_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)
    
    def hsv_color(self):
        hsv_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2HSV)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        self.im_modified = Image.fromarray(hsv_img)
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.im_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.im_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

    def lab_color(self):
        lab_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2LAB)
        width = int(round(self.w / 2, 0))
        length = int(round(self.h, 0))
        self.im_modified = Image.fromarray(lab_img)
        resized_image = self.im_modified.resize((width, length), Image.ANTIALIAS)
        self.im_modified = ImageTk.PhotoImage(image=resized_image)
        self.lbl_modified = Label(self, bg="green", image=self.im_modified)
        self.lbl_modified.place(relx=0.5, rely=0, relwidth=0.5, relheight=1)

gui = main_gui()
gui.mainloop()
