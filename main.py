"""Main module of image preprocessing
GUI.
"""

# tkinter
from tkinter import *
from tkinter import filedialog

# pillow
from PIL import ImageTk, Image 

# Local
import os
import glob

# Machine learning
from utils import get_cfg, load_obj, collate_fn, eval_model, flatten_omegaconf
import albumentations as A
from omegaconf import OmegaConf
from ml import ImgDataset, LitImg
from torch.utils.data import DataLoader
import torch


# Window properties.
WINDOW_SIZE = 690
HEIGHT = 1000
WIDTH = 1200
BORDER_PADDING = 5
CANVAS_BACKGROUND = '#243646'
ENTRY_BACKGROUND = '#ffffff'
BUTTON_BACKGROUND = '#006699'
GENERAL_BACKGROUND = '#CCFFFF'

# Font properties
FONT_COLOR = '#C0C0C0'
FONT_SIZE = 13
FONT_SIZE_LABEL = 11
FONT_SIZE_LOGO = 20

class CropCounter():
    """Handles the GUI for
    the cropcounter
    """

    def __init__(self,master):
        """Set up the main frame."""
        self.master = master
        self.master.title('Crop counter')
        self.frame = Frame(self.master, 
                           bg=CANVAS_BACKGROUND, 
                           height=HEIGHT,
                           width=WIDTH)
        self.frame.pack(fill='both')
       # self.master.resizable(width=False, height=False)

        self.cur = 0
        self.total = 0

        self.svSourcePath = StringVar()
        self.svSourcePath.set(os.getcwd())
        # ------------------ GUI stuff --------------------------------
        # Logo image
        self.logo_image = ImageTk.PhotoImage(Image.open('./logo_image.png'))
        self.logo_label = Label(self.frame,
                                image=self.logo_image)
        self.logo_label.place(relx=0.23,
                              rely=0.03,
                              relheight=0.08,
                              relwidth=0.23,
                              anchor='ne')
        self.dir_label = Label(self.frame, 
                               font=('Courier', (FONT_SIZE+4)),
                               text='Image Dir:',
                               bg=CANVAS_BACKGROUND,
                               fg=FONT_COLOR)
        self.dir_label.place(relx=0.27,
                             rely=0.03)
        
        # Directory entry and button
        self.entry = Entry(self.frame,
                           textvariable=self.svSourcePath)
        self.entry.place(relx=0.27,
                         rely=0.08,
                         relheight=0.045,
                         relwidth=0.5,)
        self.srcDirBtn = Button(self.frame, 
                                bg=BUTTON_BACKGROUND,
                                fg=GENERAL_BACKGROUND,
                                activebackground=BUTTON_BACKGROUND,
                                relief='flat',
                                text="Image input folder", 
                                command=self.selectSrcDir)
        self.srcDirBtn.place(relx=0.78,
                             rely=0.08)
        # Output textbox
        self.text_label = Label(self.frame,
                                font=('Courier', (FONT_SIZE+4)),
                                justify='left',
                                bd=BORDER_PADDING,
                                bg=CANVAS_BACKGROUND,
                                fg=FONT_COLOR,
                                text='Welcome to the crop counting assitant \nplease select a folder to begin.')
        self.text_label.place(rely=0.5,
                              relx=0.5,
                              anchor=N)

        # Submit button
        self.ldBtn = Button(self.frame,
                            bg=BUTTON_BACKGROUND,
                            fg=GENERAL_BACKGROUND,
                            activebackground=BUTTON_BACKGROUND,
                            relief='flat',
                            text='Load',
                            command = self.loadDir)
        self.ldBtn.place(relx=0.8,
                         rely=0.9)


    def selectSrcDir(self):
        path = filedialog.askdirectory(title="Select image source folder", 
                                       initialdir=self.svSourcePath.get())
        self.svSourcePath.set(path)
        return


    def loadImage(self):
        # load image
        global WINDOW_SIZE
        self.imagepath = self.imageList[self.cur - 1]
        self.img = Image.open(self.imagepath)
        self.scale = min(WINDOW_SIZE / self.img.size[0], WINDOW_SIZE / self.img.size[1])
        self.imgnewsize = int(self.scale * self.img.size[0]), int(self.scale * self.img.size[1])
        self.tkimg = ImageTk.PhotoImage(self.img.resize(self.imgnewsize,Image.ANTIALIAS))
        self.mainPanel = Canvas(self.frame, cursor='tcross')
        self.mainPanel.place(relx=0.5,
                             rely=0.15,
                             anchor=N)
        
        self.mainPanel.config(width = self.tkimg.width(), height = self.tkimg.height())
        self.mainPanel.create_image(0, 0, image = self.tkimg, anchor=NW)
        self.progLabel = Label(self.frame,
                               font=('Courier', (FONT_SIZE+4)),
                               bd=BORDER_PADDING,
                               bg=CANVAS_BACKGROUND,
                               fg=FONT_COLOR,
                               text=f'Progress: {self.cur}  /  {self.total} ')
        self.progLabel.place(relx=0.3,
                             rely=0.9,
                             anchor=N)
        self.prevBtn = Button(self.frame,
                              bg=BUTTON_BACKGROUND,
                              fg=GENERAL_BACKGROUND,
                              activebackground=BUTTON_BACKGROUND,
                              relief='flat', 
                              text='<< Prev', 
                              command = self.prevImage)
        self.prevBtn.place(relx=0.5,
                           rely=0.9,
                           anchor=N)
        self.nextBtn = Button(self.frame,
                              bg=BUTTON_BACKGROUND,
                              fg=GENERAL_BACKGROUND,
                              activebackground=BUTTON_BACKGROUND,
                              relief='flat', 
                              text='Next >>', 
                              command = self.nextImage)
        self.nextBtn.place(relx=0.6,
                           rely=0.9,
                           anchor=N)
        self.sub_button = Button(self.frame,
                                 bg=BUTTON_BACKGROUND,
                                 fg=GENERAL_BACKGROUND,
                                 activebackground=BUTTON_BACKGROUND,
                                 relief='flat',
                                 text='Count crops',
                                 command=lambda: self.get_crops())
        self.sub_button.place(rely=0.9,
                              relx=0.9)
        self.text_label.config(text='Crop: \n'+str(self.imageDir)[40:])
        self.text_label.place_configure(rely=0.2,
                                        relx=0.85,
                                        anchor=N)


    def loadDir(self, dbg = False):
        """Load selected dir."""
        if not dbg:
            s = self.entry.get()
            self.master.focus()
        # get image list
        self.imageDir = self.svSourcePath.get()
        
        self.imageList = glob.glob(os.path.join(self.imageDir, '*.JPG'))
        
        if len(self.imageList) == 0:
            self.text_label.config(text='No .jpg images found in the specified dir!')
            return

        # default to the 1st image in the collection
        self.cur = 1
        self.total = len(self.imageList)

        self.loadImage()
        print('%d images loaded from %s' %(self.total, s))


    def prevImage(self):
        """Load previous image."""
        if self.cur > 1:
            self.cur -= 1
            self.loadImage()


    def nextImage(self):
        """Load next image."""
        if self.cur < self.total:
            self.cur += 1
            self.loadImage()


    def get_crops(self):
        """Handles the request 
        of the API and input of the image
        """

        cfg = get_cfg()

        valid_augs_list = [load_obj(i['class_name'])(**i['params']) for i in cfg['augmentation']['valid']['augs']]
        valid_bbox_params = OmegaConf.to_container((cfg['augmentation']['valid']['bbox_params']))
        valid_augs = A.Compose(valid_augs_list, bbox_params=valid_bbox_params)
        
        test_dataset = ImgDataset(None,
                                  'test',
                                  self.imageDir,
                                  cfg,
                                  valid_augs)
        
        test_loader = DataLoader(test_dataset,
                                 batch_size=cfg.data.batch_size,
                                 num_workers=cfg.data.num_workers,
                                 shuffle=False,
                                 collate_fn=collate_fn)
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
        model = torch.load(os.path.dirname(os.path.abspath(__file__))+f'/{str(self.imageDir)[40:].lower()}/model.pth', 
                           map_location=device)

        detection_threshold = 0.5
        results = []
        model.eval()

        hparams = flatten_omegaconf(cfg)

        lit_model = LitImg(hparams=hparams, 
                           cfg=cfg, 
                           model=model)

        self.results = eval_model(test_loader, 
                                  results, 
                                  detection_threshold, 
                                  device, 
                                  lit_model)

        for i in range(len(self.results)):
            self.mainPanel.create_rectangle(int(int(self.results[i]['x1'])*self.scale), 
                                            int(int(self.results[i]['y1'])*self.scale),
                                            int(int(self.results[i]['x2'])*self.scale),
                                            int(int(self.results[i]['y2'])*self.scale),
                                            width=2,
                                            outline='red')
        
        self.text_label.config(text='Crop: \n'+str(self.imageDir)[40:]+'\nTotal: \n'+str(len(self.results)))

if __name__=='__main__':
    root = Tk()
    tool = CropCounter(root)
    root.mainloop()