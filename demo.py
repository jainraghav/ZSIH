from tkinter import *
from tkinter import ttk,colorchooser,filedialog
from tkinter.ttk import Progressbar
import pyscreenshot as ImageGrab
from PIL import Image,ImageDraw,ImageTk
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

import numpy as np
import pandas as pd
from os import listdir
from options import Options
from Models import net
from torch.utils.data.dataset import Dataset

class Testdata(Dataset):
    def __init__(self, datafile, datapath, transform=None):
        df = pd.read_csv(datafile)
        self.datapath = datapath
        self.transform = transform
        self.X_train = df['ImagePath']
        arr = df['ImagePath'].values.tolist()

    def __getitem__(self, index):
        img = Image.open(self.datapath + self.X_train[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        img_path = self.datapath + self.X_train[index]
        return img,img_path

    def __len__(self):
        return len(self.X_train)

final = []

def prep(path):
    filepath = path + "filelist-test.txt"
    all_imgs = listdir(path)
    for x in all_imgs[:]:
        if not(x.endswith(".jpeg") or x.endswith(".JPEG") or x.endswith(".png") or x.endswith(".PNG")):
            all_imgs.remove(x)
    with open(filepath, "w") as f:
        f.write("ImagePath\n")
        for line in all_imgs:
            f.write(line+"\n")
    return filepath

def exec_it(smod,imod,input,ret_pool):
    all_image_hashes = []
    img_paths=[]
    for batch_idx,(data_i,path) in enumerate(ret_pool):
        data_i = data_i.cuda()
        fx,_,blah = imod(data_i-0.5)
        fxd = fx.cpu().detach().numpy()
        imageb_hash = (np.sign(fxd)+1)/2
        # imageb_hash = fxd
        all_image_hashes.extend(imageb_hash)
        img_paths.extend(path)

    img_paths = np.array(img_paths)
    sketch_hash = (np.sign(smod(input.unsqueeze(0).cuda()-0.5)[0].cpu().detach().numpy())+1)/2
    hamm_d = cdist(sketch_hash, all_image_hashes, 'hamming')
    top10 = hamm_d[0].argsort()[:10]
    retrieved_imgs = img_paths[top10]

    return retrieved_imgs

def getFileName(image):
    print(image)

class main:
    def __init__(self,master,args):
        self.master = master
        self.color_fg = 'black'
        self.color_bg = 'white'
        self.old_x = None
        self.old_y = None
        self.penwidth = 5
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)
        self.c.bind('<ButtonRelease-1>',self.reset)
        self.sketch_model = args.sketch_model
        self.image_model = args.image_model
        self.img_path = args.img_path
        self.hashcode_length = args.hashcode_length

    def changeW(self,e):
        self.penwidth = e

    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)
        self.old_x = e.x
        self.old_y = e.y

    def reset(self,e):
        self.old_x = None
        self.old_y = None

    def save_exec(self):

        self.output.pack_forget()
        global final
        x = self.master.winfo_rootx() + self.c.winfo_x()
        y = self.master.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        ImageGrab.grab(bbox=(x,y,x1,y1)).save("demo_sketch/a.png")

        folder_selected = filedialog.askdirectory()
        folder_selected+="/"
        filepath = prep(folder_selected)

        checkpoint_s = torch.load(self.sketch_model+"10epoch.pth.tar")
        checkpoint_i = torch.load(self.image_model+"10epoch.pth.tar")
        transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])

        dset_test_i = Testdata(filepath, folder_selected, transformations)
        test_image_loader = DataLoader(dset_test_i,batch_size=max(len(dset_test_i),512),shuffle=True,num_workers=4,pin_memory=True)

        sketch_model = net.Net(self.hashcode_length).cuda()
        image_model = net.Net(self.hashcode_length).cuda()
        test_sketch = Image.open("demo_sketch/a.png").convert('RGB')
        input_sketch = transformations(test_sketch)
        sketch_model.load_state_dict(checkpoint_s['state_dict'])
        image_model.load_state_dict(checkpoint_i['state_dict'])
        print('here')

        top10 = exec_it(sketch_model,image_model,input_sketch,test_image_loader)
        final = []
        print(top10)
        final.extend(top10.tolist())


    def clear(self):
        self.c.delete(ALL)

    def change_fg(self):
        self.color_fg = colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):
        self.color_bg=colorchooser.askcolor(color=self.color_bg)[1]
        self.c['bg']=self.color_bg

    def res(self):
        global final
        for i in range(len(final)):
            im = Image.open(final[i])
            im = im.resize((150,150))
            tkimage = ImageTk.PhotoImage(im)
            handler = lambda img = final[i]: getFileName(img)
            imageButton = Button(self.output, image=tkimage, command=handler)
            imageButton.grid(row=i//5,column=i%5)
            imageButton.image=tkimage
        self.output.pack()

    def drawWidgets(self):
        global final
        self.controls = Frame(self.master,padx=5,pady=5)
        Label(self.controls, text='Pen Width: ',font=(15)).grid(row=0,column=0)
        self.slider = ttk.Scale(self.controls,from_=5, to=50,command=self.changeW,orient=HORIZONTAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0,column=1,ipadx=30)
        self.controls.pack()

        self.c = Canvas(self.master,width=750,height=500,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True)

        self.output = Frame(self.master,padx=5,pady=5)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        menu.add_cascade(label='File',menu=filemenu)
        filemenu.add_command(label='Export and Run',command=self.save_exec)
        opmenu = Menu(menu)
        menu.add_cascade(label='Show Results',menu=opmenu)
        opmenu.add_command(label='Results',command=self.res)
        colormenu = Menu(menu)
        menu.add_cascade(label='Colors',menu=colormenu)
        colormenu.add_command(label='Brush Color',command=self.change_fg)
        colormenu.add_command(label='Background Color',command=self.change_bg)
        optionmenu = Menu(menu)
        menu.add_cascade(label='Options',menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas',command=self.clear)
        optionmenu.add_command(label='Exit',command=self.master.destroy)


if __name__ == '__main__':
    args = Options().parse()
    root = Tk()
    root.title('SketchyApp')
    main(root,args)
    root.mainloop()
