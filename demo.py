from tkinter import *
from tkinter import ttk,colorchooser,filedialog
from tkinter.ttk import Progressbar
import pyscreenshot as ImageGrab
from PIL import Image,ImageDraw
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.spatial.distance import cdist

import numpy as np
from options import Options
from Models import net
from Datasets.load_SketchImagepairs import SketchImageDataset,Datapoints

def exec_it(smod,imod,input,ret_pool):
    all_image_hashes = []
    img_labels=[]
    img_paths=[]
    for batch_idx,(data_i,label,path) in enumerate(ret_pool):
        data_i = data_i.cuda()
        fx,_,blah = imod(data_i-0.5)
        fxd = fx.cpu().detach().numpy()
        imageb_hash = (np.sign(fxd)+1)/2
        # imageb_hash = fxd
        all_image_hashes.extend(imageb_hash)
        img_labels.extend(label)
        img_paths.extend(path)

    img_paths = np.array(img_paths)
    sketch_hash = (np.sign(smod(input.unsqueeze(0).cuda()-0.5)[0].cpu().detach().numpy())+1)/2
    hamm_d = cdist(sketch_hash, all_image_hashes, 'hamming')

    top10 = hamm_d[0].argsort()[-10:][::-1]
    retrieved_imgs = img_paths[top10]

    return retrieved_imgs

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
        #file = filedialog.asksaveasfilename(filetypes=[("Sketches","*.png")])
        #   if file:
        x = self.master.winfo_rootx() + self.c.winfo_x()
        y = self.master.winfo_rooty() + self.c.winfo_y()
        x1 = x + self.c.winfo_width()
        y1 = y + self.c.winfo_height()
        ImageGrab.grab(bbox=(x,y,x1,y1)).save("demo_sketch/a.png")

        checkpoint_s = torch.load(self.sketch_model+"1epoch.pth.tar")
        checkpoint_i = torch.load(self.image_model+"1epoch.pth.tar")
        transformations = transforms.Compose([transforms.Resize(224),transforms.ToTensor()])
        dset_test_i = Datapoints(self.img_path + "filelist-test.txt", args.img_path, transformations)
        test_image_loader = DataLoader(dset_test_i,batch_size=512,shuffle=True,num_workers=4,pin_memory=True)
        sketch_model = net.Net(self.hashcode_length).cuda()
        image_model = net.Net(self.hashcode_length).cuda()
        test_sketch = Image.open("demo_sketch/a.png").convert('RGB')
        input_sketch = transformations(test_sketch)
        sketch_model.load_state_dict(checkpoint_s['state_dict'])
        image_model.load_state_dict(checkpoint_i['state_dict'])
        print('here')

        top10 = exec_it(sketch_model,image_model,input_sketch,test_image_loader)

        print(top10)


    def clear(self):
        self.c.delete(ALL)

    def change_fg(self):
        self.color_fg = colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):
        self.color_bg=colorchooser.askcolor(color=self.color_bg)[1]
        self.c['bg']=self.color_bg

    def drawWidgets(self):
        self.controls = Frame(self.master,padx=5,pady=5)
        Label(self.controls, text='Pen Width: ',font=(15)).grid(row=0,column=0)
        self.slider = ttk.Scale(self.controls,from_=5, to=50,command=self.changeW,orient=HORIZONTAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0,column=1,ipadx=30)
        self.controls.pack()

        self.c = Canvas(self.master,width=750,height=750,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True)

        menu = Menu(self.master)
        self.master.config(menu=menu)
        filemenu = Menu(menu)
        menu.add_cascade(label='File',menu=filemenu)
        filemenu.add_command(label='Export and Run',command=self.save_exec)
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
    main(root,args)
    root.title('SketchyApp')
    root.mainloop()
