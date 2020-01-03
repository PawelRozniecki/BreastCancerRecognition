from tkinter import *
from tkinter import filedialog

from PIL import ImageTk,Image

window = Tk()
window.title("Breast Cancer Recognition")
window.geometry("512x512")

canvas = Canvas(window, width = 300, height = 300)

filename = None
def openFileChooser() :

    global filename
    filename = filedialog.askopenfilename(initialdir="/", title="Select file",
                                                          filetypes=[('all files', '.*'),
                                                                     ('text files', '.txt'),
                                                                     ('image files', '.png'),
                                                                     ('image files', '.jpg'),
                                                                     ])


fileBtn = Button(window, text="choose file", command=openFileChooser)
print(filename)
im = Image.open(filename)
fileBtn.pack()
window.mainloop()
