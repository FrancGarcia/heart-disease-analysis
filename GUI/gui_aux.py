'''
    Author:Chengming Li
    Date: 02/09/2025
    Purpose: For UCSD ECE143 Final project- GUI for data visualization
'''
import tkinter as tk
import matplotlib
import matplotlib.backends
import matplotlib.backends.backend_agg
import matplotlib.figure

'''----------------------------------------- For GUI Window -----------------------------------------'''
def create_window(title: str, geometry = '300x200'):
    '''
        title: window's name
        geometry: size of the window
    '''
    assert isinstance(title, str)
    assert isinstance(geometry, str), 'Geometry Format: 300x200'
    root = tk.Tk()
    root.title(title)
    root.geometry(geometry)
    return root

def run_gui(root):
    '''
        root: type: tkinter.Tk
    '''
    assert isinstance(root,tk.Tk), 'root type is tkinter.Tk'
    root.mainloop()

'''----------------------------------------- For GUI Label -----------------------------------------'''
def create_label(root, textin, fontin='Arial', font_size = 12, color = 'black'):
    '''
        root: type: tkinter.Tk
        textin: text of the label
        fontin: font of the text
        font_size: font size
        color: text color, str
    '''
    assert isinstance(root,tk.Tk), 'root type is tkinter.Tk'
    assert isinstance(textin,str), 'text type is str'
    assert isinstance(fontin,str), 'font type is str'
    assert isinstance(font_size,int), 'font_size type is int'
    assert isinstance(color,str), 'color type is str'
    label = tk.Label(root, text=textin, font=(fontin, font_size), fg=color)
    return label

def label_position_pack(label: tk.Label, x_pos, y_pos):
    '''
        label: tkinter.Label object
        x_pos: horizontal padding (space above and below the widget) to separate it from other elements.
        y_pos: vertical padding (space above and below the widget) to separate it from other elements.
    '''
    assert isinstance(label,tk.Label), 'label is tkinter.Label type'
    assert isinstance(x_pos, int),'padx is int type'
    assert isinstance(y_pos, int),'pady is int type'
    label.pack(padx = x_pos, pady = y_pos)

def label_position_place(label: tk.Label, x_pos, y_pos):
    '''
        label: tkinter.Label object
        x_pos: horizontal padding (space above and below the widget) to separate it from other elements.
        y_pos: vertical padding (space above and below the widget) to separate it from other elements.
    '''
    assert isinstance(label,tk.Label), 'label is tkinter.Label type'
    assert isinstance(x_pos, int),'padx is int type'
    assert isinstance(y_pos, int),'pady is int type'
    label.place(x = x_pos,  y = y_pos)

def change_labelText(label:tk.Label, textin):
    '''
        Function used to change the text content in label
        label: tkinter.Label object
        textin: new text of the label
    '''
    assert isinstance(label,tk.Label), 'label is tkinter.Label type'
    assert isinstance(textin,str), 'text type is str'
    label.config(text= textin)

'''----------------------------------------- For GUI Button -----------------------------------------'''
def create_button(root, textin, func):
    '''
        root: type: tkinter.Tk
        textin: text of the label
        func: function pointer
    '''
    assert isinstance(root,tk.Tk), 'root type is tkinter.Tk'
    assert isinstance(textin,str), 'text type is str'
    # assert isinstance(func,function), 'func type is a function pointer'
    button = tk.Button(root, text=textin, command= func)
    return button

def button_position_pack(button: tk.Button, x_pos, y_pos):
    '''
        label: tkinter.Button object
        x_pos: horizontal padding (space above and below the widget) to separate it from other elements.
        y_pos: vertical padding (space above and below the widget) to separate it from other elements.
    '''
    assert isinstance(button,tk.Button), 'label is tkinter.Label type'
    assert isinstance(x_pos, int),'padx is int type'
    assert isinstance(y_pos, int),'pady is int type'
    button.pack(padx = x_pos, pady = y_pos)

def button_position_place(button: tk.Button, x_pos, y_pos):
    '''
        label: tkinter.Button object
        x_pos: horizontal padding (space above and below the widget) to separate it from other elements.
        y_pos: vertical padding (space above and below the widget) to separate it from other elements.
    '''
    assert isinstance(button,tk.Button), 'label is tkinter.Label type'
    assert isinstance(x_pos, int),'padx is int type'
    assert isinstance(y_pos, int),'pady is int type'
    button.place(x = x_pos, y = y_pos)

'''----------------------------------------- For GUI Entry -----------------------------------------'''
def create_entry(root, width_in = 30, fontin = 'Arial', font_size = 12):
    '''
        root: type: tkinter.Tk
        fontin: font type, str
        font_size: font size, int
        width_in: width of entry box, int
    '''
    assert isinstance(root,tk.Tk), 'root type is tkinter.Tk'
    assert isinstance(width_in,int), 'width type is int'
    assert isinstance(fontin,str), 'text type is str'
    assert isinstance(font_size,int), 'font_size type is int'
    entry = tk.Entry(root, font=(fontin, font_size), width = width_in)
    return entry

def entry_position_pack(entry: tk.Entry, x_pos, y_pos):
    '''
        label: tkinter.Entry object
        x_pos: horizontal padding (space above and below the widget) to separate it from other elements.
        y_pos: vertical padding (space above and below the widget) to separate it from other elements.
    '''
    assert isinstance(entry,tk.Entry), 'label is tkinter.Entry type'
    assert isinstance(x_pos, int),'padx is int type'
    assert isinstance(y_pos, int),'pady is int type'
    entry.pack(padx = x_pos, pady = y_pos)

def entry_position_place(entry: tk.Entry, x_pos, y_pos):
    '''
        label: tkinter.Entry object
        x_pos: horizontal padding (space above and below the widget) to separate it from other elements.
        y_pos: vertical padding (space above and below the widget) to separate it from other elements.
    '''
    assert isinstance(entry,tk.Entry), 'label is tkinter.Entry type'
    assert isinstance(x_pos, int),'padx is int type'
    assert isinstance(y_pos, int),'pady is int type'
    entry.place(x = x_pos, y = y_pos)

def get_entryInput(entry:tk.Entry,label) -> str:
    '''
        entry: tkinter.Entry object
        return the input in the format of string
    '''
    assert isinstance(entry,tk.Entry), 'label is tkinter.Entry type'
    try:
        num = float(entry.get())
        label.pack_forget()
        return num
    except ValueError:
        change_labelText(label,'Input needed to be numerical')
        label.pack()
    
        

'''----------------------------------------- For GUI Plot -----------------------------------------'''
def create_figure(fig_size:tuple, dpiin = 100):
    '''
        fig_size: tuple, (width, height)
        dpiin: int, e.g. dpi=100, then the figure will be (width x 100, height x 100) pixels
    '''
    assert isinstance(fig_size,tuple),'fig_size needs to be a tuple'
    assert isinstance(dpiin, int),'pixels is int type'
    return matplotlib.figure.Figure(figsize= fig_size, dpi= dpiin)

def create_axis(fig:matplotlib.figure.Figure, axis:int):
    '''
        fig: matplotlib.figure.Figure type
        axis: int type. e.g. 111 means 1 row 1 column, first plot
    '''
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(axis, int)
    return fig.add_subplot(axis)

def ax_plot(ax:matplotlib.axes._axes.Axes, xdata, ydata):
    '''
        ax: matplotlib.axes._axes.Axes type
        xdata: data list on x-axis
        ydata: data list on y-axis
    '''
    assert isinstance(ax, matplotlib.axes._axes.Axes)
    assert isinstance(xdata, list)
    assert isinstance(ydata, list)
    ax.plot(xdata, ydata, marker="o", linestyle="-", color="b")

def ax_properties(ax:matplotlib.axes._axes.Axes, xlabel, ylabel, title):
    '''
        ax: matplotlib.axes._axes.Axes type
        xlabel: label shown on the x-axis
        ylabel: label shown on the y-axis
        title: title shown on the plot
    '''
    assert isinstance(ax, matplotlib.axes._axes.Axes)
    assert isinstance(xlabel, str)
    assert isinstance(ylabel, str)
    assert isinstance(title, str)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def create_canvas(fig, root):
    '''
        fig: matplotlib.figure.Figure
        root: root: type: tkinter.Tk
    '''
    assert isinstance(fig, matplotlib.figure.Figure)
    assert isinstance(root,tk.Tk), 'root type is tkinter.Tk'
    canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    return canvas

def canvas_position_pack(canvas, xpos = None, ypos = None):
    '''
        canvas: matplotlib.backends.backend_tkagg.FigureCanvasTkAgg type
        xpos: horizontal position of canvas
        ypos: vertical position of canvas
    '''
    assert isinstance(canvas, matplotlib.backends.backend_tkagg.FigureCanvasTkAgg)
    assert isinstance(xpos, int)
    assert isinstance(ypos, int)
    canvas.get_tk_widget().pack(padx =xpos, pady = ypos)