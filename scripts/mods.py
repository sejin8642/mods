# personal python functions and classes

import numpy as np
import matplotlib.pyplot as plt

class FUNCS:
    """
    class that returns funcs instance that returns list of custom-defined functions
    
    Attribute
    ---------
    num: int
        number variable used to keep track of total number of custom functions
        
    """
    def __init__(self):
        self.num = 0
                
    def __repr__(self):
        """
        prints custom-defined functions
        """
        self.num = 0
        items = globals().items()
        for key, value in items:
            if callable(value) and value.__module__ == __name__:
                self.num += 1
                print(key)
        return str(f"total number of custom objects is {self.num}")
    
def matd(matrix):
    """
    displays numpy array as image of matrix (without brackets) for easy view
    
    parameter
    ---------
    matrix: numpy array[float or int]
        numpy array of float or int
    """
    # set configurations
    zeros = np.zeros(matrix.shape)
    rescale_factor = 3/5
    figsize = (rescale_factor*matrix.shape[1], rescale_factor*matrix.shape[0])
    matrix = np.flip(matrix, axis=0)
    ext1 = matrix.shape[1] - 0.9
    ext2 = matrix.shape[0] - 0.9

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Set the title and remove the axes ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Show the matrix as an image with numbers
    ax.imshow(zeros, cmap=plt.cm.get_cmap('gray').reversed(), extent=[0, ext1, 0, ext2])
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, f'{value:.0f}', ha='center', va='center', fontsize=16)

    # get rid of bounding box lines around the image
    ax.spines['top'].set_linewidth(0)
    ax.spines['bottom'].set_linewidth(0)
    ax.spines['left'].set_linewidth(0)
    ax.spines['right'].set_linewidth(0)

    # Set the face color of the figure to white
    fig.patch.set_facecolor('white')

    # Show the figure
    plt.show()
    
def examines(instance):
    """
    examines the input type and its chain of bases all the way to the top level
    """
    display(instance)
    typ = type(instance)
    while typ != object:
        display(typ)
        typ = typ.__base__
    print()

def check(iterable, m=5):
    """
    checks the first 5 items of the iterable (first input). You can supply the second input 
    to change the number of items
    """
    print("checking....")
    n = 0 
    for item in iterable:
        if n < m:
            print(item)
            print()
        n += 1
    print(f"iterator size is {n}")
    print()

def float2SI(number):
    """
    Returns a string of number represented with scientific prefixes. The precision is between 1 and 3
    significant figures. For example, if 123,000 is provided, 123k as string is returned 
    
    Parameter
    ---------
    number: float or int
        The original number as float or int

    Return
    ------
    num+prefix: string
        Number string represented with scientific prefix
    """
    mantissa, exponent = f"{number:e}".split("e")
    units = {
        0:' ',
        1:'k',  2:'M',  3:'G',  4:'T',  5:'P',  6:'E',  7:'Z',  8:'Y',  9:'R',  10:'Q',
        -1:'m', -2:'\u03BC', -3:'n', -4:'p', -5:'f', -6:'a', -7:'z', -8:'y', -9:'r', -10:'q' }

    digits = float(mantissa)*pow(10, int(exponent)%3)
    num = '{:>{}}'.format(str(round(digits)), 3)
    prefix = units.get(int(exponent)//3, None)

    return num+prefix

def print_formatted(*args):
    """
    Same as print function, except it takes numbers as arguments and float2SI is used to each number
    arguments for easier inspection

    Parameter
    args: tuple[float or int]
        Tuple of numbers to change the format 
    """
    formatted_args = tuple(map(float2SI, args))
    print(*formatted_args, sep='    ')

def graph(x, y, xlabel=r'$x$', ylabel=r'$y$', save=False, filename='figure.eps'):
    """
    plotting function without fancy coloring designed for LaTeX pdf file. You can save the plot as
    eps file, which you can insert in LaTeX document.

    Parameters
    ----------
    x: list[float]
        x-axis values
    y: list[float]
        y-axis values
    xlabel: str
        lable for x-axis. The default string is raw string. In Python, an r string, also known as 
        a raw string, is a string literal prefixed with the letter r. It is used to create strings 
        that treat backslashes (\) as literal characters, rather than escape characters.
    ylabel: str
        same as xlabel except for y-axis
    save: bool
        If True, the plot will be saved as esp file. EPS is a file format used for vector graphics 
        that is based on the PostScript language.
    filename: str
        filename for eps
    """
    # LaTeX font with size 9
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": 'serif',
        "font.size": 9})

    # plots y vs. x in black line with linesize 2 with the given axes
    fig = plt.figure(figsize=(6,4), dpi=500)
    ax = fig.add_subplot(111)

    # minimums and maximums of x and y
    xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
    print(xmin, xmax, ymin, ymax, sep='    ')

    # reset minimum and maximum of y if y-range does not contain 0
    if 0 < ymin: ymin = -0.1*ymax
    if ymax < 0: ymax = -0.1*ymin

    # axis label coordinate adjustments
    x_pos_for_y_label = -xmin/(xmax - xmin)
    y_pos_for_x_label = -ymin/(ymax - ymin) + 0.02

    # configures plot axes, labels and their positions with arrow axis tips
    if (xmin <= 0) and (0 < xmax):
        ax.spines['left'].set_position(('data', 0))
        ax.set_ylabel(ylabel, rotation=0)
        ax.yaxis.set_label_coords(x_pos_for_y_label, 1.02)
        ax.plot(0, 1, "^k", markersize=3, transform=ax.get_xaxis_transform(), clip_on=False)
    else:
        ax.spines['left'].set_visible(False)
        ax.set_ylabel(ylabel).set_visible(False)

    ax.spines['bottom'].set_position(('data', 0))
    ax.set_xlabel(xlabel)
    ax.xaxis.set_label_coords(1.02, y_pos_for_x_label)
    ax.plot(1, 0, ">k", markersize=3, transform=ax.get_yaxis_transform(), clip_on=False)

    # plots y vs. x in black line with linesize 2 with the given axes
    plt.plot(x, y, 'k-', linewidth=.5)
    plt.axis([xmin, xmax, 1.1*ymin, 1.1*ymax])

    # change the spine linewidth
    plt.rcParams['axes.linewidth'] = 0.2

    # deletes top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # changes the size of ticks (both major and minor) to zero if ticks==False
    ax.tick_params(axis=u'both', which=u'both', length=0)

    # no tick labels
    plt.xticks([])
    plt.yticks([])

    # save the figure as eps vector image if save==True
    if (save == True):
        plt.savefig(filename, format='eps', transparent=True)

    # show the plot
    plt.show()
