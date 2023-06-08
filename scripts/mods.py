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

