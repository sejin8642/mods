# personal python functions and classes

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from functools import partial

from ipywidgets import Checkbox, HBox, FloatSlider, Layout, interactive_output

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
    Returns a string of number represented with scientific prefixes. The precision is between 2 and 4
    significant figures. For example, if 123,400 is provided, 123.4k as string is returned 
    for numbers less than 4 precision, zeros will be padded at the beginning of the number
    
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
        0:'',
        1:'k',  2:'M',  3:'G',  4:'T',  5:'P',  6:'E',  7:'Z',  8:'Y',  9:'R',  10:'Q',
        -1:'m', -2:'\u03BC', -3:'n', -4:'p', -5:'f', -6:'a', -7:'z', -8:'y', -9:'r', -10:'q' }

    digits = float(mantissa)*pow(10, int(exponent)%3)
    prefix = units.get(int(exponent)//3, None)
    decimal_place = 1 if prefix != '' else 2 
    num = '{:.1f}'.format(round(digits, decimal_place)).zfill(4 + decimal_place)

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

def graph(
        x, 
        y, 
        xlabel=r'$x$', 
        ylabel=r'$y$', 
        xtick_locations = [], 
        xtick_labels = [], 
        ytick_locations = [], 
        ytick_labels = [], 
        save=False, 
        filename='figure.eps',
        show=True):
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
    xtick_locations: list[float]
        horizontal locations of ticks. Default is empty list.
    xtick_labels: list[str]
        labels for xtick locations. There should be exactly the same number of labels as locations
    ytick_locations: list[float]
        vertical locations of ticks. Default is empty list.
    xtick_labels: list[str]
        labels for ytick locations. There should be exactly the same number of labels as locations
    save: bool
        If True, the plot will be saved as esp file. EPS is a file format used for vector graphics 
        that is based on the PostScript language.
    filename: str
        filename for eps
    show: bool
        show executes plt.show() inside graph function if True. Set it to false if you want to add
        more plot options after graph execution
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

    # tick locations with labels
    plt.xticks(xtick_locations, xtick_labels)
    plt.yticks(ytick_locations, ytick_labels)

    # save the figure as eps vector image if save==True
    if (save == True):
        plt.savefig(filename, format='eps', transparent=True)

    # show the plot
    if show: plt.show()

def interact_plot(
    x,
    Y,
    names=None,
    x_scale=1, 
    y_scale=1, 
    x_position=0,
    y_position=0,
    grid=False,
    x_size=800,
    y_size=600,
    **Y_dict):
    """
    function to plot multiple graphs, and it allows to change the scale and position of 
    a part of the graph you are looking at. With ipywidgets.interact, this allows
    to dynamically change the position and scale of the part of the graph to inspect the
    data easily. First, from ipywidgets import interact. Then from functools import partial.

    Examples
    --------
    x = np.linspace(-5.0, 5.0, 101)
    sin = np.sin(x)
    cos = np.cos(x)

    # from functools import partial first
    trigs = partial(interact_plot, x, [sin, cos])

    # name the function to avoid AttributeError:
    trigs.__name__ = 'interactive function'

    # from ipywidgets import interact first
    interact(
        trigs,
        x_scale=(0.01, 1, 0.01), 
        y_scale=(0.01, 1, 0.01), 
        x_position=(-1, 1, 0.02),
        y_position=(-1, 1, 0.02),
        grid=True,
        None
    )
    
    Parameter
    ---------
    x: numpy array[float]
        Domain of the graph
    y: numpy array[float]
        Range of the graph
    names: list[str]
        list of name strings for each Y (default None)
    x_scale: float
        Scale of the graph along x-axis to be viewed. The value is between 0 and 1 (default 1)
    y_scale: float
        Scale of the graph along y-axis to be viewed. The value is between 0 and 1 (default 1)
    x_position: float
        Horizontal position of graph view with respect to the center of the graph. The value is 
        between -1 and 1 (default 0)
    y_position: float
        Vertical position of graph view with respect to the center of the graph. The value is 
        between -1 and 1 (default 0)
    grid: bool
        If true, grid is shown on the graph (default False)
    figure_size: (int, int)
        figure size in pixels in horizontal and vertical axes (default (800, 600))
    **Y_dict: dict[name:bool]
        dictionary containing name strings and bools for each Y
    """
    # figure size setting
    dpi = 96  # Adjust based on your needs
    figure_width_inches = x_size / dpi
    figure_height_inches = y_size / dpi
    plt.figure(figsize=(figure_width_inches, figure_height_inches))

    # initialization of scale, min, max for axes and gridlines
    x_scale = np.clip(x_scale, 0, 1.2)
    y_scale = np.clip(y_scale, 0, 1.2)
    x_min = x[0]
    x_max = x[-1]
    y_min = np.min(Y)
    y_max = np.max(Y)
    y_height = y_max - y_min
    y_min = y_min - 0.05*y_height 
    y_max = y_max + 0.05*y_height 
    plt.grid(grid)

    number_Y = len(Y) # number of total possible plots

    # generate grid points and plot toggled graphs
    grid_points = grid_sequence(number_Y)
    for i, (key, item) in enumerate(Y_dict.items()):
        if item == True: 
            plt.plot(x, Y[i], color=grid_points[i], label=key)
    
    x_mid = (x_max + x_min)/2.0
    y_mid = (y_max + y_min)/2.0
    x_size_half = (x_max - x_min)/2.0
    y_size_half = (y_max - y_min)/2.0

    xi = x_mid - x_size_half*x_scale
    xf = x_mid + x_size_half*x_scale
    yi = y_mid - y_size_half*y_scale
    yf = y_mid + y_size_half*y_scale
    
    x_diff = x_max - xf
    y_diff = y_max - yf
    
    x_min = xi + x_diff*x_position
    x_max = xf + x_diff*x_position
    y_min = yi + y_diff*y_position
    y_max = yf + y_diff*y_position

    plt.axis([x_min, x_max, y_min, y_max])    
    plt.show()

def print_fn(*args, num=10):
    """
    Prints the input objects up to the line num. It is useful to limit the output length of print
    in order to see multiple prints all at once on Jupyter lab.
    
    Parameter
    ---------
    args: tuple
        Tuple of multiple input objects
    num: int
        Line number of strings to truncate after (default 10)
    """
    # get each object to print
    for arg in args:
        # initialize first index, end string, and object string
        ind = 0
        end_str = ' ............'
        arg_str = arg.__str__()

        # iterate num times to find num-th index of '\n' for arg_str
        for count in range(num):
            new_ind = arg_str[ind:].find('\n')

            # stop iterating if there are fewer than num lines in the string
            if new_ind == -1:
                ind = new_ind
                end_str = ''
                break

            # update the index until it reaches num-th line
            ind = new_ind + ind + 2
            
        # print the string until num-th line
        print(arg_str[:ind-2] + end_str + '\n\n')

def grid_sequence(num_points):
    """
    Generate grid sequence of RGB space from a positive integer such that the number of
    grid points are greater than or equal to the input number. 

    parameter
    ---------
    num_points: int
        minimum number of grid points

    return
    ------
    grid_points: numpy array
        numpy array that contains grid points for RGB coloring
    """
    # pick three base numbers for grid points distribution
    base1 = base2 = base3 = np.floor(num_points**(1/3))
    if base1*base2*base3 < num_points:
        base1 += 1
    if base1*base2*base3 < num_points:
        base2 += 1
    if base1*base2*base3 < num_points:
        base3 += 1

    # from number of grid points, generate grid_points
    num_points_per_dim = [int(base1), int(base2), int(base3)]
    grid_fn = partial(np.linspace, 0.0, 1.0, endpoint=True)
    intervals = [grid_fn(num_points) for num_points in num_points_per_dim]
    grid_points = np.meshgrid(*intervals, indexing='ij')
    grid_points = np.stack(grid_points, axis=-1).reshape(-1, 3)
    return grid_points

def interactive_graph(x, Y, Y_names=None):
    """
    Takes x-axis domain as a numpy array and a list of y-axis ranges to plot. Additionally
    you can also provide names for each y axis graph. This graph fn allows to plot multiple
    graphs and change the scale and position of the view. You can also check/uncheck a plot
    to appear on the view
   
    Example
    -------
    x = np.linspace(0, 2*np.pi, 101) # x domain
    RD = np.random.random # numpy random generator fn
    N = 4 # number of plots
    Y = [RD()*np.sin(4*RD()*x + 2*np.pi*RD()) for _ in range(N)] # N graphs
    Y_names = ['first', 'second', 'third', 'forth'] # names for each y graph
    mods.interactive_graph(x, Y, Y_names=Y_names)

    Parameters
    ----------
    x: numpy array
        numpy array of x domain for graph
    Y: list[numpy array]
        list of numpy arrays
    Y_names: list[str]
        list of y graph name strings
    """
    # sliders for scale and position of the plot view and figure size
    x_scale=FloatSlider(
        value=1,
        min=0.01,
        max=1.0,
        step=0.01,
        description="x scale")
    y_scale=FloatSlider(
        value=1,
        min=0.01,
        max=1.0,
        step=0.01,
        description="y scale")
    x_position=FloatSlider(
        value=0,
        min=-1,
        max=1,
        step=0.02,
        description="x pos")
    y_position=FloatSlider(
        value=0,
        min=-1,
        max=1,
        step=0.02,
        description="y pos")
    x_size=FloatSlider(
        value=800,
        min=0,
        max=1600,
        step=1,
        description="x size")
    y_size=FloatSlider(
        value=600,
        min=0,
        max=1200,
        step=1,
        description="y size")

    # to make user interface for scale and position sliders
    ui1 = HBox([x_scale, y_scale])
    ui2 = HBox([x_position, y_position])

    # Checkbox method shortened
    CB = partial(
        Checkbox,
        value=True,
        indent=False,
        layout=Layout(width='100px', height='50px'))

    # first check box (gridlines)
    grid = CB(description=f'grid') 

    # create a function to return Y graphs names
    if Y_names == None:
        names = lambda i: f'plot {i}'
    else:
        names = lambda i: Y_names[i]

    # check boxes and descpription dict for Y graphs
    N = len(Y)
    cbs = [CB(description=names(i)) for i in range(N)]
    Y_dict = {cb.description: cb for cb in cbs}
    ui3 = HBox([grid, *cbs])
    ui4 = HBox([x_size, y_size])

    # from functools import partial first
    plot_fn = partial(interact_plot, x, Y)

    # display interactive plot
    inputs = {
        'x_scale': x_scale,
        'y_scale': y_scale,
        'x_position': x_position,
        'y_position': y_position,
        'grid': grid,
        'x_size': x_size,
        'y_size': y_size,
        **Y_dict
    }
    out = interactive_output(plot_fn, inputs)
    display(ui1, ui2, out, ui3, ui4)
