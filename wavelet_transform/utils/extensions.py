import pywt
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar


class ScrollableWindow(QtWidgets.QMainWindow):
    def __init__(self, fig):
        self.qapp = QtWidgets.QApplication([])

        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0,0,0,0)
        self.widget.layout().setSpacing(0)

        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.scroll = QtWidgets.QScrollArea(self.widget)
        self.scroll.setWidget(self.canvas)

        self.nav = NavigationToolbar(self.canvas, self.widget)
        self.widget.layout().addWidget(self.nav)
        self.widget.layout().addWidget(self.scroll)

        self.show()
        exit(self.qapp.exec_())

def sci_form(number,dp = 3):
    'returns a string of the scientific form of a number'
    e_power = 0
    if number == 0:
        return f'{number}'
    while number >= 10:
        number /= 10
        e_power += 1
    while number < 1:
        number *= 10
        e_power -= 1
    return f'{round(number,dp)}e{e_power}'

def max_level_check(data, levels):
    'checks if there are too many levels'
    no_sections = int(2**levels)
    error_message = f'''too many levels. length of data is length of data is:\
    {len(data)}, this many levels requires {no_sections} data points
    '''
    if no_sections > len(data):
        raise ValueError(error_message)
    
def pywt_filter_bank(wavelet = 'haar'):
    'obtains the filter coefficients from the pywt filter bank function'
    w = pywt.Wavelet(wavelet)
    return w.filter_bank