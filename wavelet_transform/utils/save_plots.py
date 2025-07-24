# from matplotlib.backends.backend_pdf import PdfPages
import os
import plotly.graph_objects as go
import matplotlib.pyplot as plt

FILE_PATH = os.path.abspath(__file__)
RELATIVE_PATH = "../../../notebooks/figures"
FOLDER_PATH = os.path.abspath(os.path.join(FILE_PATH, RELATIVE_PATH))


def save_figure(figure, file_name, file_path=None):
    """
    Save a figure with a file_name to the figures folder in notebooks.

    matplotlib plots are saved as pdfs, plotly plots are saved using html.
    """

    if isinstance(figure, plt.Figure):
        if not file_path:
            file_path = get_path(file_name, ".pdf")
        else:
            file_path = file_path + "\\" + file_name + ".pdf"
        figure.savefig(file_path, format="pdf")

    elif isinstance(figure, go.Figure):
        if not isinstance(file_path, str):
            file_path_pdf = get_path(file_name, ".pdf")
            # file_path_html = get_path(file_name, ".html")
        else:
            file_path_pdf = file_path + "\\" + file_name + ".pdf"
            # file_path_html = file_path + "\\" + file_name + ".html"
        # figure.write_html(file_path_html)
        figure.write_image(file_path_pdf, format="pdf")

    else:
        raise ValueError("figure has to be a plt.Figure or go.Figure")


def get_path(file_name, file_type):
    if not isinstance(file_name, str):
        raise ValueError("file_name must be a string")
    file_name = file_name + file_type
    file_path = os.path.join(FOLDER_PATH, file_name)
    return file_path
