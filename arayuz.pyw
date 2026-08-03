"""Double-click this file on Windows to open the local web interface."""

from multiprocessing import freeze_support

from turkish_markov.webgui import main


if __name__ == "__main__":
    freeze_support()
    main()
