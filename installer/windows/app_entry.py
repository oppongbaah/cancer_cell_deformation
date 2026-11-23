"""The program the installer ships: ``celldform.exe`` starts here."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from celldform.demo.gui import main
    main()
