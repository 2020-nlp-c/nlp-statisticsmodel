from pathlib2 import Path
import inspect
import importlib


def run_files(pattern, path = ".") :
    ls = []
    for f in Path(path).rglob(pattern) :
        user = f._parts[0]
        filename = f.name

        f='print1'

        mod = __import__('mhlee')
        inspect.getmembers(mod, inspect.isclass)

        #runpy.run_path(file_path=f)
        print(user)
        print(filename)


if __name__ == "__main__" :
    run_files('print1.py')



