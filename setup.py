import os
import sys

import setuptools

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(__file__))

    import versioneer

    setuptools.setup(version=versioneer.get_version(),
                     cmdclass=versioneer.get_cmdclass())