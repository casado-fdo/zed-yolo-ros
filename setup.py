from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

# fetch values from package.xml
setup_args = generate_distutils_setup(
    scripts=['scripts/detector.py'],
    packages=['cv_viewer', 'ogl_viewer'],
    package_dir={'': 'src'})

setup(**setup_args)
