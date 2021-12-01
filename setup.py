from setuptools import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['multi_cam_calib_py'],
    package_dir={'': 'src'}
)

setup(**d)