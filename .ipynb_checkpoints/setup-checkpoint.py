import os
import sys
from setuptools import setup, find_packages


package_basename = 'folps'
package_dir = os.path.join(os.path.dirname(__file__), package_basename)
sys.path.insert(0, package_dir)
import _version
version = _version.__version__


setup(name=package_basename,
      version=version,
      author='Hernan E. Noriega',
      author_email='',
      description='FOLPS code (includes JAX)',
      license='BSD3',
      url='http://github.com/cosmodesi/folpsax',
      install_requires=['numpy', 'scipy', 'jax', 'interpax'],
      extras_require={},
      packages=find_packages())
