import io
from setuptools import find_packages, setup

def long_description():
    with io.open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme

setup(name='hpa-simulator',
      version='0.1',
      description='kubernetes hpa simulator',
      long_description=long_description(),
      url='https://github.com/wkd3475/hpa-simulator',
      author='wkd3475',
      author_email='wkd3475@gmail.com',
      license='MIT',
      packages=find_packages(),
      classifiers=[
          'Programming Language :: Python :: 3',
          ],
      zip_safe=False)