from setuptools import setup

setup(name='tdlm-python',
      version='0.4.1',
      description='TDLM implementation for Python',
      long_description='Temporally delayed linear modelling quantifies sequential occurrences of events in time series and biosignals such as EEG or MEG',
      long_description_content_type="text/markdown",
      url='http://github.com/skjerns/TDLM-Python',
      author='skjerns',
      author_email='nomail@nomail.com',
      license='GPLv4',
      packages=['tdlm'],
      install_requires=['numpy', 'numba', 'seaborn'],
      zip_safe=False,
      classifiers=[
          "Programming Language :: Python :: 3",
          "Operating System :: OS Independent",
      ],)
