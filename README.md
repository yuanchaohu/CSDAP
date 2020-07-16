**Computer Simulation Data Analysis Package (CSDAP)**

This package is designed for who is interested in analyzing the snapshots from molecular dynamics simulations, i.e. by [LAMMPS](http://lammps.sandia.gov/) , [Hoomd-blue](http://glotzerlab.engin.umich.edu/hoomd-blue/) et al. It is flexible for other computer simulations as long as you change the method of reading coordinates to suitable formats in 'dump.py'. The modules in the package are written in [Python3](https://www.python.org/) by importing some high-efficiency modules like [Numpy](http://www.numpy.org/) and [Pandas](http://pandas.pydata.org/). I strongly recommend the user to install [Anaconda3](https://www.anaconda.com/download/) or/and [VS Code](https://code.visualstudio.com/), or [Sublime Text 3](https://www.sublimetext.com/).

To use the package efficiently, one intelligent way is to write a python script by importing desired modules and functions. In this way, all results can be obtained in sequence with suitable settings.

The package is distributed in the hope that it will be helpful, but WITHOUT ANY WARRANTY. Although the codes were benchmarked when developed, the users are responsible for the correctness.

A detailed manual is provided in the corresponding [Github-Wiki](https://github.com/yuanchaohu/CSDAP/wiki). To modifty the math written by Latex, the editor typora is recommended and the extension 'MathJax Plugin for Github' for Google Chrome is required to show the equations correctly online.

A pdf version of the manual is also available from doc/.