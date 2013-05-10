from distutils.core import setup

setup(name='PODI',
      version='0.1',
      description='Program optimisation by dependency injection',
      author='James McDermott',
      author_email='jmmcd@jmmcd.net',
      url='https://www.github.com/jmmcd/PODI',
      packages=['PODI'],
      requires=['numpy', 'scipy', 'matplotlib', 'sklearn', 'pylab', 'zss', 'editdist', ]
)
