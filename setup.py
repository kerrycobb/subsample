import setuptools as st

st.setup(
    name='Subsample',
    version='1.0',
    description='Obtain dispersed subsample for a sample of points',
    url='http://github.com/kerrycobb/subsample',
    author='Kerry A. Cobb',
    author_email='cobbkerry@gmail.com',
    license='GPLv3',
    packages=st.find_packages(),
    install_requires=[
        'pandas',
        'networkx',
        'numpy',
        'scipy',
        'networkx',
        'matplotlib',

    ],
    # entry_points={
    #     'console_scripts':[
    #         'subsample=subsample.cli:cli'
    #     ]
    # },
)
