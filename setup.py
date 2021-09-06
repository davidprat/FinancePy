import setuptools

INSTALL_REQUIRES = [
    'numba==0.49'
    'numpy==1.19.2'
    'scipy==1.5.2'
]

setuptools.setup(
    name="financepy",
    version="0.200",
    author="Dominic O'Kane",
    author_email="quant@financepy.com",
    description="A Finance Library",
    long_description='long_description',
    long_description_content_type="text/markdown",
    url="https://github.com/domokane/FinancePy",
    keywords=['FINANCE', 'OPTIONS', 'BONDS', 'VALUATION', 'DERIVATIVES'],
    install_requires=INSTALL_REQUIRES,
    package_data={'': ['*.npz'], },
    include_package_date=True,
    packages=setuptools.find_packages(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
