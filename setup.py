from setuptools import setup, find_packages

setup(
    name="subclassificator",
    version='0.1',
    packages=['src', 'src.preprocessor', 'src.data', 'src.evaluation',
              'src.manage', 'src.models','src.data.model'],
    package_data={'src':['config.json','evaluation/results.json','models/*.h5'], 'src/data':['model/*','preprocessed/*', 'raw/*'],
                  'src/preprocessor':['data/*.txt']},
    include_package_data=True,
)

