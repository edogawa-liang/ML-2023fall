from setuptools import setup

APP=['model.py']
OPTIONS = {
    'iconfile':'python-file.png',
    'argv_emulation':True
}
setup(
    app=APP,
    options={'py2app':OPTIONS},
    setup_requires=['py2app']
)