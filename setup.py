from setuptools import setup

setup(
    name="illm",  # 项目名称
    version="0.1",
    packages=["illm"],  # 包的名称
    install_requires=[
        'fastapi',
        'uvicorn',
        'pydantic',
        'requests',
    ],
    entry_points={
        "console_scripts": [
            "illm = illm.server:run",
        ],
    },
)
