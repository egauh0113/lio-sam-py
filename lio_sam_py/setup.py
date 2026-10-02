from setuptools import find_packages, setup
import os
from glob import glob

package_name = "lio_sam_py"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Eugene Auh",
    maintainer_email="egauh@g.skku.edu",
    description="LIO-SAM in Python",
    license="BSD-3-Clause",
    extras_require={
        "test": [
            "pytest",
        ],
    },
    entry_points={
        "console_scripts": [
            "feature_extraction = lio_sam_py.feature_extraction:main",
            "image_projection = lio_sam_py.image_projection:main",
            "imu_preintegration = lio_sam_py.imu_preintegration:main",
            "map_optimization = lio_sam_py.map_optimization:main",
        ],
    },
)
