from setuptools import find_packages, setup

package_name = 'robthepupper'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='anand',
    maintainer_email='ank029@ucsd.edu',
    description='Rob The Pupper: An interactive puzzle solving game with mini-pupper bot.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['gui_service = lab2.gui_pupper_service:main',
                            'gui_client_ssh = lab2.gui_pupper_client_ssh:main',
                            'saveImg = lab2.saveImg:main',
        ],
    },
)

