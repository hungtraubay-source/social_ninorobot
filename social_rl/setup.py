from glob import glob

from setuptools import find_packages, setup

package_name = 'social_rl'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'requirements.txt',
                                   'README.md', 'RUN_RL.txt']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ninorobot2',
    maintainer_email='user@example.com',
    description='Recurrent PPO (LSTM) person avoidance for linorobot2.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'train_rl = social_rl.train:main',
            'rl_agent = social_rl.agent_node:main',
            'zone_markers = social_rl.zone_markers:main',
        ],
    },
)
