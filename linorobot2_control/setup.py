from setuptools import find_packages, setup

package_name = 'linorobot2_control'

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
    maintainer='heithhara',
    maintainer_email='mrphucdu2002@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simple_follower = linorobot2_control.simple_follower:main',
            'simple_follower_motion_law = linorobot2_control.simple_follower_motion_law:main',
            'simple_follower_motion_law_rw = linorobot2_control.simple_follower_motion_law_rw:main',
            'simple_follower_motion_law_av = linorobot2_control.simple_follower_motion_law_av:main',
            'simple_follower_motion_law_av_rw = linorobot2_control.simple_follower_motion_law_av_rw:main',
            'fbl_follower = linorobot2_control.feedback_lin_follower:main',
            'fbl_follower_motion_law = linorobot2_control.feedback_lin_follower_motion_law:main',
            'fbl_follower_motion_law_rw = linorobot2_control.feedback_lin_follower_motion_law_rw:main',
            'fbl_follower_motion_law_av = linorobot2_control.feedback_lin_follower_motion_law_av:main',
            'fbl_follower_motion_law_av_rw = linorobot2_control.feedback_lin_follower_motion_law_av_rw:main',
            'fbl_follower_motion_law_av_po = linorobot2_control.feedback_lin_follower_motion_law_av_po:main',
            'fbl_follower_motion_law_av_po_rw = linorobot2_control.feedback_lin_follower_motion_law_av_po_rw:main',
            'lyapunov_follower = linorobot2_control.lyapunov_follower:main',
            'lyapunov_follower_motion_law = linorobot2_control.lyapunov_follower_motion_law:main',
            'lyapunov_follower_motion_law_rw = linorobot2_control.lyapunov_follower_motion_law_rw:main',
            'lyapunov_follower_motion_law_av = linorobot2_control.lyapunov_follower_motion_law_av:main',
            'lyapunov_follower_motion_law_av_rw = linorobot2_control.lyapunov_follower_motion_law_av_rw:main',
            'mpc_ltv_follower = linorobot2_control.mpc_ltv_follower:main',
            'mpc_ltv_follower_motion_law = linorobot2_control.mpc_ltv_follower_motion_law:main',
            'mpc_ltv_follower_motion_law_rw = linorobot2_control.mpc_ltv_follower_motion_law_rw:main',
            'mpc_ltv_follower_motion_law_av = linorobot2_control.mpc_ltv_follower_motion_law_av:main',
            'mpc_ltv_follower_motion_law_av_rw = linorobot2_control.mpc_ltv_follower_motion_law_av_rw:main',
            'ha2=linorobot2_control.ha2:main',
            'hareal=linorobot2_control.hareal:main',
            'hanav1=linorobot2_control.hanav1:main',
        ],
    },
)
