#!/usr/bin/env python3
"""Own a removable pair of animated Gazebo actors for one terminal session."""

import signal
import threading

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Bool, Empty


def main(args=None):
    # Own signal handling so the hide command is published while the ROS
    # context is still valid. The default rclpy handler shuts the context down
    # first, which made the actors remain visible after Ctrl+C.
    context = Context()
    rclpy.init(
        args=args,
        context=context,
        signal_handler_options=SignalHandlerOptions.NO,
    )
    node = Node('release_animated_people', context=context)
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)

    # "talking"   : two actors stand face to face for the whole session.
    # "gathering" : they walk in, talk, walk out, and repeat, so the costmap
    #               has to both create and remove the social region.
    # "crossing"  : one actor repeatedly walks across the RGB-D camera.
    # "standing"  : one actor remains at the fixed bookshelf-test position.
    node.declare_parameter('scenario', 'talking')
    scenario = str(node.get_parameter('scenario').value).strip().lower()
    topics = {
        'talking': '/animated_people/release',
        'gathering': '/animated_people/gather',
        'crossing': '/animated_people/crossing',
        'standing': '/animated_people/standing',
    }
    if scenario not in topics:
        node.get_logger().error(
            f'Unknown scenario "{scenario}". Use one of: {", ".join(topics)}.')
        return
    release = node.create_publisher(Empty, topics[scenario], 1)
    hide = node.create_publisher(Empty, '/animated_people/hide', 1)

    # Waiting for perception means people appear only after the RGB-D/YOLO
    # pipeline can already see them, rather than standing unobserved during
    # camera or model initialization.
    node.declare_parameter('wait_for_perception', False)
    node.declare_parameter('ready_timeout', 600.0)
    wait_for_perception = bool(node.get_parameter('wait_for_perception').value)
    ready_timeout = float(node.get_parameter('ready_timeout').value)
    ready = {'received': False}

    def on_ready(message):
        ready['received'] = True
        node.get_logger().info(
            'Perception is ready. Releasing actors.')

    if wait_for_perception:
        node.create_subscription(
            Bool, '/social_perception/ready', on_ready,
            QoSProfile(
                depth=1,
                history=HistoryPolicy.KEEP_LAST,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL))
    stop_requested = threading.Event()

    def request_stop(_signum, _frame):
        stop_requested.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    if hasattr(signal, 'SIGHUP'):
        signal.signal(signal.SIGHUP, request_stop)

    released = False
    try:
        deadline = node.get_clock().now().nanoseconds + 10_000_000_000
        while (not stop_requested.is_set() and
               release.get_subscription_count() == 0 and
               node.get_clock().now().nanoseconds < deadline):
            executor.spin_once(timeout_sec=0.1)
        if stop_requested.is_set():
            return
        if release.get_subscription_count() == 0:
            node.get_logger().error('Gazebo animated-people plugin is unavailable.')
            return

        if wait_for_perception and not ready['received']:
            node.get_logger().info(
                'Waiting for /social_perception/ready before releasing actors '
                f'(up to {ready_timeout:.0f}s)...')
            # Time out rather than block forever: a missing GPU or a wrong
            # camera topic must not leave an empty world with no explanation.
            ready_deadline = (node.get_clock().now().nanoseconds +
                              int(ready_timeout * 1e9))
            while (not stop_requested.is_set() and not ready['received'] and
                   node.get_clock().now().nanoseconds < ready_deadline):
                executor.spin_once(timeout_sec=0.1)
            if stop_requested.is_set():
                return
            if not ready['received']:
                node.get_logger().warn(
                    'Perception never reported ready. Releasing the actors '
                    'anyway so the scene is not left empty.')

        # One command is intentional: the plugin ignores repeats and creates
        # real actors with looping talk.dae skeletal animations.
        release.publish(Empty())
        released = True
        node.get_logger().info(
            f'Spawned "{scenario}" scenario. Press Ctrl+C to remove the actors.')
        while context.ok() and not stop_requested.is_set():
            executor.spin_once(timeout_sec=0.2)
    except KeyboardInterrupt:
        stop_requested.set()
    finally:
        if released and context.ok():
            # Give DDS discovery and Gazebo's world-update queue time to
            # receive the removal command before this process exits.
            node.get_logger().info('Removing animated people...')
            for _ in range(5):
                hide.publish(Empty())
                executor.spin_once(timeout_sec=0.1)
            node.get_logger().info('Animated people removed.')
        executor.remove_node(node)
        executor.shutdown(timeout_sec=1.0)
        node.destroy_node()
        if context.ok():
            rclpy.shutdown(context=context)


if __name__ == '__main__':
    main()
