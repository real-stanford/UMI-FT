"""Logs mocap pose during streaming"""

from util.mocap.mocap_node import MocapNode
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", type=str, required=True)
    parser.add_argument("--rigid_body_ids", type=int, nargs="+", required=True)
    args = parser.parse_args()
    
    agent = MocapNode(rigid_body_dict={id: "iphone_right" for id in args.rigid_body_ids}, ip=args.ip)

    while True:
        for id in args.rigid_body_ids:
            if agent.is_ready[id]:
                print('id', id, agent.trans[id], agent.quat_wxyz[id], agent.time[id])
