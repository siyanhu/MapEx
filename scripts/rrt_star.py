"""

Path planning Sample Code with RRT*

author: Atsushi Sakai(@Atsushi_twi)

# From UPEN https://raw.githubusercontent.com/ggeorgak11/UPEN/master/planning/rrt_star.py
"""

import math
import os
import sys
import cv2
import numpy as np

#import matplotlib.pyplot as plt


from rrt import RRT

show_animation = False#True


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y):
            super().__init__(x, y)
            self.cost = 0.0

    def __init__(self,
                 start,
                 goal,
                 obstacle_list,
                 rand_area,
                 expand_dis=30.0,
                 path_resolution=1.0,
                 goal_sample_rate=50,
                 max_iter=600,
                 connect_circle_dist=50.0,
                 search_until_max_iter=False,
                 occupancy_map=None,
                 image=None):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        super().__init__(start, goal, obstacle_list, rand_area, expand_dis,
                         path_resolution, goal_sample_rate, max_iter, occupancy_map, image)
        self.connect_circle_dist = connect_circle_dist
        self.goal_node = self.Node(goal[0], goal[1])
        self.search_until_max_iter = search_until_max_iter

    def planning(self, animation=True, use_straight_line=False, exploration=False, horizon=10):
        """
        rrt star path planning

        animation: flag for animation on or off .
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            # if i % 500 == 0:
            #     print("Iter:", i, ", number of nodes:", len(self.node_list))
            #print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd)
            new_node = self.steer(self.node_list[nearest_ind], rnd,
                                  self.expand_dis)
            near_node = self.node_list[nearest_ind]
            new_node.cost = near_node.cost + \
                math.hypot(new_node.x-near_node.x,
                           new_node.y-near_node.y)

            #if self.check_collision(new_node, self.obstacle_list):
            if self.check_collision_map(new_node, self.occupancy_map):
                near_inds = self.find_near_nodes(new_node)
                node_with_updated_parent = self.choose_parent(
                    new_node, near_inds)
                if node_with_updated_parent:
                    self.rewire(node_with_updated_parent, near_inds)
                    self.node_list.append(node_with_updated_parent)
                else:
                    self.node_list.append(new_node)

            if not exploration:
                if i % 20 == 0 and use_straight_line:
                    goalnode = self.straight_line_to_goal(new_node)
                    if goalnode:
                        self.node_list.append(goalnode)
                        self.generate_final_course(len(self.node_list)-1)

            if animation:
                self.draw_graph(rnd)

            if not exploration:
                if ((not self.search_until_max_iter)
                        and new_node):  # if reaches goal
                    last_index = self.search_best_goal_node()
                    if last_index is not None:
                        return self.generate_final_course(last_index)

        print("reached max iteration")

        if exploration:
            nodes = []
            valid_paths = []
            for node in self.node_list:
                cur = node
                skip = False
                path = []
                for i in range(horizon):
                    if cur.parent:
                        path.append([cur.x, cur.y])
                        cur = cur.parent
                    else:
                        skip = True
                if cur == self.start and not skip:
                    nodes.append(cur)
                    valid_paths.append(path)
            return valid_paths
        else:
            last_index = self.search_best_goal_node()
            if last_index is not None:
                return self.generate_final_course(last_index)
            #print(self.node_list)
            return None

    def straight_line_to_goal(self, node):
        prevx = self.goal_node.x
        prevy = self.goal_node.y
        if self.occupancy_map[int(node.x)][int(node.y)]:
            return False

        maxdist = max(abs(node.x-prevx), abs(node.y-prevy)) +1

        xs = np.linspace(int(node.x), int(prevx), int(maxdist))
        ys = np.linspace(int(node.y), int(prevy), int(maxdist))
        for i in range(len(xs)):
            x = int(xs[i])
            y = int(ys[i])
            if self.occupancy_map[x][y]:
                return False

        prevnode = node
        for i in range(len(xs)):
            x = int(xs[i])
            y = int(ys[i])
            newnode = self.Node(int(x),int(y))
            newnode.parent = prevnode
            prevnode = newnode

        return prevnode

    def choose_parent(self, new_node, near_inds):
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
            Arguments:
            --------
                new_node, Node
                    randomly generated node with a path from its neared point
                    There are not coalitions between this node and th tree.
                near_inds: list
                    Indices of indices of the nodes what are near to new_node

            Returns.
            ------
                Node, a copy of new_node
        """
        if not near_inds:
            return None

        # search nearest cost in near_inds
        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            #if t_node and self.check_collision(t_node, self.obstacle_list):
            if t_node and self.check_collision_map(t_node, self.occupancy_map):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.cost = min_cost

        return new_node

    def search_best_goal_node(self):
        dist_to_goal_list = [
            self.calc_dist_to_goal(n.x, n.y) for n in self.node_list
        ]
        goal_inds = [
            dist_to_goal_list.index(i) for i in dist_to_goal_list
            if i <= self.expand_dis
        ]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            #if self.check_collision(t_node, self.obstacle_list):
            if self.check_collision_map(t_node, self.occupancy_map):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):
        """
        1) defines a ball centered on new_node
        2) Returns all nodes of the three that are inside this ball
            Arguments:
            ---------
                new_node: Node
                    new randomly generated node, without collisions between
                    its nearest node
            Returns:
            -------
                list
                    List with the indices of the nodes inside the ball of
                    radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        # if expand_dist exists, search vertices in a range no more than
        # expand_dist
        if hasattr(self, 'expand_dis'):
            r = min(r, self.expand_dis)
        dist_list = [(node.x - new_node.x)**2 + (node.y - new_node.y)**2
                     for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def rewire(self, new_node, near_inds):
        """
            For each node in near_inds, this will check if it is cheaper to
            arrive to them from new_node.
            In such a case, this will re-assign the parent of the nodes in
            near_inds to new_node.
            Parameters:
            ----------
                new_node, Node
                    Node randomly added which can be joined to the tree

                near_inds, list of uints
                    A list of indices of the self.new_node which contains
                    nodes within a circle of a given radius.
            Remark: parent is designated in choose_parent.

        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            #no_collision = self.check_collision(edge_node, self.obstacle_list)
            no_collision = self.check_collision_map(edge_node, self.occupancy_map)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.path_x = edge_node.path_x
                near_node.path_y = edge_node.path_y
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        return from_node.cost + d

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


def main():
    print("Start " + __file__)

    img = cv2.imread("storeep0_gt_grid_crop_spatial.png",0)
    img = np.rot90(img,2)
    binary = (img < 255)
    import pdb
    pdb.set_trace()

    # ====Search Path with RRT====
    obstacle_list = [
        (5, 5, 1),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2),
        (8, 10, 1),
        (6, 12, 1),
    ]  # [x,y,size(radius)]

    # Set Initial parameters
    rrt_star = RRTStar(
        start=[0, 0],
        goal=[30,40],
        rand_area=[0, 64],
        obstacle_list=obstacle_list,
        expand_dis=5,
        occupancy_map=binary,
        image=img)
    #paths = []
    #for _ in range(10):
    #    path = rrt_star.planning(animation=show_animation)
    #    paths.append(path)
    #import pdb
    #pdb.set_trace()
    path = rrt_star.planning(animation=show_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        if show_animation:
            rrt_star.draw_graph()
            plt.plot([x for (x, y) in path], [y for (x, y) in path], 'r--')
            plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()
