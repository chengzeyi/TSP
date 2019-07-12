# -*- coding: utf-8 -*-

from geo import City, World
from tspsolver import AntSolver, GeneticSolver, SimAnnealSolver
import sys
import tkinter


# 顶层类
class TSP(object):
    def __init__(self, world, solvers, root, width=800, height=600):
        self.world = world
        self.solver = solvers
        self.root = root
        self.width = width
        self.height = height

        tkinter_canvas = tkinter.Canvas(root, width=self.width, height=self.height, bg="#EBEBEB", xscrollincrement=1,
                                        yscrollincrement=1)
        self.canvas = tkinter_canvas

        self.canvas.pack(expand=tkinter.YES, fill=tkinter.BOTH)

        self.key2Solver = {}
        for solver in solvers:
            self.key2Solver[solver.active_key] = solver

        key2NameStr = ""
        for solver in solvers:
            key2NameStr += ", {}: {}".format(solver.active_key, solver.name)
        self.root.title("TSP Demo Program, q: Quit, r: Reset" + key2NameStr)

        self.reset()

        self.root.bind("q", self.quit)
        self.root.bind("r", self.reset)
        for solver in solvers:
            self.root.bind(solver.active_key, self.solve)

    # 重置画布状态
    def reset(self, evt = None):
        for item in self.canvas.find_all():
            self.canvas.delete(item)

        self.node_coords = []
        self.node_objs = []

        for city in self.world.cities:
            self.node_coords.append((city.x, city.y))
            node = self.canvas.create_oval(city.x - 5, city.y - 5, city.x + 5, city.y + 5,
                                           fill="#00FF00", outline="#000000", tags="node")
            self.node_objs.append(node)
            self.canvas.create_text(city.x, city.y - 10, text="({}, {})".format(city.x, city.y), fill="black")
        self.canvas.update()

    # 将节点按order顺序连线
    def link(self, order):
        self.canvas.delete("line")
        for i in range(-1, len(order) - 1):
            p1, p2 = self.node_coords[order[i]], self.node_coords[order[i + 1]]
            self.canvas.create_line(p1, p2, fill="#000000", tags="line")
        self.canvas.update()

    # 退出程序:
    def quit(self, evt):
        self.root.destroy()
        sys.exit(0)

    def solve(self, evt):

        key = evt.char
        solver = self.key2Solver[key]
        order = solver.solve(self.world)
        self.link(order)

    def mainloop(self):
        self.root.mainloop()


if __name__ == "__main__":
    print(
        """
        TSP Demo Program
        Author: Cheng Zeyi
        Date: 2019-05
        """
    )

    coords = [
        (100, 30), (200, 180), (380, 450), (350, 120), (330, 300),
        (220, 300), (550, 300), (200, 520), (100, 550), (100, 250),
        (420, 200), (430, 260), (210, 400), (600, 350), (720, 350),
        (650, 450), (620, 300), (760, 480), (450, 550), (300, 120),
        (670, 35), (570, 140), (320, 80), (580, 230), (70, 120)
    ]

    world = World()
    for coord in coords:
        city = City(coord[0], coord[1])
        world.cities.append(city)
    world.reset_distance_graph()

    TSP(world, [AntSolver(), GeneticSolver(), SimAnnealSolver()], tkinter.Tk()).mainloop()
