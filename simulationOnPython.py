import arcade
import numpy as np

WIDTH = 1000
HEIGHT = 1000
TITLE = "Simulation"
CELL_SIZE = 10
WHITE = (255, 255, 255)
LITE_BLUE = (200, 230, 255)

"""
Условно частицы воздуха покоятся их скорость относительно Земли равна 0
Крыло (объект) летит со скоростью V
перейдя в систему отсчета связанную с крылом получим, что скорость набегающего потока равна V
"""


class Window(arcade.Window):
    def __init__(self, width, height, title):
        super().__init__(width, height, title)
        self.background_color = arcade.color.WHITE

        self.N = 100
        self.V = 10  # м * с^-1
        self.dx = 0.001
        self.dy = 0.001
        self.dt = 0.0000625

        self.u_old = np.full((self.N, self.N), self.V, dtype=np.float32)
        self.u_new = np.zeros((self.N, self.N), dtype=np.float32)
        self.v_old = np.zeros((self.N, self.N), dtype=np.float32)
        self.v_new = np.zeros((self.N, self.N), dtype=np.float32)
        self.collision = np.zeros((self.N, self.N), dtype=np.float32)
        self.p = np.zeros((self.N, self.N), dtype=np.float32)
        self.p_old = np.zeros((self.N, self.N), dtype=np.float32)

        self.time_total = 0

        self.left_border = 40   # X_min
        self.right_border = 60  # X_max
        self.low_border = 40    # Y_min
        self.up_border = 60     # Y_max

        self.rho = 1.225 / 10 ** 6
        self.epsilon = 1e-3
        self.max_disp = 0


        for j in range(self.low_border, self.up_border):
            for i in range(self.left_border, self.right_border):
                self.v_old[j][i] = 0
                self.u_old[j][i] = 0
                self.collision[j][i] = 1

    def zeroingCollisionSpeed(self):
        # j — Y, i — X
        for j in range(self.low_border, self.up_border):
            for i in range(self.left_border, self.right_border):
                self.u_old[j][i] = 0
                self.v_old[j][i] = 0

    def speedTransmission(self):
        for j in range(self.low_border, self.up_border):
            for i in range(self.left_border, self.right_border):
                self.u_new[j][i] = 0
                self.v_new[j][i] = 0

        for j in range(1, self.N - 1):
            for i in range(1, self.N - 1):
                if self.collision[j][i] == 1:
                    continue

                if self.u_old[j][i] > 0:
                    dudx = (self.u_old[j][i] - self.u_old[j][i-1]) / self.dx
                else:
                    dudx = (self.u_old[j][i+1] - self.u_old[j][i]) / self.dx

                if self.v_old[j][i] > 0:
                    dudy = (self.u_old[j][i] - self.u_old[j-1][i]) / self.dy
                else:
                    dudy = (self.u_old[j+1][i] - self.u_old[j][i]) / self.dy

                self.u_new[j][i] = self.u_old[j][i] - self.dt * (
                        self.u_old[j][i] * dudx + self.v_old[j][i] * dudy)

                if self.u_old[j][i] > 0:
                    dvdx = (self.v_old[j][i] - self.v_old[j][i-1]) / self.dx
                else:
                    dvdx = (self.v_old[j][i+1] - self.v_old[j][i]) / self.dx

                if self.v_old[j][i] > 0:
                    dvdy = (self.v_old[j][i] - self.v_old[j-1][i]) / self.dy
                else:
                    dvdy = (self.v_old[j+1][i] - self.v_old[j][i]) / self.dy

                self.v_new[j][i] = self.v_old[j][i] - self.dt * (
                        self.u_old[j][i] * dvdx + self.v_old[j][i] * dvdy)


        self.u_new[:, 0] = self.V
        self.v_new[:, 0] = 0

        self.u_new[:, self.N-1] = self.u_new[:, self.N-2]
        self.v_new[:, self.N-1] = self.v_new[:, self.N-2]

        self.u_new[0, :] = self.u_new[1, :]
        self.v_new[0, :] = 0
        self.u_new[self.N-1, :] = self.u_new[self.N-2, :]
        self.v_new[self.N-1, :] = 0

        for j in range(self.low_border, self.up_border):
            for i in range(self.left_border, self.right_border):
                self.u_new[j][i] = 0
                self.v_new[j][i] = 0

    def updatePressure(self):
        self.max_disp = 0
        for j in range(1, self.N - 1):
            for i in range(1, self.N - 1):
                if self.collision[j][i] == 0:
                    dudx = (self.u_new[j][i+1] - self.u_new[j][i-1]) / (2 * self.dx)
                    dvdy = (self.v_new[j+1][i] - self.v_new[j-1][i]) / (2 * self.dy)
                    div = dudx + dvdy

                    self.p[j][i] = (self.dx**2 * self.dy**2 * (self.rho / self.dt) * div -
                                    (self.p[j][i+1] + self.p[j][i-1]) * self.dy**2 -
                                    (self.p[j+1][i] + self.p[j-1][i]) * self.dx**2) / \
                                   (-2 * (self.dx**2 + self.dy**2))

                    self.max_disp += abs(self.p[j][i] - self.p_old[j][i])

        self.p[:, 0] = self.p[:, 1]
        self.p[:, self.N-1] = self.p[:, self.N-2]
        self.p[0, :] = self.p[1, :]
        self.p[self.N-1, :] = self.p[self.N-2, :]

        for j in range(self.low_border, self.up_border):
            for i in range(self.left_border, self.right_border):
                self.p[j][i] = 0

        self.p_old = self.p.copy()

    def finalUpdateSpeed(self):
        for j in range(1, self.N - 1):
            for i in range(1, self.N - 1):
                if self.collision[j][i] == 1:
                    continue
                self.u_new[j][i] -= (self.dt / self.rho) * (self.p[j][i+1] - self.p[j][i-1]) / (2 * self.dx)
                self.v_new[j][i] -= (self.dt / self.rho) * (self.p[j+1][i] - self.p[j-1][i]) / (2 * self.dy)

        for j in range(self.low_border, self.up_border):
            for i in range(self.left_border, self.right_border):
                self.u_new[j][i] = 0
                self.v_new[j][i] = 0

        max_allowed = self.V * 5
        self.u_new = np.clip(self.u_new, -max_allowed, max_allowed)
        self.v_new = np.clip(self.v_new, -max_allowed, max_allowed)

    def on_update(self, delta_time):
        self.zeroingCollisionSpeed()
        self.speedTransmission()

        iteration = 0
        self.max_disp = 1.0
        while abs(self.max_disp) > self.epsilon and iteration < 100:
            self.updatePressure()
            iteration += 1

        self.finalUpdateSpeed()

        self.v_old = self.v_new.copy()
        self.u_old = self.u_new.copy()

    def on_draw(self):
        self.clear()
        max_speed = np.max(np.sqrt(self.u_new ** 2 + self.v_new ** 2))
        if max_speed == 0:
            max_speed = 1.0

        for j in range(self.N):
            for i in range(self.N):
                x = i * CELL_SIZE
                y = j * CELL_SIZE

                if self.collision[j][i] == 1:
                    arcade.draw_lbwh_rectangle_filled(x, y, CELL_SIZE - 1, CELL_SIZE - 1, WHITE)
                else:
                    speed = np.sqrt(self.u_new[j][i] ** 2 + self.v_new[j][i] ** 2)
                    norm = min(speed / max_speed, 1.0)
                    # Синий — медленно, красный — быстро
                    color = (int(norm * 255), 0, int((1 - norm) * 255))

                    arcade.draw_lbwh_rectangle_filled(x, y, CELL_SIZE - 1, CELL_SIZE - 1, color)


def main():
    sm = Window(WIDTH, HEIGHT, TITLE)
    arcade.run()


if __name__ == "__main__":
    main()