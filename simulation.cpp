#include <math.h>
#include <string.h>

extern "C" {
    void zeroingCollisionSpeed(float* u, float* v,
        int low_border, int up_border,
        int left_border, int right_border, int N) {
        for (int i = low_border; i < up_border; i++) {
            for (int j = left_border; j < right_border; j++) {
                u[i * N + j] = 0;
                v[i * N + j] = 0;
            }
        }
    }

    void zeroingCollisionPressure(float* p,
        int low_border, int up_border,
        int left_border, int right_border, int N) {
        for (int i = low_border; i < up_border; i++) {
            for (int j = left_border; j < right_border; j++) {
                p[i * N + j] = 0;
            }
        }
    }

    void updatePressure(float* p, float* p_old, float* u, float* v, float* collision,
        int N, float epsilon, float dx, float dy, float rho, float dt,
        int low_border, int up_border, int left_border, int right_border) {

        int iteration = 0;
        float max_disp = 1.0f;
        float dudx, dvdy, div;

        while (fabs(max_disp) > epsilon && iteration < 100) {
            max_disp = 0.0f;
            for (int i = 1; i < (N - 1); i++) {
                for (int j = 1; j < (N - 1); j++) {
                    int idx = i * N + j;
                    if (collision[idx] == 0) {
                        dudx = (u[i * N + (j + 1)] - u[i * N + (j - 1)]) / (2 * dx);
                        dvdy = (v[(i + 1) * N + j] - v[(i - 1) * N + j]) / (2 * dy);
                        div = dudx + dvdy;

                        p[idx] = (dx*dx * dy*dy * (rho / dt) * div
                            - (p[i * N + (j + 1)] + p[i * N + (j - 1)]) * dy*dy
                            - (p[(i + 1) * N + j] + p[(i - 1) * N + j]) * dx*dx)
                            / (-2 * (dx*dx + dy*dy));

                        max_disp += fabs(p[idx] - p_old[idx]);
                    }
                }
            }

            for (int k = 0; k < N; k++) {
                p[k * N + 0] = p[k * N + 1];
                p[k * N + (N - 1)] = p[k * N + (N - 2)];
                p[0 * N + k] = p[1 * N + k];
                p[(N - 1) * N + k] = p[(N - 2) * N + k];
            }

            zeroingCollisionPressure(p, low_border, up_border, left_border, right_border, N);

            iteration += 1;

            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    p_old[i * N + j] = p[i * N + j];
                }
            }
        }
    }
}