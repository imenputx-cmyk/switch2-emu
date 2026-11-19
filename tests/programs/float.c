/* Floating-point test program: scalar single and double precision. */
#include "rt.h"

/* Prints a double with 6 decimals, using only integer formatting. */
static void print_f64(double v) {
    if (v != v) {
        print("nan");
        return;
    }
    if (v < 0) {
        print("-");
        v = -v;
    }
    u64 whole = (u64)v;
    u64 frac = (u64)((v - (double)whole) * 1000000.0 + 0.5);
    if (frac >= 1000000) {
        whole++;
        frac -= 1000000;
    }
    print_u64(whole);
    print(".");
    for (u64 d = 100000; d > 0; d /= 10) {
        char c = (char)('0' + (frac / d) % 10);
        sys_write(&c, 1);
    }
}

static void mandelbrot(void) {
    static const char shades[] = " .:-=+*#%@";
    for (int y = 0; y < 12; y++) {
        char line[41];
        for (int x = 0; x < 40; x++) {
            float cr = -2.2f + 3.0f * (float)x / 40.0f;
            float ci = -1.2f + 2.4f * (float)y / 12.0f;
            float zr = 0.0f, zi = 0.0f;
            int i = 0;
            while (i < 50 && zr * zr + zi * zi < 4.0f) {
                float t = zr * zr - zi * zi + cr;
                zi = 2.0f * zr * zi + ci;
                zr = t;
                i++;
            }
            line[x] = shades[i * 9 / 50];
        }
        line[40] = 0;
        print(line);
        print("\n");
    }
}

/* Newton's method next to the hardware square root (FSQRT). */
static double newton_sqrt(double x) {
    double g = x > 1.0 ? x / 2.0 : 1.0;
    for (int i = 0; i < 30; i++) g = 0.5 * (g + x / g);
    return g;
}

static void matmul(const float a[3][3], const float b[3][3], float c[3][3]) {
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++) {
            float s = 0.0f;
            for (int k = 0; k < 3; k++) s += a[i][k] * b[k][j];
            c[i][j] = s;
        }
}

static volatile double vzero = 0.0;

int main(void) {
    mandelbrot();

    print("sqrt(2): fsqrt=");
    print_f64(__builtin_sqrt(2.0));
    print(" newton=");
    print_f64(newton_sqrt(2.0));
    print("\n");

    const float a[3][3] = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    const float b[3][3] = {{0.5f, 0, 0}, {0, 0.25f, 0}, {0, 0, -1}};
    float c[3][3];
    matmul(a, b, c);
    print("matmul row 2:");
    for (int j = 0; j < 3; j++) {
        print(" ");
        print_f64(c[2][j]);
    }
    print("\n");

    /* A falling ball with drag, integrated with explicit Euler steps. */
    double pos = 100.0, vel = 0.0;
    int steps = 0;
    while (pos > 0.0) {
        double acc = -9.81 - 0.1 * vel;
        vel += acc * 0.01;
        pos += vel * 0.01;
        steps++;
    }
    print("ball lands after ");
    print_i64(steps);
    print(" steps at v=");
    print_f64(vel);
    print("\n");

    /* volatile inputs force runtime FCVTZS/FCVTZU. Out-of-range values are
     * undefined in C, but AArch64 defines them: they saturate. */
    static volatile double neg = -3.75, big = 1e30, minus_one = -1.0;
    static volatile float tenth = 0.1f;
    print("conversions: ");
    print_i64((i64)neg);
    print(" ");
    print_i64((i64)big);
    print(" ");
    print_u64((u64)minus_one);
    print(" ");
    print_f64((double)tenth);
    print("\n");

    double inf = 1.0 / vzero;
    print("special: 1/0 > 1e308 is ");
    print(inf > 1e308 ? "true" : "false");
    print(", 0/0 is ");
    print_f64(vzero / vzero);
    print("\n");
    return 0;
}
