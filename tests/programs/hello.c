/* First guest program for switch2-emu: exercises compiler-generated code. */
#include "rt.h"

static u64 fib(u64 n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

static u64 factorial(u64 n) { return n <= 1 ? 1 : n * factorial(n - 1); }

static int count_primes(int limit) {
    u8 composite[256];
    memset(composite, 0, sizeof composite);
    int count = 0;
    for (int i = 2; i < limit; i++) {
        if (composite[i]) continue;
        count++;
        for (int j = i * i; j < limit; j += i) composite[j] = 1;
    }
    return count;
}

static void sort(i64 *a, int n) {
    for (int i = 1; i < n; i++) {
        i64 key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) {
            a[j + 1] = a[j];
            j--;
        }
        a[j + 1] = key;
    }
}

/* FNV-1a, a typical hashing loop with 64-bit multiplies. */
static u64 fnv1a(const char *s) {
    u64 h = 0xcbf29ce484222325ul;
    while (*s) {
        h ^= (u8)*s++;
        h *= 0x100000001b3ul;
    }
    return h;
}

static i64 apply(i64 (*f)(i64, i64), i64 a, i64 b) { return f(a, b); }
static i64 sub(i64 a, i64 b) { return a - b; }
static i64 divide(i64 a, i64 b) { return a / b; }

static volatile i64 zero = 0; /* defeat constant folding */

int main(void) {
    print("Hello from the guest CPU!\n");

    print("fib(20) = ");
    print_u64(fib(20));
    print("\nfactorial(20) = ");
    print_u64(factorial(20));
    print("\nprimes below 256 = ");
    print_i64(count_primes(256));

    i64 values[8] = {42, -7, 1000000007, 0, -123456789012, 5, 5, 99};
    sort(values, 8);
    print("\nsorted =");
    for (int i = 0; i < 8; i++) {
        print(" ");
        print_i64(values[i]);
    }

    print("\nfnv1a(\"switch2-emu\") = ");
    print_hex(fnv1a("switch2-emu"));
    print("\napply(sub, 10, 52) = ");
    print_i64(apply(sub, 10, 52));
    print("\napply(divide, -100, 7) = ");
    print_i64(apply(divide, -100, 7));
    print("\ndivide by zero = ");
    print_i64(divide(1, zero));
    print("\n");
    return 7;
}
