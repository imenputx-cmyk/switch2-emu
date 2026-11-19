#pragma once
#include <stddef.h>
void *malloc(size_t n);
void *calloc(size_t n, size_t s);
void *realloc(void *p, size_t n);
void *aligned_alloc(size_t a, size_t n);
void *memalign(size_t a, size_t n);
void free(void *p);
void abort(void) __attribute__((noreturn));
void exit(int) __attribute__((noreturn));
