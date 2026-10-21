#include <stddef.h>

/* Minimal freestanding replacement used when the compiler folds zeroing of
 * the LSTM state arrays into memset. */
void *memset(void *dst, int value, size_t count) {
  unsigned char *out = (unsigned char *)dst;
  while (count-- != 0) *out++ = (unsigned char)value;
  return dst;
}
