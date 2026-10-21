#include <stdint.h>

/* RV32I has no M extension. Keep this helper free of C multiplication so the
 * compiler does not recursively request __mulsi3 while building it. */
int __mulsi3(int left, int right) {
  uint32_t a = (uint32_t)left;
  uint32_t b = (uint32_t)right;
  uint32_t result = 0;
  int negative = (left < 0) != (right < 0);
  if (left < 0) a = 0u - a;
  if (right < 0) b = 0u - b;
  while (b != 0) {
    if (b & 1u) result += a;
    a <<= 1;
    b >>= 1;
  }
  return negative ? (int)(0u - result) : (int)result;
}
