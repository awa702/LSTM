#include "lstm_fixed.h"

/* This reference intentionally contains no floating point or dynamic allocation.
 * Weight arrays are generated into model_fixed_weights.h by quantize_model.py. */
#include "model_fixed_weights.h"

static int32_t clamp_i32(int32_t x, int32_t lo, int32_t hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

/* Placeholder piecewise activations: replace coefficients/table after error sweep.
 * Inputs and outputs are Q1.15. Saturation is explicit and deterministic. */
static int16_t sigmoid_q15(int32_t x) {
  if (x <= -8 * 32768) return 0;
  if (x >= 8 * 32768) return 32767;
  return (int16_t)clamp_i32(16384 + x / 4, 0, 32767);
}

static int16_t tanh_q15(int32_t x) {
  if (x <= -4 * 32768) return -32767;
  if (x >= 4 * 32768) return 32767;
  return (int16_t)clamp_i32(x / 4, -32767, 32767);
}

void lstm_fixed_infer(const int8_t input[LSTM_FIXED_INPUT], int16_t logits[2]) {
  int16_t h[LSTM_FIXED_HIDDEN] = {0};
  int16_t c[LSTM_FIXED_HIDDEN] = {0};
  for (int t = 0; t < LSTM_FIXED_INPUT; ++t) {
    int16_t gate[4 * LSTM_FIXED_HIDDEN];
    for (int g = 0; g < 4 * LSTM_FIXED_HIDDEN; ++g) {
      int32_t acc = model_fixed_bias_ih[g] + model_fixed_bias_hh[g];
      acc += (int32_t)input[t] * model_fixed_weight_ih[g];
      for (int j = 0; j < LSTM_FIXED_HIDDEN; ++j)
        acc += (int32_t)h[j] * model_fixed_weight_hh[g * LSTM_FIXED_HIDDEN + j];
      gate[g] = (int16_t)clamp_i32(acc >> 7, -8 * 32768, 8 * 32768);
    }
    for (int j = 0; j < LSTM_FIXED_HIDDEN; ++j) {
      int16_t i = sigmoid_q15(gate[j]);
      int16_t f = sigmoid_q15(gate[LSTM_FIXED_HIDDEN + j]);
      int16_t g = tanh_q15(gate[2 * LSTM_FIXED_HIDDEN + j]);
      int16_t o = sigmoid_q15(gate[3 * LSTM_FIXED_HIDDEN + j]);
      int32_t cn = ((int32_t)f * c[j] + (int32_t)i * g) >> 15;
      c[j] = (int16_t)clamp_i32(cn, -32768, 32767);
      h[j] = (int16_t)(((int32_t)o * tanh_q15(c[j])) >> 15);
    }
  }
  for (int k = 0; k < 2; ++k) {
    int32_t acc = model_fixed_head_bias[k];
    for (int j = 0; j < LSTM_FIXED_HIDDEN; ++j)
      acc += (int32_t)h[j] * model_fixed_head_weight[k * LSTM_FIXED_HIDDEN + j];
    logits[k] = (int16_t)clamp_i32(acc >> 7, -32768, 32767);
  }
}
