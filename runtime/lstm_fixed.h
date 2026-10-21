#pragma once

#include <stdint.h>

#define LSTM_FIXED_INPUT 256
#define LSTM_FIXED_HIDDEN 16

/* Q1.7 input/weights, Q3.5 recurrent state, Q1.15 gate outputs. */
void lstm_fixed_infer(const int8_t input[LSTM_FIXED_INPUT], int16_t logits[2]);
