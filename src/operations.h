#ifndef OPERATIONS_H_INCLUDED
#define OPERATIONS_H_INCLUDED
#include <math.h>
#include <stdint.h>

#define MAX_ARITY 2

typedef uint16_t env_index;

union argtype{
    double imm; // immediate
    env_index reg[MAX_ARITY]; // register(s)
};

typedef void (*op_func)(double*, const env_index, const union argtype*);

struct operation{
    op_func function;
    char *name;
    int32_t arity;
};

extern const struct operation Null_Op;
extern const struct operation Move;
extern const struct operation MoveOne;
extern const struct operation Minimum;
extern const struct operation Maximum;
extern const struct operation Inverse;
extern const struct operation Addition;
extern const struct operation Subtraction;
extern const struct operation Multiplication;
extern const struct operation Division;
extern const struct operation SafeDivision;
extern const struct operation Percentage;
extern const struct operation Power;
extern const struct operation SquareRoot;
extern const struct operation Absolute;
extern const struct operation Softmax;
extern const struct operation Exponential;
extern const struct operation Logarithm;
extern const struct operation Logarithm2;
extern const struct operation Sine;
extern const struct operation Cosine;

#endif