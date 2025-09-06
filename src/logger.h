#ifndef LOGGER_H_INCLUDED
#define LOGGER_H_INCLUDED
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stddef.h>
#include <stdint.h>
#include "macros.h"

#define LOG_FILE "genetic.log"

// FUNCTIONS FOR LOG AN ERROR AND EXIT THE EXECUTION

// DON'T USE THIS

NORETURN_ATTRIBUTE void log_error_exit(const char* error_message, const char* file, const size_t line);

NORETURN_ATTRIBUTE void log_error_exit_ts(const char* error_message, const char* file, const size_t line);

// USE THIS INSTEAD

#define LOG_EXIT(message) log_error_exit(message, __FILE__, __LINE__)
#define LOG_EXIT_THREADSAFE(message) log_error_exit_ts(message, __FILE__, __LINE__)

#define MALLOC_FAIL(byte) LOG_EXIT("malloc failed, tryed to allocate " #byte " bytes")
#define MALLOC_FAIL_THREADSAFE(byte) LOG_EXIT_THREADSAFE("malloc failed, tryed to allocate " #byte " bytes")

#endif
