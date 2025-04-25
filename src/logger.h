#include "logger.h"

inline void log_error_exit(const char* error_message, const char* file, const size_t line){
    time_t now;
    time(&now);
    struct tm* timeinfo;
    char actualtime[20];
    timeinfo = localtime(&now);
    strftime(actualtime, 20, "%Y-%m-%d %H:%M:%S", timeinfo);
    FILE *logfile = fopen(LOG_FILE, "a");
    fprintf(logfile, "[%s] ERROR: %s in file: %s, at line %ld\n", actualtime, error_message, file, line);
    fclose(logfile);
    exit(-1);
}

void log_error_exit_ts(const char* error_message, const char* file, const size_t line){
#pragma omp critical
        {
            log_error_exit((error_message), __FILE__, __LINE__);
        }
}