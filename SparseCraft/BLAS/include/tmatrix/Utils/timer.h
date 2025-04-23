#pragma once
#include <stddef.h>
#include <sys/time.h>

typedef struct
{
    struct timeval start;
    struct timeval end;
} Timer;

#define timer_start(t) gettimeofday(&(t).start, NULL)
#define timer_end(t) gettimeofday(&(t).end, NULL)
#define timer_duration(t) (((t).end.tv_sec - (t).start.tv_sec) * 1000.0 + ((t).end.tv_usec - (t).start.tv_usec) / 1000.0)
