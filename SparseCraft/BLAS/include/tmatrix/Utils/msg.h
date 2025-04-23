#pragma once

#if !defined(MSG_H)
#define MSG_H

typedef enum msg_type
{
    info,
    error,
    debug,
    success,
    warning,
    title,
    rule,
    markdown,
    start_status,
    stop_status,
    custom
} msg_type;

void echo(msg_type type, const char *format, ...);

#endif