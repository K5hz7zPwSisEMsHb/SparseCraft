#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <tmatrix/Utils/msg.h>

const char *_msg_template[] = {
    "[bold cyan][INFO][/]",
    "[bold red][ERRO][/]",
    "[bold yellow][DBUG][/]",
    "[bold green][SUCC][/]",
    "[bold yellow][WARN][/]",
    "__TITLE__",
    "__RULE__",
    "__MARKDOWN__",
    "__START__",
    "__STOP__",
    "__EXECUTE__"};

void echo(msg_type type, const char *format, ...)
{
    printf("%s ", _msg_template[type]);
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
    fflush(stdout);
}