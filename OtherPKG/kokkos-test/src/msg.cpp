#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <msg.h>

const char *_msg_template[] = {
    "[bold cyan][提示][/]",
    "[bold red][错误][/]",
    "[bold yellow][调试][/]",
    "[bold green][成功][/]",
    "[bold yellow][警告][/]",
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