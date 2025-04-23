#include <string>

#include <tmatrix/Utils/msg.h>
#include <tmatrix/Calculation/spmv.cuh>
#include <tmatrix/Calculation/spmm.cuh>
#include <tmatrix/Calculation/spgemm.cuh>

int main(int argc, char **argv)
{
    std::string op = argv[argc - 1];
    int device = atoi(argv[argc - 2]);
    if (op == "spmv") return test_spmv(argv[1], atoi(argv[2]), device);
    else if (op == "spmm") return test_spmm(argv[1], atoi(argv[2]), atoi(argv[argc-3]), device);
    else if (op == "spgemm") return test_spgemm(argv[1], atoi(argv[2]), device);
    else {
        echo(error, "Error: Unknown operation %s", op.c_str());
    }
    return 0;
}
