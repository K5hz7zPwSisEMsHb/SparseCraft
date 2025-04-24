<h1 style="text-align: center"> SparseCraft Artifact </h1>

## Install

Before you start, you need to have the cuda 12.8 and Python^3.8 environments configured.

1. Clone the repository:

   ```shell
   git clone git@github.com:zxnvsvsbpxevmuxa/SparseCraft.git
   ```

2. Install Python dependencies

   ```shell
   cd SparseCraft
   pip3 install -r requirements.txt
   ```

3. Use the `nvidia-smi` command to view the GPU device ID you want to use and record it. Here, it is assumed that the device ID of 3090 is 1.

## Execution

**When executing, make sure you are always in the project root directory.**

1. Initialize platform (5 minutes): `qrun init <3090 | 4090 | 5090>`. This command will prompt you to enter the device ID of the corresponding GPU. Please make sure that the device ID is consistent with that displayed by the `nvidia-smi` command.

   Examples:

   `qrun init 3090`: Initialize for RTX 3090

   `qrun init 4090`: Initialize for RTX 4090

   `qrun init 5090`: Initialize for RTX 5090

2. (L1 Task, 20 minutes) After completing the initialization, you can start to reproduce Figures 6, 10 and Table 10.

   1. Generate the data for Figures 6 and 10: `qrun representative`
   2. Show Table 10: `qrun fine-tune`

3. (L2 Task, 1 day) With this level of tasks, you can reproduce Figure 7, Figure 8, and Figure 9.

   1. Download the matrix set: `qrun download-matrices`. If this step is successful, you can find 2853 matrix files in the `SparseCraft/Matrices` directory.
   2. Generate Figure 7 data: `qrun spmv`
   3. Generate Figure 8 data: `qrun spmm`
   4. Generate Figure 9 data: `qrun spgemm`

4. If your GPUs are distributed across multiple machines, you will need to manually aggregate the intermediate results generated in the res directory. For your convenience, we have provided complete test results in the artifact (`res/ready_to_draw`). These precomputed results allow you to reproduce the charts using the following command, even if your hardware specifications do not match the requirements stated in the artifact.

   1. Draw Figure 6: `qrun draw-figure 6`
   2. Draw Figure 6: `qrun draw-figure 7`
   3. Draw Figure 6: `qrun draw-figure 8`
   4. Draw Figure 6: `qrun draw-figure 9`
   5. Draw Figure 6: `qrun draw-figure 10`

