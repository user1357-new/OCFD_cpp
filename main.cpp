#include "mesh.h"
#include "mesh_mutiblock.h"
#include <petsc.h>
#include <mpi.h>
#include <vector>
#include <array>
#include <string>

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    {
        PetscMPIInt rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

        PetscInt scheme_vis = OCFD_Scheme_CD6;
        PetscInt LAP = (scheme_vis == OCFD_Scheme_CD8) ? 4 : 3;

        // 定义每个块的进程分解，顺序要与 Tecplot 文件中的块顺序一致
        std::vector<std::array<PetscInt,3>> procs = {
            {2, 1, 1},   // 第1块
            {2, 1, 1},   // 第2块
            {2, 1, 1},   // 第3块
            {2, 1, 1},   // 第4块
            {2, 1, 1},   // 第5块
            {2, 1, 1}    // 第6块
        };

        MultiBlockMesh multiMesh(procs, LAP, scheme_vis);
        multiMesh.Initialize("1.dat");

        for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
            Mesh* block = multiMesh.getBlock(b);
            if (block) {
                block->printInfo();
                block->ExportToTecplot("block_" + std::to_string(b));
            }
        }
    } // multiMesh 在这里析构，MPI 仍有效
    PetscFinalize();
    return 0;
}
