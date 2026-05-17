#include "mesh.h"
#include "mesh_mutiblock.h"
#include "BC_ghost_filler.h"
#include <petsc.h>
#include <mpi.h>
#include <vector>
#include <array>
#include <string>
#include <iostream>

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    {
        PetscMPIInt rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

        PetscInt scheme_vis = OCFD_Scheme_CD8;
        PetscInt LAP = (scheme_vis == OCFD_Scheme_CD8) ? 4 : 3;

        std::vector<std::array<PetscInt, 3>> procs = {
            {2, 2, 1}, {2, 1, 1}, {2, 1, 1},
            {2, 2, 1}, {2, 1, 1}, {2, 1, 1}
        };

        MultiBlockMesh multiMesh(procs, LAP, scheme_vis);
        multiMesh.Initialize("1.dat");

        PetscInt face_gtype   = 2;
        PetscInt metric_gtype = 2;
        JacGhostExtentBC ghostFiller(face_gtype, LAP);

        for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
            Mesh* blk = multiMesh.getBlock(b);
            if (!blk) continue;

            // ① 逐面填充（使用无 Vec 重载版本）
            for (int face = 0; face < 6; ++face) {
                ghostFiller.fillGhostCellOnFace(blk, face, &multiMesh);
            }

            // ② 边角
            blk->fillAllEdgeAndCornerGhost(face_gtype, metric_gtype);

            // ③ 导出
            GhostCellFiller::ExportGhostToTecplot(blk, "ghost_block_" + std::to_string(b));
        }
        if (rank == 0)
            std::cout << "虚网格坐标已导出到 ghost_block_*.dat" << std::endl;
    }
    PetscFinalize();
    return 0;
}
