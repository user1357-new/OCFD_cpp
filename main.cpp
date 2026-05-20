#include "sim_config.h"
#include "mesh.h"
#include "mesh_mutiblock.h"
#include "BC_ghost_filler.h"
#include <petsc.h>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    {
        // 命令行: ./a.out [input.txt]，默认 input.txt
        std::string input_file = (argc >= 2) ? argv[1] : "input.txt";

        try {
            SimConfig cfg(input_file);
            cfg.print();

            MultiBlockMesh multiMesh(cfg.getProcs(), cfg.getLAP(), cfg.getSchemeVis());
            multiMesh.Initialize(cfg.getGridFile());

            JacGhostExtentBC ghostFiller(cfg.getFaceGtype(), cfg.getLAP());

            for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
                Mesh* blk = multiMesh.getBlock(b);
                if (!blk) continue;

                // ① 面 ghost
                for (int face = 0; face < 6; ++face)
                    ghostFiller.fillGhostCellOnFace(blk, face, &multiMesh);

                // ② 边角 ghost（独立阶数）
                blk->fillAllEdgeAndCornerGhost(cfg.getEdgeGtype(), cfg.getMetricGtype());

                // ③ 导出
                GhostCellFiller::ExportGhostToTecplot(blk, "ghost_block_" + std::to_string(b));
            }

            PetscMPIInt rank;
            MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
            if (rank == 0)
                std::cout << "Ghost grid exported to ghost_block_*.dat" << std::endl;

        } catch (const std::exception &e) {
            PetscPrintf(PETSC_COMM_WORLD, "Fatal error: %s\n", e.what());
            MPI_Abort(PETSC_COMM_WORLD, 1);
        }
    }
    PetscFinalize();
    return 0;
}
