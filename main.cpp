#include "sim_config.h"
#include "mesh.h"
#include "mesh_mutiblock.h"
#include "BC_ghost_filler.h"
#include "flow_init.h"
#include "solver.h"
#include <petsc.h>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    {
        PetscMPIInt rank;
        MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

        std::string input_file = (argc >= 2) ? argv[1] : "input.txt";

        try {
            // ---- 1. 配置 ----
            SimConfig cfg(input_file);
            cfg.print();

            // ---- 2. 网格 ----
            MultiBlockMesh multiMesh(cfg.getProcs(), cfg.getLAP(),
                                     cfg.getSchemeVis(), cfg.getMetricDiffType());
            multiMesh.Initialize(cfg.getGridFile(), cfg.getGridFormat());
            multiMesh.applyBCOverrides(cfg.getBCOverrides());
            multiMesh.applyPeriodic(cfg.getPeriodicPairs());

            JacGhostExtentBC ghostFiller(cfg.getFaceGtype(), cfg.getLAP());
            multiMesh.exchangeDonorSlabsPeriodic();
            multiMesh.setupMetrics(ghostFiller,
                                   cfg.getEdgeGtype(),
                                   cfg.getMetricGtype());

            // ---- 3. 初场 ----
            initializeFlowField(multiMesh, cfg);

            // ---- 初始化网格 + 初场输出 ----
            multiMesh.ExportAllFlowToTecplot("init_node_flow");
            multiMesh.ExportAllCellFlowToTecplot("init_cell_flow");

            // ---- 4. 求解（含 BC 注册、ghost 填充、时间推进）----
            setupAndRunSolver(multiMesh, cfg, ghostFiller);

            // ---- 5. 输出 ----
            writeRestart(multiMesh, cfg.getRestartFile());
            multiMesh.ExportAllCellFlowToTecplot("all_blocks_flow_cell");
            multiMesh.ExportAllFlowToTecplot("all_blocks_flow");
            multiMesh.gatherFullMetrics();
            multiMesh.ExportAllMetricsToTecplot("all_blocks_metrics");

            for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
                Mesh* blk = multiMesh.getBlock(b);
                if (!blk) continue;
                blk->ExportCellGhostFlowToTecplot(
                    "cell_ghost_flow_block_" + std::to_string(b));
            }
            if (rank == 0)
                PetscPrintf(PETSC_COMM_WORLD,
                    "Output written: all_blocks_flow_cell.dat, "
                    "all_blocks_flow.dat, all_blocks_metrics.dat, "
                    "cell_ghost_flow_block_*.dat\n");

        } catch (const std::exception &e) {
            PetscPrintf(PETSC_COMM_WORLD, "Fatal error: %s\n", e.what());
            MPI_Abort(PETSC_COMM_WORLD, 1);
        }
    }
    PetscFinalize();
    return 0;
}
