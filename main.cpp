#include "sim_config.h"
#include "mesh.h"
#include "mesh_mutiblock.h"
#include "BC_ghost_filler.h"
#include "BC_filler.h"
#include "flow_init.h"
#include <petsc.h>
#include <cmath>
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
            SimConfig cfg(input_file);
            cfg.print();

            MultiBlockMesh multiMesh(cfg.getProcs(), cfg.getLAP(),
                                     cfg.getSchemeVis(), cfg.getMetricDiffType());
            multiMesh.Initialize(cfg.getGridFile(), cfg.getGridFormat());
            multiMesh.applyBCOverrides(cfg.getBCOverrides());
            multiMesh.applyPeriodic(cfg.getPeriodicPairs());
            multiMesh.printTopology();

            JacGhostExtentBC ghostFiller(cfg.getFaceGtype(), cfg.getLAP());

            multiMesh.exchangeDonorSlabsPeriodic();

            multiMesh.setupMetrics(ghostFiller,
                                   cfg.getEdgeGtype(),
                                   cfg.getMetricGtype());

            initializeFlowField(multiMesh, cfg);

            // ---- U block-to-block ghost exchange ----
            multiMesh.gatherFullFlow();
            multiMesh.exchangeDonorSlabsU();
            multiMesh.exchangeDonorSlabsUPeriodic();

            // Fill U ghosts on connected faces (physical BC faces skipped)
            for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
                Mesh* blk = multiMesh.getBlock(b);
                if (!blk) continue;
                for (int face = 0; face < 6; ++face) {
                    auto conn = multiMesh.findConnectedFace(b, face);
                    if (conn.first >= 0) {
                        ghostFiller.fillGhostOnFaceFromVecs(blk, face,
                            blk->getLocalUVecs(), &multiMesh, b, 3);
                    }
                }
            }
            if (rank == 0) PetscPrintf(PETSC_COMM_WORLD, "U ghost fill done.\n");

            // ================================================================
            // 物理边界条件 ghost 填充（工厂 + 注册模式）
            // ================================================================
            {
                PetscReal gamma = cfg.getGamma();
                PetscReal Ma    = cfg.getMach();
                PetscReal AoA   = cfg.getAttack()  * M_PI / 180.0;
                PetscReal beta  = cfg.getSideslip() * M_PI / 180.0;
                PetscReal rho_inf = 1.0;
                PetscReal u_inf   = std::cos(AoA) * std::cos(beta);
                PetscReal v_inf   = std::sin(AoA) * std::cos(beta);
                PetscReal w_inf   = std::sin(beta);
                PetscReal p_inf   = 1.0 / (gamma * Ma * Ma);

                BCFillerRegistry& reg = BCFillerRegistry::instance();
                reg.add("BCWall",
                        std::unique_ptr<BCFaceFiller>(
                            new BCWallFiller(gamma)));
                reg.add("BCInlet",
                        std::unique_ptr<BCFaceFiller>(
                            new BCInletFiller(gamma,
                                cfg.getInletRho(), cfg.getInletU(),
                                cfg.getInletV(), cfg.getInletW(),
                                cfg.getInletP())));
                reg.add("BCFarfield",
                        std::unique_ptr<BCFaceFiller>(
                            new BCFarfieldFiller(gamma, rho_inf,
                                                 u_inf, v_inf, w_inf, p_inf)));
                reg.add("BCExtrapolate",
                        std::unique_ptr<BCFaceFiller>(
                            new BCExtrapolateFiller()));

                // 遍历所有块的物理边界面
                for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
                    Mesh* blk = multiMesh.getBlock(b);
                    if (!blk) continue;
                    for (int face = 0; face < 6; ++face) {
                        // 块间连接面已在上面处理过，跳过
                        if (multiMesh.findConnectedFace(b, face).first >= 0)
                            continue;

                        // 查找该面的 BC 类型
                        std::string bc_type;
                        for (const auto& fbc : multiMesh.getFaceBCs()) {
                            if (fbc.block == b && fbc.face == face) {
                                bc_type = fbc.bc_type;
                                break;
                            }
                        }
                        if (bc_type.empty()) continue;  // 没标 BC 的面跳过

                        BCFaceFiller* filler = reg.get(bc_type);
                        if (!filler) {
                            if (rank == 0)
                                PetscPrintf(PETSC_COMM_WORLD,
                                    "WARNING: Unknown BC type '%s' on block %d face %d, skipped\n",
                                    bc_type.c_str(), b, face);
                            continue;
                        }
                        if (rank == 0)
                            PetscPrintf(PETSC_COMM_WORLD,
                                "  [BC] block %d face %d: type='%s' → %s\n",
                                b, face, bc_type.c_str(), filler->name());

                        PetscReal*** u[5];
                        DMDALocalInfo info;
                        blk->getLocalUArrays(u, info);
                        filler->fillGhostFace(blk, face, u, info);
                        blk->restoreLocalUArrays(u);
                    }
                }
                if (rank == 0)
                    PetscPrintf(PETSC_COMM_WORLD, "Physical BC ghost fill done.\n");
            }

            // ---- 写出二进制重启文件 ----
            writeRestart(multiMesh, cfg.getRestartFile());

            multiMesh.ExportAllCellFlowToTecplot("all_blocks_flow_cell");
            multiMesh.ExportAllFlowToTecplot("all_blocks_flow");
            multiMesh.gatherFullMetrics();
            multiMesh.ExportAllMetricsToTecplot("all_blocks_metrics");
            // cell ghost (coords + metrics + flow)
            for (int b = 0; b < multiMesh.getNumBlocks(); ++b) {
                Mesh* blk = multiMesh.getBlock(b);
                if (!blk) continue;
                blk->ExportCellGhostFlowToTecplot(
                    "cell_ghost_flow_block_" + std::to_string(b));
            }

            if (rank == 0) {
                std::cout << "Flow field (cell-center) -> all_blocks_flow_cell.dat"  << std::endl;
                std::cout << "Flow field (node-based)  -> all_blocks_flow.dat"       << std::endl;
                std::cout << "Metrics (node-based)    -> all_blocks_metrics.dat" << std::endl;
                std::cout << "Cell ghost+flow grid    -> cell_ghost_flow_block_*.dat" << std::endl;
            }

        } catch (const std::exception &e) {
            PetscPrintf(PETSC_COMM_WORLD, "Fatal error: %s\n", e.what());
            MPI_Abort(PETSC_COMM_WORLD, 1);
        }
    }
    PetscFinalize();
    return 0;
}
