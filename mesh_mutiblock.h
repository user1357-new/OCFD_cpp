#ifndef mesh_mutiblock_H
#define mesh_mutiblock_H

#include "mesh.h"
#include "sim_config.h"
#include <vector>
#include <string>
#include <array>
#include <memory>
#include <mpi.h>
#include <cgnslib.h>

struct FaceConnectivity {
    int block_a;        // 本块索引
    int face_a;         // 本块面 (0~5)
    int block_b;        // 对端块索引
    int face_b;         // 对端面 (0~5)
    int transform[3];   // CGNS Transform: 本块(ijk)→对端(ijk) 映射
    cgsize_t pointrange[6];   // 本块连接面的索引范围 [imin,jmin,kmin,imax,jmax,kmax]
    cgsize_t donorrange[6];   // 对端块连接面的索引范围
    bool is_periodic = false; // 是否为平移周期性连接
    PetscReal translate[3] = {0,0,0}; // 周期平移向量 (MIN面→MAX面)
};

struct FaceBC {
    int block;
    int face;            // 0~5
    std::string bc_type; // 边界条件类型
    std::string family;
};

struct DonorSlab {
    bool valid = false;
    PetscInt ni, nj, nk;
    PetscInt i0, j0, k0;
    std::vector<PetscReal> x, y, z;
    std::vector<PetscReal> u[5];    // 守恒量 slab
    std::vector<PetscReal> m[10];   // 度量系数 slab (akx,aky,akz, aix,aiy,aiz, asx,asy,asz, ajac)
};

// 网格块信息（解析用）
struct BlockInfo {
    std::string name;
    PetscInt ni, nj, nk;
    std::vector<PetscReal> x, y, z;
};

class MultiBlockMesh {
private:
    PetscInt LAP;
    PetscInt scheme_vis;
    PetscInt metric_diff_type_;
    std::vector<std::string> block_names_;
    std::vector<std::array<PetscInt,3>> block_procs_;
    std::vector<MPI_Comm> block_comms_;
    std::vector<std::unique_ptr<Mesh>> blocks_;

    std::vector<FaceConnectivity> face_connections_;
    std::vector<FaceBC> face_bcs_;

    struct BlockFullCoords {
        bool valid = false;
        bool flow_ready = false;     // U 是否已汇集
        bool metrics_ready = false;  // 度量系数是否已汇集
        PetscInt ni, nj, nk;
        std::vector<PetscReal> x, y, z;
        std::vector<PetscReal> u[5]; // 守恒量
        std::vector<PetscReal> m[10];// 度量系数
        // 原始格点坐标（输出用）
        PetscInt node_ni, node_nj, node_nk;
        std::vector<PetscReal> node_x, node_y, node_z;
    };
    std::vector<BlockFullCoords> full_coords_;                // 每个块的完整坐标（仅本块进程持有）
    std::vector<std::array<DonorSlab, 6>> donor_slabs_;       // [block_id][face]
    std::vector<PetscInt> block_start_rank_;                  // ★ 每个块的根进程（世界通信域）

    void exchangeDonorSlabs(const std::vector<PetscInt>& block_start_rank);

    std::string extract_string(const std::string &line, const std::string &key);
    PetscInt extract_int(const std::string &line, const std::string &key);
    void ParseTecplotFile(const std::string &filename,
                          std::vector<BlockInfo> &block_infos);
    void ParsePlot3DFile(const std::string &filename,
                          std::vector<BlockInfo> &block_infos);
    void ParseCGNSFile(const std::string &filename,
                       std::vector<BlockInfo> &block_infos);

public:
    MultiBlockMesh(const std::vector<std::array<PetscInt,3>> &procs,
                                      PetscInt lap, PetscInt scheme_vis,PetscInt metric_diff_type);

    ~MultiBlockMesh();

    void Initialize(const std::string &filename, PetscInt grid_format = 1);
    void ExportAllToTecplot(const std::string &filename);
    void ExportAllFlowToTecplot(const std::string &filename);
    void ExportAllCellFlowToTecplot(const std::string &filename);
    void ExportAllMetricsToTecplot(const std::string &filename);
    PetscInt getNumBlocks() const { return blocks_.size(); }

    Mesh* getBlock(PetscInt i) { return blocks_[i].get(); }
    const Mesh* getBlock(PetscInt i) const { return blocks_[i].get(); }
    MPI_Comm getBlockComm(PetscInt i) const { return block_comms_[i]; }

    const std::vector<FaceConnectivity>& getFaceConnections() const { return face_connections_; }
    const std::vector<FaceBC>& getFaceBCs() const { return face_bcs_; }

    std::pair<int,int> findConnectedFace(int block, int face) const;
    void printTopology() const;
    /// @brief 三阶段度量系数初始化（需在 Initialize + exchangeDonorSlabs 之后调用）
    /// Phase 1: 填所有块的坐标 ghost（面 + 边角）
    /// Phase 2: 所有块计算度量系数（CD6/CD8，满模板，不降阶）
    /// Phase 3: 填所有块的度量系数 ghost（面 + 边角，全部外推）
    void setupMetrics(class JacGhostExtentBC& filler, int edge_gtype, int metric_gtype);

    /// @brief 从 DMDA 汇集各块完整守恒量到 full_coords_
    void gatherFullFlow();
    /// @brief 交换守恒量 donor slab（与 exchangeDonorSlabs 同理）
    void exchangeDonorSlabsU();

    /// @brief 从 DMDA 汇集各块完整度量系数到 full_coords_
    void gatherFullMetrics();
    /// @brief 为周期性连接打包度量系数 DonorSlab
    void exchangeDonorSlabsMetricsPeriodic();

    /// @brief 用用户定义的 BC 覆写 CGNS 默认值
    void applyBCOverrides(const std::vector<BCOverride>& overrides);

    /// @brief 注册周期性边界对（含校验）
    void applyPeriodic(const std::vector<PeriodicPair>& pairs);

    /// @brief 为周期性连接打包坐标 DonorSlab
    void exchangeDonorSlabsPeriodic();

    /// @brief 为周期性连接打包守恒量 DonorSlab
    void exchangeDonorSlabsUPeriodic();

    /// @brief 获取块间连接面 donor slab（方案B），不存在返回 nullptr
    const DonorSlab* getDonorSlab(int block_id, int face) const;

    /// @brief 获取已汇集的完整块数据（只读）
    const BlockFullCoords& getFullCoords(int b) const { return full_coords_[b]; }

    /// @brief 两级自动进程分配（块间比例 + 块内最优分解）
    static std::vector<std::array<PetscInt,3>> autoAllocateProcs(
        const std::vector<BlockInfo>& infos, PetscInt total_ranks);
};
#endif
