#ifndef mesh_mutiblock_H
#define mesh_mutiblock_H

#include "mesh.h"
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
    std::vector<std::string> block_names_;
    std::vector<std::array<PetscInt,3>> block_procs_;
    std::vector<MPI_Comm> block_comms_;
    std::vector<std::unique_ptr<Mesh>> blocks_;

    std::vector<FaceConnectivity> face_connections_;
    std::vector<FaceBC> face_bcs_;

    struct BlockFullCoords {
        bool valid = false;
        PetscInt ni, nj, nk;
        std::vector<PetscReal> x, y, z;
    };
    std::vector<BlockFullCoords> full_coords_;                // 每个块的完整坐标（仅本块进程持有）
    std::vector<std::array<DonorSlab, 6>> donor_slabs_;       // [block_id][face]

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
                   PetscInt lap, PetscInt scheme_vis);
    ~MultiBlockMesh();

    void Initialize(const std::string &filename, PetscInt grid_format = 1);
    void ExportAllToTecplot(const std::string &filename);
    PetscInt getNumBlocks() const { return blocks_.size(); }

    Mesh* getBlock(PetscInt i) { return blocks_[i].get(); }
    const Mesh* getBlock(PetscInt i) const { return blocks_[i].get(); }
    MPI_Comm getBlockComm(PetscInt i) const { return block_comms_[i]; }

    const std::vector<FaceConnectivity>& getFaceConnections() const { return face_connections_; }
    const std::vector<FaceBC>& getFaceBCs() const { return face_bcs_; }

    std::pair<int,int> findConnectedFace(int block, int face) const;
    void printTopology() const;

    /// @brief 获取块间连接面 donor slab（方案B），不存在返回 nullptr
    const DonorSlab* getDonorSlab(int block_id, int face) const;

    /// @brief 两级自动进程分配（块间比例 + 块内最优分解）
    static std::vector<std::array<PetscInt,3>> autoAllocateProcs(
        const std::vector<BlockInfo>& infos, PetscInt total_ranks);
};
#endif
