#ifndef mesh_mutiblock_H
#define mesh_mutiblock_H

#include "mesh.h"
#include <vector>
#include <string>
#include <array>
#include <memory>
#include <mpi.h>
#include "mesh_mutiblock.h"

class MultiBlockMesh {
private:
    PetscInt LAP;
    PetscInt scheme_vis;
    std::vector<std::string> block_names_; 
    // 每个块的进程分解：{{npx, npy, npz}, ...}
    std::vector<std::array<PetscInt,3>> block_procs_;
    // 每个块的子通信域（非本块进程为 MPI_COMM_NULL）
    std::vector<MPI_Comm> block_comms_;
    // 每个块的网格指针（不属于本块则为 nullptr）
    std::vector<std::unique_ptr<Mesh>> blocks_;

    struct BlockInfo {
        std::string name;
        PetscInt ni, nj, nk;
        std::vector<PetscReal> x, y, z;
    };

    std::string extract_string(const std::string &line, const std::string &key);
    PetscInt extract_int(const std::string &line, const std::string &key);
    void ParseTecplotFile(const std::string &filename,
                          std::vector<BlockInfo> &block_infos);

public:
    MultiBlockMesh(const std::vector<std::array<PetscInt,3>> &procs,
                   PetscInt lap, PetscInt scheme_vis);
    
    ~MultiBlockMesh();

    void Initialize(const std::string &tecplot_filename);
    void ExportAllToTecplot(const std::string &filename);
    PetscInt getNumBlocks() const { return blocks_.size(); }

    // 获取本块所属网格（若本进程不属于该块则返回 nullptr）
    Mesh* getBlock(PetscInt i) { return blocks_[i].get(); }
    const Mesh* getBlock(PetscInt i) const { return blocks_[i].get(); }

    // 获取本块通信域（可能为 MPI_COMM_NULL）
    MPI_Comm getBlockComm(PetscInt i) const { return block_comms_[i]; }
};

#endif