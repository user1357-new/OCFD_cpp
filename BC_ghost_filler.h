#ifndef GHOST_CELL_FILLER_H
#define GHOST_CELL_FILLER_H

#include <petscvec.h>
#include <petscdmda.h>
#include <string>
#include <vector>
#include <functional>

class Mesh;
class MultiBlockMesh;

// ====================================================================
// GhostCellFiller — 基类
//
// 职责分层：
//   基类：Vec 遍历、GetArray/RestoreArray、ghost 范围计算
//   派生类：具体填什么值（外推 / 抄数 / 将来物理量传输）
//
// 扩展路径（物理量传输）：
//   1. MultiBlockMesh 中 DonorSlab 扩展物理量数组（rho, u, v, w, p...）
//   2. Mesh 增加 getLocalConservativeVecs() / getLocalPrimitiveVecs()
//   3. 调用 fillGhostOnFaceFromVecs 传入 block_id >= 0 进行抄数
//   4. 派生类 assignGhostOnConnection 中根据 comp_idx 区分坐标/物理量：
//        comp_idx 0-2   → 从 DonorSlab::x/y/z 抄
//        comp_idx 3-N   → 从 DonorSlab::q[n] 抄（未来扩展）
// ====================================================================
class GhostCellFiller
{
public:
    enum FaceID : int {
        FACE_LEFT   = 0,  // i 负方向
        FACE_RIGHT  = 1,  // i 正方向
        FACE_BOTTOM = 2,  // j 负方向
        FACE_TOP    = 3,  // j 正方向
        FACE_BACK   = 4,  // k 负方向
        FACE_FRONT  = 5   // k 正方向
    };

    GhostCellFiller(PetscInt lap) : lap_(lap) {}
    virtual ~GhostCellFiller() = default;

    PetscInt getLap() const { return lap_; }

    // ==================== 派生类需实现 ====================
    virtual PetscErrorCode assignGhostOnFace(Mesh* mesh,
                                             int face,
                                             PetscReal*** arr,
                                             const DMDALocalInfo& info,
                                             MultiBlockMesh* all_blocks) = 0;

    /// @brief 块间连接抄数（virtual，默认 no-op）
    virtual PetscErrorCode assignGhostOnConnection(Mesh* mesh,
                                                   int block_id,
                                                   int face,
                                                   PetscReal*** arr,
                                                   const DMDALocalInfo& info,
                                                   MultiBlockMesh* all_blocks,
                                                   int comp_idx);

    // ==================== 唯一的 Vec 遍历器 ====================
    /// @brief 对 Vec 列表逐个填充 ghost
    /// @param block_id   -1（默认）= 外推，走 assignGhostOnFace
    ///                   >=0 = 连接抄数，走 assignGhostOnConnection
    /// @param comp_offset 抄数时的起始 comp_idx（仅 block_id>=0 时生效）
    PetscErrorCode fillGhostOnFaceFromVecs(
        Mesh* mesh, int face,
        const std::vector<Vec>& localVecs,
        MultiBlockMesh* all_blocks = nullptr,
        int block_id = -1,
        int comp_offset = 0);

    // ==================== 原有接口（不改） ====================
    virtual PetscErrorCode fillGhostCellOnFace(Mesh* mesh, int face,
                                                MultiBlockMesh* all_blocks = nullptr);

    static void GetGhostRange(int face, int LAP,
                              int NX, int NY, int NZ,
                              const DMDALocalInfo& info,
                              int& i_min, int& i_max,
                              int& j_min, int& j_max,
                              int& k_min, int& k_max);

    static PetscErrorCode ExportGhostToTecplot(Mesh* mesh,
                                               const std::string& base_filename);
    virtual const char* name() const = 0;

protected:
    PetscInt lap_;
};

// ==================== JacGhostExtentBC ====================
class JacGhostExtentBC : public GhostCellFiller
{
public:
    JacGhostExtentBC(PetscInt gtype, PetscInt lap);
    const char* name() const override { return "JacGhostExtent"; }

    // 物理外推
    PetscErrorCode assignGhostOnFace(Mesh* mesh,
                                     int face,
                                     PetscReal*** arr,
                                     const DMDALocalInfo& info,
                                     MultiBlockMesh* all_blocks) override;

    // 块间连接抄数
    PetscErrorCode assignGhostOnConnection(Mesh* mesh,
                                           int block_id,
                                           int face,
                                           PetscReal*** arr,
                                           const DMDALocalInfo& info,
                                           MultiBlockMesh* all_blocks,
                                           int comp_idx) override;

    // 分流入口：度量外推 + 坐标分流（连接面抄数 / 物理面外推）
    PetscErrorCode fillGhostCellOnFace(Mesh* mesh, int face, int block_id,
                                       MultiBlockMesh* all_blocks = nullptr);

    PetscInt getGhostCellFace(int face) const { return ghost_cell_face_[face]; }

private:
    PetscInt ghost_cell_face_[6];
};
#endif
