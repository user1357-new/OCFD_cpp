#ifndef GHOST_CELL_FILLER_H
#define GHOST_CELL_FILLER_H

#include <petscvec.h>
#include <petscdmda.h>
#include <string>
#include <vector>
#include <functional>

class Mesh;
class MultiBlockMesh;

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

    // ================ 核心虚函数（单个数组外推）================
    virtual PetscErrorCode assignGhostOnFace(Mesh* mesh,
                                             int face,
                                             PetscReal*** arr,
                                             const DMDALocalInfo& info,
                                             MultiBlockMesh* all_blocks) = 0;

    // ================ 面填充（默认填充 xyz，可重写）================
    /// @brief 对网格块的单个 ghost 面执行填充（默认：坐标 x,y,z）
    /// 派生类可重写，亦可显式调用 GhostCellFiller::fillGhostCellOnFace 复用坐标填充
    virtual PetscErrorCode fillGhostCellOnFace(Mesh* mesh, int face,
                                                MultiBlockMesh* all_blocks = nullptr);

    // ================ ★ 新增：批量填充任意数组 ★ ================
    /// @brief 对一组已获取的数组指针统一做 ghost 外推（不负责获取/释放）
    /// @param mesh      当前网格块
    /// @param face      面编号
    /// @param arrays    待填充的数组指针列表（如 {rho, u, v, w, p}）
    /// @param info      DMDALocalInfo（与数组对应的）
    /// @param all_blocks 多块信息
    PetscErrorCode fillArraysOnFace(Mesh* mesh, int face,
                                     const std::vector<PetscReal***>& arrays,
                                     const DMDALocalInfo& info,
                                     MultiBlockMesh* all_blocks = nullptr);

    // ================ ★ 新增：用 Vec 列表批量填充 ★ ================
    /// @brief 从 local Vec 列表获取数组 → 填充 ghost → 释放数组
    /// @param mesh      当前网格块
    /// @param face      面编号
    /// @param localVecs 含 ghost 的 local Vec 列表
    /// @param all_blocks 多块信息
    PetscErrorCode fillGhostOnFaceFromVecs(Mesh* mesh, int face,
                                            const std::vector<Vec>& localVecs,
                                            MultiBlockMesh* all_blocks = nullptr);

    // ================ 工具方法（不变）================
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

// ================ JacGhostExtentBC 不变 ================
class JacGhostExtentBC : public GhostCellFiller
{
public:
    JacGhostExtentBC(PetscInt gtype, PetscInt lap);
    const char* name() const override { return "JacGhostExtent"; }

    PetscErrorCode fillGhostCellOnFace(Mesh* mesh, int face,
                                        MultiBlockMesh* all_blocks = nullptr) override;
    PetscErrorCode assignGhostOnFace(Mesh* mesh,
                                     int face,
                                     PetscReal*** arr,
                                     const DMDALocalInfo& info,
                                     MultiBlockMesh* all_blocks) override;

public:
    // ghost_cell_face_ getter（供外部查询填充类型）
    PetscInt getGhostCellFace(int face) const { return ghost_cell_face_[face]; }

private:
    PetscInt ghost_cell_face_[6];
};

// ====================================================================
// 多面组合器：每个面用不同的边界条件
// ====================================================================
class CompositeBC : public GhostCellFiller
{
public:
    CompositeBC(GhostCellFiller* left,   GhostCellFiller* right,
                GhostCellFiller* bottom, GhostCellFiller* top,
                GhostCellFiller* back,   GhostCellFiller* front)
        : GhostCellFiller(left->getLap())
        , face_bc_{left, right, bottom, top, back, front} {}

    const char* name() const override { return "Composite"; }

protected:
    PetscErrorCode assignGhostOnFace(Mesh* mesh,
                                     int face,
                                     PetscReal*** arr,
                                     const DMDALocalInfo& info,
                                     MultiBlockMesh* all_blocks) override;

private:
    GhostCellFiller* face_bc_[6];
};

#endif // GHOST_CELL_FILLER_H
