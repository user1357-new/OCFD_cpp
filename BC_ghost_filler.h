#ifndef GHOST_CELL_FILLER_H
#define GHOST_CELL_FILLER_H

#include <petscvec.h>
#include <petscdmda.h>
#include <string>
#include <vector>

class Mesh;
class MultiBlockMesh;

/// @brief 虚网格填充器的抽象基类（策略模式）
///
/// 继承此类并实现 assignGhostOnFace()，即可为每个面赋值虚网格。
/// 每个面可单独指定边界条件类型，也可统一处理。
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

    /// @brief 获取虚网格层数
    PetscInt getLap() const { return lap_; }

    /// @brief 对网格块的单个 ghost 面执行填充（坐标 + 度量系数）
    /// @param mesh      当前网格块
    /// @param face      面编号（0~5），对应 FaceID
    /// @param all_blocks 多块信息（单块时为 nullptr）
    virtual PetscErrorCode fillGhostCellOnFace(Mesh* mesh, int face,
                                                MultiBlockMesh* all_blocks = nullptr) = 0;
    /// @brief 获取某个面的 ghost 区域索引范围（自动裁剪到当前 rank 拥有的范围）
    /// @param face    面编号
    /// @param LAP     虚网格层数
    /// @param NX,NY,NZ 物理域尺寸
    /// @param info    DMDALocalInfo，用于裁剪到当前 rank 的局部范围
    /// @param i_min... 输出：该面 ghost 区域索引范围（已裁剪）
    static void GetGhostRange(int face, int LAP,
                              int NX, int NY, int NZ,
                              const DMDALocalInfo& info,
                              int& i_min, int& i_max,
                              int& j_min, int& j_max,
                              int& k_min, int& k_max);

    /// @brief 导出 ghost 数据到 Tecplot（调试用）
    static PetscErrorCode ExportGhostToTecplot(Mesh* mesh,
                                               const std::string& base_filename);

    /// @brief 返回该填充器的名称（用于调试输出）
    virtual const char* name() const = 0;

    /// @brief 派生类实现此函数，对某个面的 ghost 区域赋值
    /// @param mesh     当前网格块
    /// @param face     面编号（0~5），对应 FaceID
    /// @param arr      局部数组指针（3D，arr[k][j][i]）
    /// @param info     DMDALocalInfo，包含索引范围信息
    /// @param all_blocks 多块信息（单块时为 nullptr）
    virtual PetscErrorCode assignGhostOnFace(Mesh* mesh,
                                             int face,
                                             PetscReal*** arr,
                                             const DMDALocalInfo& info,
                                             MultiBlockMesh* all_blocks) = 0;

    PetscInt lap_;  // 虚网格层数
};

// ====================================================================
// 雅可比/坐标系数值的 Ghost Cell 填充（策略模式包装）
// 等价 Fortran Jac_Ghost_Extent_1st 和 Jac_Ghost_Extent_2nd
//
// 根据 ghost_cell_face[face] 的值决定填充方式：
//   0 = 不填充
//   1 = 一阶外推（度量系数常数延拓 + 坐标线性外推）
//   2 = 二阶外推（坐标镜像外推 + 重算雅可比系数）
//
// 流程：fillGhostCellOnFace 对 Axx/Ayy/Azz 和各度量系数逐个调 assignGhostOnFace 外推
// ====================================================================
class JacGhostExtentBC : public GhostCellFiller
{
public:
    JacGhostExtentBC(PetscInt gtype, PetscInt lap);
    const char* name() const override { return "JacGhostExtent"; }

    /// @brief 重写 fillGhostCellOnFace，只填充指定面
    PetscErrorCode fillGhostCellOnFace(Mesh* mesh, int face,
                                        MultiBlockMesh* all_blocks = nullptr) override;
    /// @brief 对单个数组做 ghost 外推（gtype=1 一阶线性，gtype=2 二阶镜像）
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

#endif // GHOST_CELL_FILER_H
