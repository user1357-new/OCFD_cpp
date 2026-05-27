#ifndef MESH_H
#define MESH_H

#include <petscdmda.h>
#include <petscvec.h>
#include <functional>

// 网格类型常量（暂时没用）
const int GRID1D           = 10;   //→  一维网格
const int GRID2D_PLANE     = 20;   //→  二维平面网格
const int GRID2D_AXIAL_SYMM= 21;   //→  二维轴对称网格
const int GRID3D           = 30;   //→  三维网格（不计算雅可比）
const int GRID_AND_JACOBIAN3D = 31; //→ 三维网格 + 雅可比


// 数值格式常量
const int OCFD_Scheme_CD6 = 6;
const int OCFD_Scheme_CD8 = 8;

class Mesh {
private:
    MPI_Comm comm; 
    DM da;
    PetscInt my_id;
    // 全局和局部网格尺寸
    PetscInt nx_global, ny_global, nz_global;
    PetscInt nx, ny, nz;
    
    // 保留您规定的进程划分参数
    PetscInt npx0, npy0, npz0;
    
    PetscInt LAP;
    PetscInt Iflag_Gridtype;
    PetscInt Scheme_Vis;
    
    // 坐标数组
    Vec Axx, Ayy, Azz;
    
    // 雅可比系数数组
    Vec Akx, Aky, Akz;
    Vec Aix, Aiy, Aiz;
    Vec Asx, Asy, Asz;
    Vec Ajac;
    
    // 归一化的雅可比系数
    Vec Akx1, Aky1, Akz1;
    Vec Aix1, Aiy1, Aiz1;
    Vec Asx1, Asy1, Asz1;
    // ---- 统一持久化 local vector 池 ----
    // 坐标（3个）：Axx_local, Ayy_local, Azz_local
    Vec Axx_local, Ayy_local, Azz_local;
    // 度量系数（10个）：Akx_local ~ Ajac_local
    Vec Akx_local, Aky_local, Akz_local;
    Vec Aix_local, Aiy_local, Aiz_local;
    Vec Asx_local, Asy_local, Asz_local;
    Vec Ajac_local;
    bool local_pool_initialized;  // 标记池是否已创建
    
    // 边界条件处理函数指针（留空）
    std::function<void()> boundary_condition_handler;
    
    // 私有方法
    PetscErrorCode comput_Jacobian3d();
    
    // x/y/z方向导数计算（对应Fortran的OCFD_dx0/dy0/dz0）
    void compute_derivative_x(PetscReal ***f, PetscReal ***df);
    void compute_derivative_y(PetscReal ***f, PetscReal ***df);
    void compute_derivative_z(PetscReal ***f, PetscReal ***df);
    // MPI相关函数接口（留空）
    void exchange_boundary_xyz(Vec &f);
    // ---- 统一持久化 local vector 池管理 ----
    /// @brief 创建所有 13 个 local vectors（仅首次调用生效）
    PetscErrorCode ensureLocalVectors();

    /// @brief 将所有 global 向量同步到对应的持久化 local vector
    PetscErrorCode syncGlobalToLocal();

public:
    // 构造函数：保留 npx0, npy0, npz0 以便您规定分块
    Mesh(PetscInt nx_g, PetscInt ny_g, PetscInt nz_g,
     PetscInt my_id_, 
     PetscInt npx0_, PetscInt npy0_, PetscInt npz0_,
     PetscInt lap, PetscInt grid_type, PetscInt scheme_vis,
     MPI_Comm comm = PETSC_COMM_WORLD);   // 默认值保持单块兼容
    
    ~Mesh();
    PetscErrorCode InitializeFromCoordinates(
        const std::vector<PetscReal> &x_global,
        const std::vector<PetscReal> &y_global,
        const std::vector<PetscReal> &z_global);
    // 注册边界条件处理函数（您自己实现周期性边界等）
    void RegisterBoundaryConditionHandler(std::function<void()> handler);
    
    // 获取局部数组指针（用于访问数据）
    PetscErrorCode GetLocalArrays(
        PetscReal*** &Axx_arr, PetscReal*** &Ayy_arr, PetscReal*** &Azz_arr,
        PetscReal*** &Akx_arr, PetscReal*** &Aky_arr, PetscReal*** &Akz_arr,
        PetscReal*** &Aix_arr, PetscReal*** &Aiy_arr, PetscReal*** &Aiz_arr,
        PetscReal*** &Asx_arr, PetscReal*** &Asy_arr, PetscReal*** &Asz_arr,
        PetscReal*** &Ajac_arr,
        PetscReal*** &Akx1_arr, PetscReal*** &Aky1_arr, PetscReal*** &Akz1_arr,
        PetscReal*** &Aix1_arr, PetscReal*** &Aiy1_arr, PetscReal*** &Aiz1_arr,
        PetscReal*** &Asx1_arr, PetscReal*** &Asy1_arr, PetscReal*** &Asz1_arr,
        DMDALocalInfo &info
    );
    
    PetscErrorCode RestoreLocalArrays(
        PetscReal*** &Axx_arr, PetscReal*** &Ayy_arr, PetscReal*** &Azz_arr,
        PetscReal*** &Akx_arr, PetscReal*** &Aky_arr, PetscReal*** &Akz_arr,
        PetscReal*** &Aix_arr, PetscReal*** &Aiy_arr, PetscReal*** &Aiz_arr,
        PetscReal*** &Asx_arr, PetscReal*** &Asy_arr, PetscReal*** &Asz_arr,
        PetscReal*** &Ajac_arr,
        PetscReal*** &Akx1_arr, PetscReal*** &Aky1_arr, PetscReal*** &Akz1_arr,
        PetscReal*** &Aix1_arr, PetscReal*** &Aiy1_arr, PetscReal*** &Aiz1_arr,
        PetscReal*** &Asx1_arr, PetscReal*** &Asy1_arr, PetscReal*** &Asz1_arr
    );
    // ---- 统一持久化 local vector 数组访问 ----
    /// @brief 获取持久化坐标 local vector 的数组指针（含 ghost）
    PetscErrorCode getLocalCoordinateArrays(PetscReal*** &axx, PetscReal*** &ayy, PetscReal*** &azz,
                                            DMDALocalInfo &info);
    PetscErrorCode restoreLocalCoordinateArrays(PetscReal*** &axx, PetscReal*** &ayy, PetscReal*** &azz);
    
    /// @brief 获取持久化度量系数 local vector 的数组指针（含 ghost）
    PetscErrorCode getLocalMetricArrays(PetscReal*** &akx, PetscReal*** &aky, PetscReal*** &akz,
                                        PetscReal*** &aix, PetscReal*** &aiy, PetscReal*** &aiz,
                                        PetscReal*** &asx, PetscReal*** &asy, PetscReal*** &asz,
                                        PetscReal*** &ajac, DMDALocalInfo &info);
    PetscErrorCode restoreLocalMetricArrays(PetscReal*** &akx, PetscReal*** &aky, PetscReal*** &akz,
                                            PetscReal*** &aix, PetscReal*** &aiy, PetscReal*** &aiz,
                                            PetscReal*** &asx, PetscReal*** &asy, PetscReal*** &asz,
                                            PetscReal*** &ajac);

    // ---- 统一 ghost 填充入口：暴露持久化 local Vec ----
    /// @brief 获取持久化坐标 local Vec 列表
    std::vector<Vec> getLocalCoordinateVecs() const;
    /// @brief 获取持久化度量系数 local Vec 列表（10个）
    std::vector<Vec> getLocalMetricVecs() const;

    /// @brief 填充所有持久化数组的边角虚网格
    /// @param coord_gtype  坐标外推类型（1=线性, 2=镜像）
    /// @param metric_gtype 度量系数外推类型（1=线性, 2=镜像）
    PetscErrorCode fillAllEdgeAndCornerGhost(int coord_gtype, int metric_gtype);
    
    /// @brief 检查持久化 local vector 池是否已初始化
    bool hasLocalPool() const { return local_pool_initialized; }
    
    // 打印网格信息
    void printInfo() const;
    
    PetscErrorCode ExportToTecplot(const std::string &filename);
    
    // 获取网格尺寸
    PetscInt getNx() const { return nx; }
    PetscInt getNy() const { return ny; }
    PetscInt getNz() const { return nz; }
    PetscInt getNxGlobal() const { return nx_global; }
    PetscInt getNyGlobal() const { return ny_global; }
    PetscInt getNzGlobal() const { return nz_global; }
    PetscInt getLAP() const { return LAP; }

    // 获取内部 PETSc 对象
    DM getDM() const { return da; }
    MPI_Comm getComm() const { return comm; }
    Vec getAxx() const { return Axx; }
    Vec& getAxxRef() { return Axx; }
    Vec getAyy() const { return Ayy; }
    Vec& getAyyRef() { return Ayy; }
    Vec getAzz() const { return Azz; }
    Vec& getAzzRef() { return Azz; }
    Vec getAkx() const { return Akx; }
    Vec getAky() const { return Aky; }
    Vec getAkz() const { return Akz; }
    Vec getAix() const { return Aix; }
    Vec getAiy() const { return Aiy; }
    Vec getAiz() const { return Aiz; }
    Vec getAsx() const { return Asx; }
    Vec getAsy() const { return Asy; }
    Vec getAsz() const { return Asz; }
    Vec getAjac() const { return Ajac; }
    bool hasCoordinates() const { return Axx != nullptr; }
};

#endif // MESH_H