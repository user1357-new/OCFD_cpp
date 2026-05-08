#ifndef MESH_H
#define MESH_H

#include <petscdmda.h>
#include <petscvec.h>
#include <functional>

// 网格类型常量
const int GRID1D = 10;
const int GRID2D_PLANE = 20;
const int GRID2D_AXIAL_SYMM = 21;
const int GRID3D = 30;
const int GRID_AND_JACOBIAN3D = 31;

// 数值格式常量
const int OCFD_Scheme_CD6 = 6;
const int OCFD_Scheme_CD8 = 8;

class Mesh {
private:
    MPI_Comm comm; 
    DM da;
    
    // 全局和局部网格尺寸
    PetscInt nx_global, ny_global, nz_global;
    PetscInt nx, ny, nz;
    
    // MPI进程信息
    PetscInt my_id;
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
    
    // 边界条件处理函数指针（您自己实现）
    std::function<void()> boundary_condition_handler;
    
    // 私有方法
    PetscErrorCode comput_Jacobian3d();
    
    // x/y/z方向导数计算（对应Fortran的OCFD_dx0/dy0/dz0）
    void compute_derivative_x(PetscReal ***f, PetscReal ***df);
    void compute_derivative_y(PetscReal ***f, PetscReal ***df);
    void compute_derivative_z(PetscReal ***f, PetscReal ***df);
    
    // MPI相关函数接口（留空给您实现）
    void exchange_boundary_xyz(Vec &f);
    PetscErrorCode Jac_Ghost_boundary();

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
};

#endif // MESH_H