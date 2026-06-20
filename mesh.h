#ifndef MESH_H
#define MESH_H

#include <petscdmda.h>
#include <petscvec.h>
#include <functional>
#include <vector>

const int GRID1D           = 10;
const int GRID2D_PLANE     = 20;
const int GRID2D_AXIAL_SYMM= 21;
const int GRID3D           = 30;
const int GRID_AND_JACOBIAN3D = 31;

const int OCFD_Scheme_CD6 = 6;
const int OCFD_Scheme_CD8 = 8;

const int METRIC_DIFF_UNIFORM = 0;
const int METRIC_DIFF_REDUCED = 1;

class Mesh {
private:
    MPI_Comm comm;

    // ========== 格心 DMDA（存坐标 + 度量 + 守恒变量，所有计算在此）==========
    DM da;
    PetscInt nx_global, ny_global, nz_global;
    PetscInt nx, ny, nz;
    PetscInt npx0, npy0, npz0;
    PetscInt LAP;
    PetscInt Iflag_Gridtype;
    PetscInt Scheme_Vis;
    PetscInt metric_diff_type_;

    Vec Axx, Ayy, Azz;
    Vec Axx_local, Ayy_local, Azz_local;

    Vec Akx, Aky, Akz;
    Vec Aix, Aiy, Aiz;
    Vec Asx, Asy, Asz;
    Vec Ajac;
    Vec Akx_local, Aky_local, Akz_local;
    Vec Aix_local, Aiy_local, Aiz_local;
    Vec Asx_local, Asy_local, Asz_local;
    Vec Ajac_local;

    Vec U[5];
    Vec U_local[5];

    bool cell_pool_initialized;

    std::function<void()> boundary_condition_handler;

    PetscErrorCode ensureCellLocalVectors();
    PetscErrorCode syncCellGlobalToLocal();

public:
    void compute_derivative_x(PetscReal ***f, PetscReal ***df);
    void compute_derivative_y(PetscReal ***f, PetscReal ***df);
    void compute_derivative_z(PetscReal ***f, PetscReal ***df);

    // nx_g/ny_g/nz_g 现在是格心维度
    Mesh(PetscInt nx_g, PetscInt ny_g, PetscInt nz_g,
         PetscInt my_id_,
         PetscInt npx0_, PetscInt npy0_, PetscInt npz0_,
         PetscInt lap, PetscInt grid_type, PetscInt scheme_vis,
         PetscInt metric_diff_type,
         MPI_Comm comm = PETSC_COMM_WORLD);

    ~Mesh();

    PetscErrorCode InitializeFromCoordinates(
        const std::vector<PetscReal> &x_global,
        const std::vector<PetscReal> &y_global,
        const std::vector<PetscReal> &z_global);

    PetscErrorCode computeCellJacobianFromFilledGhosts();

    // 坐标 local vector 访问
    PetscErrorCode getCellCoordinateArrays(
        PetscReal*** &axx, PetscReal*** &ayy, PetscReal*** &azz,
        DMDALocalInfo &info);
    PetscErrorCode restoreCellCoordinateArrays(
        PetscReal*** &axx, PetscReal*** &ayy, PetscReal*** &azz);

    // 度量系数 local vector 访问
    PetscErrorCode getCellMetricArrays(
        PetscReal*** &akx, PetscReal*** &aky, PetscReal*** &akz,
        PetscReal*** &aix, PetscReal*** &aiy, PetscReal*** &aiz,
        PetscReal*** &asx, PetscReal*** &asy, PetscReal*** &asz,
        PetscReal*** &ajac, DMDALocalInfo &info);
    PetscErrorCode restoreCellMetricArrays(
        PetscReal*** &akx, PetscReal*** &aky, PetscReal*** &akz,
        PetscReal*** &aix, PetscReal*** &aiy, PetscReal*** &aiz,
        PetscReal*** &asx, PetscReal*** &asy, PetscReal*** &asz,
        PetscReal*** &ajac);

    // 守恒变量访问
    PetscErrorCode getLocalUArrays(PetscReal*** u[5], DMDALocalInfo &info);
    PetscErrorCode restoreLocalUArrays(PetscReal*** u[5]);
    PetscErrorCode syncUGlobalToLocal();
    PetscErrorCode syncULocalToGlobal();
    Vec getU(int comp) const;
    Vec& getURef(int comp);

    // ghost 填充接口 — 全部格心
    std::vector<Vec> getLocalCoordinateVecs() const;
    std::vector<Vec> getLocalMetricVecs() const;
    std::vector<Vec> getGlobalMetricVecs() const;  // 全局度量 Vec（无 ghost）
    std::vector<Vec> getLocalUVecs() const;

    PetscErrorCode fillCellCoordGhost_CoordsOnly(int gtype);
    PetscErrorCode fillCellCoordGhost_MetricsOnly(int gtype);

    // 尺寸
    PetscInt getNx()       const { return nx; }
    PetscInt getNy()       const { return ny; }
    PetscInt getNz()       const { return nz; }
    PetscInt getNxGlobal() const { return nx_global; }
    PetscInt getNyGlobal() const { return ny_global; }
    PetscInt getNzGlobal() const { return nz_global; }
    PetscInt getLAP() const { return LAP; }
    Vec getAxx() const { return Axx; }
    Vec getAyy() const { return Ayy; }
    Vec getAzz() const { return Azz; }
    PetscErrorCode ExportCellToTecplot(const std::string &filename);

    DM getDM()     const { return da; }
    MPI_Comm getComm() const { return comm; }

    bool hasCellPool() const { return cell_pool_initialized; }

    void printInfo() const;
    PetscErrorCode ExportToTecplot(const std::string &filename);
    PetscErrorCode ExportCellGhostToTecplot(const std::string &filename);
    PetscErrorCode ExportCellGhostFlowToTecplot(const std::string &filename);  // ★ 含守恒量的 ghost 输出
    PetscErrorCode ExportFlowToTecplot(const std::string &filename);

};

#endif
