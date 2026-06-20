#include "mesh.h"
#include "mesh_mutiblock.h"
#include <petscdmda.h>
#include <petscvec.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <string>

Mesh::Mesh(PetscInt nx_g, PetscInt ny_g, PetscInt nz_g,
           PetscInt my_id_, PetscInt npx0_, PetscInt npy0_, PetscInt npz0_,
           PetscInt lap, PetscInt grid_type, PetscInt scheme_vis,
           PetscInt metric_diff_type,
           MPI_Comm comm_)
    : comm(comm_),
      da(nullptr),
      nx_global(nx_g), ny_global(ny_g), nz_global(nz_g),
      nx(0), ny(0), nz(0),
      npx0(npx0_), npy0(npy0_), npz0(npz0_),
      LAP(lap),
      Iflag_Gridtype(grid_type),
      Scheme_Vis(scheme_vis),
      metric_diff_type_(metric_diff_type),
      Axx(nullptr), Ayy(nullptr), Azz(nullptr),
      Axx_local(nullptr), Ayy_local(nullptr), Azz_local(nullptr),
      Akx(nullptr), Aky(nullptr), Akz(nullptr),
      Aix(nullptr), Aiy(nullptr), Aiz(nullptr),
      Asx(nullptr), Asy(nullptr), Asz(nullptr),
      Ajac(nullptr),
      Akx_local(nullptr), Aky_local(nullptr), Akz_local(nullptr),
      Aix_local(nullptr), Aiy_local(nullptr), Aiz_local(nullptr),
      Asx_local(nullptr), Asy_local(nullptr), Asz_local(nullptr),
      Ajac_local(nullptr),
      cell_pool_initialized(false),
      boundary_condition_handler(nullptr)
{
    for (int i = 0; i < 5; ++i) {
        U[i]       = nullptr;
        U_local[i] = nullptr;
    }
}

Mesh::~Mesh()
{
    if (Axx_local) { VecDestroy(&Axx_local); Axx_local = nullptr; }
    if (Ayy_local) { VecDestroy(&Ayy_local); Ayy_local = nullptr; }
    if (Azz_local) { VecDestroy(&Azz_local); Azz_local = nullptr; }
    if (Axx) VecDestroy(&Axx);
    if (Ayy) VecDestroy(&Ayy);
    if (Azz) VecDestroy(&Azz);
    if (Akx_local) { VecDestroy(&Akx_local); Akx_local = nullptr; }
    if (Aky_local) { VecDestroy(&Aky_local); Aky_local = nullptr; }
    if (Akz_local) { VecDestroy(&Akz_local); Akz_local = nullptr; }
    if (Aix_local) { VecDestroy(&Aix_local); Aix_local = nullptr; }
    if (Aiy_local) { VecDestroy(&Aiy_local); Aiy_local = nullptr; }
    if (Aiz_local) { VecDestroy(&Aiz_local); Aiz_local = nullptr; }
    if (Asx_local) { VecDestroy(&Asx_local); Asx_local = nullptr; }
    if (Asy_local) { VecDestroy(&Asy_local); Asy_local = nullptr; }
    if (Asz_local) { VecDestroy(&Asz_local); Asz_local = nullptr; }
    if (Ajac_local) { VecDestroy(&Ajac_local); Ajac_local = nullptr; }
    for (int i = 0; i < 5; ++i) {
        if (U_local[i]) { VecDestroy(&U_local[i]); U_local[i] = nullptr; }
    }
    if (Akx) VecDestroy(&Akx);
    if (Aky) VecDestroy(&Aky);
    if (Akz) VecDestroy(&Akz);
    if (Aix) VecDestroy(&Aix);
    if (Aiy) VecDestroy(&Aiy);
    if (Aiz) VecDestroy(&Aiz);
    if (Asx) VecDestroy(&Asx);
    if (Asy) VecDestroy(&Asy);
    if (Asz) VecDestroy(&Asz);
    if (Ajac) VecDestroy(&Ajac);
    for (int i = 0; i < 5; ++i) {
        if (U[i]) VecDestroy(&U[i]);
    }
    if (da) DMDestroy(&da);
}

// ====================================================================
// InitializeFromCoordinates — 只有格心 DMDA，直接填格心坐标
// ====================================================================
PetscErrorCode Mesh::InitializeFromCoordinates(
    const std::vector<PetscReal> &x_global,
    const std::vector<PetscReal> &y_global,
    const std::vector<PetscReal> &z_global)
{
    PetscErrorCode ierr;

    ierr = DMDACreate3d(comm,
                        DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED, DM_BOUNDARY_GHOSTED,
                        DMDA_STENCIL_BOX,
                        nx_global, ny_global, nz_global,
                        npx0, npy0, npz0,
                        1, LAP,
                        NULL, NULL, NULL,
                        &da); CHKERRQ(ierr);
    ierr = DMSetUp(da); CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da, &Axx); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Ayy); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Azz); CHKERRQ(ierr);

    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
    nx = info.xm; ny = info.ym; nz = info.zm;

    PetscReal ***axx, ***ayy, ***azz;
    ierr = DMDAVecGetArray(da, Axx, &axx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &ayy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &azz); CHKERRQ(ierr);

    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                PetscInt idx = (k * ny_global + j) * nx_global + i;
                axx[k][j][i] = x_global[idx];
                ayy[k][j][i] = y_global[idx];
                azz[k][j][i] = z_global[idx];
            }
        }
    }

    ierr = DMDAVecRestoreArray(da, Axx, &axx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy, &ayy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz, &azz); CHKERRQ(ierr);



    // 度量系数全局 Vec（和坐标同一个 DMDA）
    ierr = VecDuplicate(Axx, &Akx); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Aky); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Akz); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Aix); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Aiy); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Aiz); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Asx); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Asy); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Asz); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Ajac); CHKERRQ(ierr);

    for (int i = 0; i < 5; ++i) {
        ierr = VecDuplicate(Axx, &U[i]); CHKERRQ(ierr);
    }
    ierr = ensureCellLocalVectors(); CHKERRQ(ierr);
    ierr = syncCellGlobalToLocal(); CHKERRQ(ierr);
    PetscPrintf(comm, "Initial Mesh OK: cell %d×%d×%d\n",
                (int)nx_global, (int)ny_global, (int)nz_global);
    return 0;
}

// ====================================================================
// ensureCellLocalVectors
// ====================================================================
PetscErrorCode Mesh::ensureCellLocalVectors()
{
    PetscErrorCode ierr;
    if (cell_pool_initialized) return 0;

    ierr = DMCreateLocalVector(da, &Axx_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Ayy_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Azz_local); CHKERRQ(ierr);

    ierr = VecDuplicate(Axx_local, &Akx_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Aky_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Akz_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Aix_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Aiy_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Aiz_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Asx_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Asy_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Asz_local); CHKERRQ(ierr);
    ierr = VecDuplicate(Axx_local, &Ajac_local); CHKERRQ(ierr);

    for (int i = 0; i < 5; ++i) {
        ierr = VecDuplicate(Axx_local, &U_local[i]); CHKERRQ(ierr);
    }

    cell_pool_initialized = true;
    return 0;
}

// ====================================================================
// syncCellGlobalToLocal
// ====================================================================
PetscErrorCode Mesh::syncCellGlobalToLocal()
{
    PetscErrorCode ierr;
    if (!cell_pool_initialized) {
        ierr = ensureCellLocalVectors(); CHKERRQ(ierr);
    }
    ierr = DMGlobalToLocalBegin(da, Axx, INSERT_VALUES, Axx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Axx, INSERT_VALUES, Axx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Ayy, INSERT_VALUES, Ayy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Ayy, INSERT_VALUES, Ayy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Azz, INSERT_VALUES, Azz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Azz, INSERT_VALUES, Azz_local); CHKERRQ(ierr);

    ierr = DMGlobalToLocalBegin(da, Akx, INSERT_VALUES, Akx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Akx, INSERT_VALUES, Akx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Aky, INSERT_VALUES, Aky_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Aky, INSERT_VALUES, Aky_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Akz, INSERT_VALUES, Akz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Akz, INSERT_VALUES, Akz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Aix, INSERT_VALUES, Aix_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Aix, INSERT_VALUES, Aix_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Aiy, INSERT_VALUES, Aiy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Aiy, INSERT_VALUES, Aiy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Aiz, INSERT_VALUES, Aiz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Aiz, INSERT_VALUES, Aiz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Asx, INSERT_VALUES, Asx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Asx, INSERT_VALUES, Asx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Asy, INSERT_VALUES, Asy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Asy, INSERT_VALUES, Asy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Asz, INSERT_VALUES, Asz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Asz, INSERT_VALUES, Asz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Ajac, INSERT_VALUES, Ajac_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Ajac, INSERT_VALUES, Ajac_local); CHKERRQ(ierr);

    for (int i = 0; i < 5; ++i) {
        ierr = DMGlobalToLocalBegin(da, U[i], INSERT_VALUES, U_local[i]); CHKERRQ(ierr);
        ierr = DMGlobalToLocalEnd(da, U[i], INSERT_VALUES, U_local[i]); CHKERRQ(ierr);
    }
    return 0;
}

// ====================================================================
// syncUGlobalToLocal
// ====================================================================
PetscErrorCode Mesh::syncUGlobalToLocal()
{
    PetscErrorCode ierr;
    if (!cell_pool_initialized) {
        ierr = ensureCellLocalVectors(); CHKERRQ(ierr);
    }
    for (int i = 0; i < 5; ++i) {
        ierr = DMGlobalToLocalBegin(da, U[i], INSERT_VALUES, U_local[i]); CHKERRQ(ierr);
        ierr = DMGlobalToLocalEnd(da, U[i], INSERT_VALUES, U_local[i]); CHKERRQ(ierr);
    }
    return 0;
}

// ====================================================================
// syncULocalToGlobal — 把 U_local 推回 U_global（solver 每步后调用）
// ====================================================================
PetscErrorCode Mesh::syncULocalToGlobal()
{
    PetscErrorCode ierr;
    if (!cell_pool_initialized) {
        ierr = ensureCellLocalVectors(); CHKERRQ(ierr);
    }
    for (int i = 0; i < 5; ++i) {
        ierr = DMLocalToGlobalBegin(da, U_local[i], INSERT_VALUES, U[i]); CHKERRQ(ierr);
        ierr = DMLocalToGlobalEnd(da, U_local[i], INSERT_VALUES, U[i]); CHKERRQ(ierr);
    }
    return 0;
}

// ====================================================================
// getLocalCoordinateVecs
// ====================================================================
std::vector<Vec> Mesh::getLocalCoordinateVecs() const
{
    return {Axx_local, Ayy_local, Azz_local};
}

// ====================================================================
// getLocalMetricVecs
// ====================================================================
std::vector<Vec> Mesh::getLocalMetricVecs() const
{
    return {Akx_local, Aky_local, Akz_local,
            Aix_local, Aiy_local, Aiz_local,
            Asx_local, Asy_local, Asz_local,
            Ajac_local};
}

// ====================================================================
// getGlobalMetricVecs
// ====================================================================
std::vector<Vec> Mesh::getGlobalMetricVecs() const
{
    return {Akx, Aky, Akz,
            Aix, Aiy, Aiz,
            Asx, Asy, Asz,
            Ajac};
}

// ====================================================================
// getLocalUVecs
// ====================================================================
std::vector<Vec> Mesh::getLocalUVecs() const
{
    return {U_local[0], U_local[1], U_local[2], U_local[3], U_local[4]};
}

// ====================================================================
// getCellCoordinateArrays
// ====================================================================
PetscErrorCode Mesh::getCellCoordinateArrays(
    PetscReal*** &axx, PetscReal*** &ayy, PetscReal*** &azz,
    DMDALocalInfo &info)
{
    PetscErrorCode ierr;
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Axx_local, &axx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy_local, &ayy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz_local, &azz); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode Mesh::restoreCellCoordinateArrays(
    PetscReal*** &axx, PetscReal*** &ayy, PetscReal*** &azz)
{
    PetscErrorCode ierr;
    ierr = DMDAVecRestoreArray(da, Axx_local, &axx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy_local, &ayy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz_local, &azz); CHKERRQ(ierr);
    return 0;
}

// ====================================================================
// getCellMetricArrays
// ====================================================================
PetscErrorCode Mesh::getCellMetricArrays(
    PetscReal*** &akx, PetscReal*** &aky, PetscReal*** &akz,
    PetscReal*** &aix, PetscReal*** &aiy, PetscReal*** &aiz,
    PetscReal*** &asx, PetscReal*** &asy, PetscReal*** &asz,
    PetscReal*** &ajac, DMDALocalInfo &info)
{
    PetscErrorCode ierr;
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akx_local, &akx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aky_local, &aky); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akz_local, &akz); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aix_local, &aix); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiy_local, &aiy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiz_local, &aiz); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asx_local, &asx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asy_local, &asy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asz_local, &asz); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ajac_local, &ajac); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode Mesh::restoreCellMetricArrays(
    PetscReal*** &akx, PetscReal*** &aky, PetscReal*** &akz,
    PetscReal*** &aix, PetscReal*** &aiy, PetscReal*** &aiz,
    PetscReal*** &asx, PetscReal*** &asy, PetscReal*** &asz,
    PetscReal*** &ajac)
{
    PetscErrorCode ierr;
    ierr = DMDAVecRestoreArray(da, Akx_local, &akx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aky_local, &aky); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Akz_local, &akz); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aix_local, &aix); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiy_local, &aiy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiz_local, &aiz); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asx_local, &asx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asy_local, &asy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asz_local, &asz); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ajac_local, &ajac); CHKERRQ(ierr);
    return 0;
}

PetscErrorCode Mesh::getLocalUArrays(PetscReal*** u[5], DMDALocalInfo &info)
{
    PetscErrorCode ierr;
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
    for (int c = 0; c < 5; ++c) {
        ierr = DMDAVecGetArray(da, U_local[c], &u[c]); CHKERRQ(ierr);
    }
    return 0;
}

PetscErrorCode Mesh::restoreLocalUArrays(PetscReal*** u[5])
{
    PetscErrorCode ierr;
    for (int c = 0; c < 5; ++c) {
        ierr = DMDAVecRestoreArray(da, U_local[c], &u[c]); CHKERRQ(ierr);
    }
    return 0;
}
Vec Mesh::getU(int comp) const { return U[comp]; }
Vec& Mesh::getURef(int comp) { return U[comp]; }
