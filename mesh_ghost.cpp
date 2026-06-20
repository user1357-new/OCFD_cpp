#include "mesh.h"
#include <petscdmda.h>
#include <petscvec.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <string>

// ====================================================================
// 以下为格心数组的边角 ghost 填充（新增）
// ====================================================================

// 对格心数组填充边角 ghost，逻辑和格点版本完全相同，只是尺寸用格心的
static void fillEdgeCornerArray_cell(
    PetscReal*** arr,
    const DMDALocalInfo& info,
    PetscInt nxg, PetscInt nyg, PetscInt nzg,
    int gtype)
{
    PetscInt gxs = info.gxs, gxe = info.gxs + info.gxm;
    PetscInt gys = info.gys, gye = info.gys + info.gym;
    PetscInt gzs = info.gzs, gze = info.gzs + info.gzm;

    auto nGhostDirs = [&](PetscInt i, PetscInt j, PetscInt k) -> int {
        int n = 0;
        if (i < 0 || i >= nxg) n++;
        if (j < 0 || j >= nyg) n++;
        if (k < 0 || k >= nzg) n++;
        return n;
    };
    auto isPhysical = [&](PetscInt i, PetscInt j, PetscInt k) -> bool {
        return i >= 0 && i < nxg && j >= 0 && j < nyg && k >= 0 && k < nzg;
    };

    if (gtype == 1) {
        for (PetscInt k=gzs; k<gze; ++k)
            for (PetscInt j=gys; j<gye; ++j)
                for (PetscInt i=gxs; i<gxe; ++i) {
                    if (isPhysical(i,j,k)) continue;
                    if (nGhostDirs(i,j,k)!=2) continue;
                    PetscReal sum=0.0; int cnt=0;
                    if (i<0||i>=nxg){
                        PetscInt ib=(i<0)?0:nxg-1, i_inner=(i<0)?1:nxg-2;
                        PetscReal dist=(i<0)?(PetscReal)(-i):(PetscReal)(i-(nxg-1));
                        sum+=arr[k][j][ib]+dist*(arr[k][j][ib]-arr[k][j][i_inner]); cnt++;
                    }
                    if (j<0||j>=nyg){
                        PetscInt jb=(j<0)?0:nyg-1, j_inner=(j<0)?1:nyg-2;
                        PetscReal dist=(j<0)?(PetscReal)(-j):(PetscReal)(j-(nyg-1));
                        sum+=arr[k][jb][i]+dist*(arr[k][jb][i]-arr[k][j_inner][i]); cnt++;
                    }
                    if (k<0||k>=nzg){
                        PetscInt kb=(k<0)?0:nzg-1, k_inner=(k<0)?1:nzg-2;
                        PetscReal dist=(k<0)?(PetscReal)(-k):(PetscReal)(k-(nzg-1));
                        sum+=arr[kb][j][i]+dist*(arr[kb][j][i]-arr[k_inner][j][i]); cnt++;
                    }
                    arr[k][j][i]=sum/(PetscReal)cnt;
                }
        for (PetscInt k=gzs; k<gze; ++k)
            for (PetscInt j=gys; j<gye; ++j)
                for (PetscInt i=gxs; i<gxe; ++i) {
                    if (isPhysical(i,j,k)) continue;
                    if (nGhostDirs(i,j,k)!=3) continue;
                    PetscReal sum=0.0;
                    {
                        PetscInt ib=(i<0)?0:nxg-1, i_inner=(i<0)?1:nxg-2;
                        PetscReal dist=(i<0)?(PetscReal)(-i):(PetscReal)(i-(nxg-1));
                        sum+=arr[k][j][ib]+dist*(arr[k][j][ib]-arr[k][j][i_inner]);
                    }
                    {
                        PetscInt jb=(j<0)?0:nyg-1, j_inner=(j<0)?1:nyg-2;
                        PetscReal dist=(j<0)?(PetscReal)(-j):(PetscReal)(j-(nyg-1));
                        sum+=arr[k][jb][i]+dist*(arr[k][jb][i]-arr[k][j_inner][i]);
                    }
                    {
                        PetscInt kb=(k<0)?0:nzg-1, k_inner=(k<0)?1:nzg-2;
                        PetscReal dist=(k<0)?(PetscReal)(-k):(PetscReal)(k-(nzg-1));
                        sum+=arr[kb][j][i]+dist*(arr[kb][j][i]-arr[k_inner][j][i]);
                    }
                    arr[k][j][i]=sum/3.0;
                }
        return;
    }

    for (PetscInt k=gzs; k<gze; ++k)
        for (PetscInt j=gys; j<gye; ++j)
            for (PetscInt i=gxs; i<gxe; ++i) {
                if (isPhysical(i,j,k)) continue;
                if (nGhostDirs(i,j,k)!=2) continue;
                PetscReal sum=0.0; int cnt=0;
                if (i<0||i>=nxg){
                    PetscInt ib=(i<0)?0:nxg-1, im=(i<0)?-i:2*(nxg-1)-i;
                    sum+=2.0*arr[k][j][ib]-arr[k][j][im]; cnt++;
                }
                if (j<0||j>=nyg){
                    PetscInt jb=(j<0)?0:nyg-1, jm=(j<0)?-j:2*(nyg-1)-j;
                    sum+=2.0*arr[k][jb][i]-arr[k][jm][i]; cnt++;
                }
                if (k<0||k>=nzg){
                    PetscInt kb=(k<0)?0:nzg-1, km=(k<0)?-k:2*(nzg-1)-k;
                    sum+=2.0*arr[kb][j][i]-arr[km][j][i]; cnt++;
                }
                arr[k][j][i]=sum/(PetscReal)cnt;
            }
    for (PetscInt k=gzs; k<gze; ++k)
        for (PetscInt j=gys; j<gye; ++j)
            for (PetscInt i=gxs; i<gxe; ++i) {
                if (isPhysical(i,j,k)) continue;
                if (nGhostDirs(i,j,k)!=3) continue;
                PetscReal sum=0.0;
                {
                    PetscInt ib=(i<0)?0:nxg-1, im=(i<0)?-i:2*(nxg-1)-i;
                    sum+=2.0*arr[k][j][ib]-arr[k][j][im];
                }
                {
                    PetscInt jb=(j<0)?0:nyg-1, jm=(j<0)?-j:2*(nyg-1)-j;
                    sum+=2.0*arr[k][jb][i]-arr[k][jm][i];
                }
                {
                    PetscInt kb=(k<0)?0:nzg-1, km=(k<0)?-k:2*(nzg-1)-k;
                    sum+=2.0*arr[kb][j][i]-arr[km][j][i];
                }
                arr[k][j][i]=sum/3.0;
            }
}

// ====================================================================
// fillCellCoordGhost_CoordsOnly — 只填格心坐标边角 ghost
// ====================================================================
PetscErrorCode Mesh::fillCellCoordGhost_CoordsOnly(int gtype)
{
    PetscErrorCode ierr;
    DMDALocalInfo info;
    PetscReal ***axx, ***ayy, ***azz;
    ierr = getCellCoordinateArrays(axx, ayy, azz, info); CHKERRQ(ierr);
    fillEdgeCornerArray_cell(axx, info, nx_global, ny_global, nz_global, gtype);
    fillEdgeCornerArray_cell(ayy, info, nx_global, ny_global, nz_global, gtype);
    fillEdgeCornerArray_cell(azz, info, nx_global, ny_global, nz_global, gtype);
    ierr = restoreCellCoordinateArrays(axx, ayy, azz); CHKERRQ(ierr);
    return 0;
}

// ====================================================================
// fillCellCoordGhost_MetricsOnly — 只填格心度量系数边角 ghost
// ====================================================================
PetscErrorCode Mesh::fillCellCoordGhost_MetricsOnly(int gtype)
{
    PetscErrorCode ierr;
    DMDALocalInfo info;
    PetscReal ***akx, ***aky, ***akz;
    PetscReal ***aix, ***aiy, ***aiz;
    PetscReal ***asx, ***asy, ***asz;
    PetscReal ***ajac;
    ierr = getCellMetricArrays(akx,aky,akz, aix,aiy,aiz, asx,asy,asz, ajac, info); CHKERRQ(ierr);

    fillEdgeCornerArray_cell(akx, info, nx_global, ny_global, nz_global, gtype);
    fillEdgeCornerArray_cell(aky, info, nx_global, ny_global, nz_global, gtype);
    fillEdgeCornerArray_cell(akz, info, nx_global, ny_global, nz_global, gtype);
    fillEdgeCornerArray_cell(aix, info, nx_global, ny_global, nz_global, gtype);
    fillEdgeCornerArray_cell(aiy, info, nx_global, ny_global, nz_global, gtype);
    fillEdgeCornerArray_cell(aiz, info, nx_global, ny_global, nz_global, gtype);
    fillEdgeCornerArray_cell(asx, info, nx_global, ny_global, nz_global, gtype);
    fillEdgeCornerArray_cell(asy, info, nx_global, ny_global, nz_global, gtype);
    fillEdgeCornerArray_cell(asz, info, nx_global, ny_global, nz_global, gtype);
    fillEdgeCornerArray_cell(ajac, info, nx_global, ny_global, nz_global, gtype);

    ierr = restoreCellMetricArrays(akx,aky,akz, aix,aiy,aiz, asx,asy,asz, ajac); CHKERRQ(ierr);
    return 0;
}

