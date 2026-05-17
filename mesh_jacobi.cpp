#include "mesh.h"
#include <petscdmda.h>
#include <petscvec.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <string> 
// x方向导数计算
// f  : 来自 DMDAVecGetArray，用全局索引（含 ghost）
// df : 手动分配，用 0-based 局部索引 [kl][jl][il]
void Mesh::compute_derivative_x(PetscReal ***f, PetscReal ***df)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    
    PetscReal hx = 1.0 / (PetscReal)(nx_global - 1);
    
    // CD6系数
    PetscReal a1 = 1.0 / (60.0 * hx);
    PetscReal a2 = -3.0 / (20.0 * hx);
    PetscReal a3 = 3.0 / (4.0 * hx);
    
    // CD8系数
    PetscReal c1 = 0.8 / hx;
    PetscReal c2 = -0.2 / hx;
    PetscReal c3 = 3.80952380952380952e-2 / hx;
    PetscReal c4 = -3.571428571428571428e-3 / hx;
    
    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;
    
    // 遍历所有局部网格点，根据距离边界的层数自动选择差分格式
    for (PetscInt k = zs; k < zs + zm; k++) {
        PetscInt kl = k - zs;
        for (PetscInt j = ys; j < ys + ym; j++) {
            PetscInt jl = j - ys;
            for (PetscInt i = xs; i < xs + xm; i++) {
                PetscInt il = i - xs;
                if (i < 0 || i >= nx_global) {
                    continue;
                }
                // 计算到左右边界的最小距离
                int dist_to_left = i;
                int dist_to_right = nx_global - 1 - i;
                int min_dist = (dist_to_left < dist_to_right) ? dist_to_left : dist_to_right;
                
                if (min_dist == 0) {
                    // 第1层：边界点，使用单侧2阶差分
                    if (i == 0) {
                        df[kl][jl][il] = (-3.0 * f[k][j][i] + 4.0 * f[k][j][i+1] - f[k][j][i+2]) / (2.0 * hx);
                    } else {
                        df[kl][jl][il] = (f[k][j][i-2] - 4.0 * f[k][j][i-1] + 3.0 * f[k][j][i]) / (2.0 * hx);
                    }
                } else if (min_dist == 1) {
                    // 第2层：使用中心2阶差分
                    df[kl][jl][il] = (f[k][j][i+1] - f[k][j][i-1]) / (2.0 * hx);
                } else if (min_dist == 2) {
                    // 第3层：使用4阶中心差分
                    df[kl][jl][il] = (8.0 * (f[k][j][i+1] - f[k][j][i-1]) 
                                  - (f[k][j][i+2] - f[k][j][i-2])) / (12.0 * hx);
                } else {
                    // 第4层及以上：使用高阶中心差分
                    if (Scheme_Vis == OCFD_Scheme_CD6) {
                        df[kl][jl][il] = a1 * (f[k][j][i+3] - f[k][j][i-3]) 
                                    + a2 * (f[k][j][i+2] - f[k][j][i-2]) 
                                    + a3 * (f[k][j][i+1] - f[k][j][i-1]);
                    } else if (Scheme_Vis == OCFD_Scheme_CD8) {
                        df[kl][jl][il] = c1 * (f[k][j][i+1] - f[k][j][i-1]) 
                                    + c2 * (f[k][j][i+2] - f[k][j][i-2]) 
                                    + c3 * (f[k][j][i+3] - f[k][j][i-3]) 
                                    + c4 * (f[k][j][i+4] - f[k][j][i-4]);
                    }
                }
            }
        }
    }
}
// y方向导数计算
// f  : 来自 DMDAVecGetArray，用全局索引（含 ghost）
// df : 手动分配，用 0-based 局部索引 [kl][jl][il]
void Mesh::compute_derivative_y(PetscReal ***f, PetscReal ***df)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    
    PetscReal hy = 1.0 / (PetscReal)(ny_global - 1);
    
    // CD6系数
    PetscReal a1 = 1.0 / (60.0 * hy);
    PetscReal a2 = -3.0 / (20.0 * hy);
    PetscReal a3 = 3.0 / (4.0 * hy);
    
    // CD8系数
    PetscReal c1 = 0.8 / hy;
    PetscReal c2 = -0.2 / hy;
    PetscReal c3 = 3.80952380952380952e-2 / hy;
    PetscReal c4 = -3.571428571428571428e-3 / hy;
    
    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;
    
    // 遍历所有局部网格点，根据距离边界的层数自动选择差分格式
    for (PetscInt k = zs; k < zs + zm; k++) {
        PetscInt kl = k - zs;
        for (PetscInt j = ys; j < ys + ym; j++) {
            PetscInt jl = j - ys;
            for (PetscInt i = xs; i < xs + xm; i++) {
                PetscInt il = i - xs;
                if (j < 0 || j >= ny_global) {
                    continue;
                }
                // 计算到上下边界的最小距离
                int dist_to_bottom = j;
                int dist_to_top = ny_global - 1 - j;
                int min_dist = (dist_to_bottom < dist_to_top) ? dist_to_bottom : dist_to_top;
                
                if (min_dist == 0) {
                    // 第1层：边界点，使用单侧2阶差分
                    if (j == 0) {
                        df[kl][jl][il] = (-3.0 * f[k][j][i] + 4.0 * f[k][j+1][i] - f[k][j+2][i]) / (2.0 * hy);
                    } else {
                        df[kl][jl][il] = (f[k][j-2][i] - 4.0 * f[k][j-1][i] + 3.0 * f[k][j][i]) / (2.0 * hy);
                    }
                } else if (min_dist == 1) {
                    // 第2层：使用中心2阶差分
                    df[kl][jl][il] = (f[k][j+1][i] - f[k][j-1][i]) / (2.0 * hy);
                } else if (min_dist == 2) {
                    // 第3层：使用4阶中心差分
                    df[kl][jl][il] = (8.0 * (f[k][j+1][i] - f[k][j-1][i]) 
                                  - (f[k][j+2][i] - f[k][j-2][i])) / (12.0 * hy);
                } else {
                    // 第4层及以上：使用高阶中心差分
                    if (Scheme_Vis == OCFD_Scheme_CD6) {
                        df[kl][jl][il] = a1 * (f[k][j+3][i] - f[k][j-3][i]) 
                                    + a2 * (f[k][j+2][i] - f[k][j-2][i]) 
                                    + a3 * (f[k][j+1][i] - f[k][j-1][i]);
                    } else if (Scheme_Vis == OCFD_Scheme_CD8) {
                        df[kl][jl][il] = c1 * (f[k][j+1][i] - f[k][j-1][i]) 
                                    + c2 * (f[k][j+2][i] - f[k][j-2][i]) 
                                    + c3 * (f[k][j+3][i] - f[k][j-3][i]) 
                                    + c4 * (f[k][j+4][i] - f[k][j-4][i]);
                    }
                }
            }
        }
    }
}
// z方向导数计算
// f  : 来自 DMDAVecGetArray，用全局索引（含 ghost）
// df : 手动分配，用 0-based 局部索引 [kl][jl][il]
void Mesh::compute_derivative_z(PetscReal ***f, PetscReal ***df)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);
    
    PetscReal hz = 1.0 / (PetscReal)(nz_global - 1);
    
    // CD6系数
    PetscReal a1 = 1.0 / (60.0 * hz);
    PetscReal a2 = -3.0 / (20.0 * hz);
    PetscReal a3 = 3.0 / (4.0 * hz);
    
    // CD8系数
    PetscReal c1 = 0.8 / hz;
    PetscReal c2 = -0.2 / hz;
    PetscReal c3 = 3.80952380952380952e-2 / hz;
    PetscReal c4 = -3.571428571428571428e-3 / hz;
    
    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;
    
    // 遍历所有局部网格点，根据距离边界的层数自动选择差分格式
    for (PetscInt k = zs; k < zs + zm; k++) {
        PetscInt kl = k - zs;
        for (PetscInt j = ys; j < ys + ym; j++) {
            PetscInt jl = j - ys;
            for (PetscInt i = xs; i < xs + xm; i++) {
                PetscInt il = i - xs;
                if (k < 0 || k >= nz_global) {
                    continue;
                }
                // 计算到前后边界的最小距离
                int dist_to_front = k;
                int dist_to_back = nz_global - 1 - k;
                int min_dist = (dist_to_front < dist_to_back) ? dist_to_front : dist_to_back;
                
                if (min_dist == 0) {
                    // 第1层：边界点，使用单侧2阶差分
                    if (k == 0) {
                        df[kl][jl][il] = (-3.0 * f[k][j][i] + 4.0 * f[k+1][j][i] - f[k+2][j][i]) / (2.0 * hz);
                    } else {
                        df[kl][jl][il] = (f[k-2][j][i] - 4.0 * f[k-1][j][i] + 3.0 * f[k][j][i]) / (2.0 * hz);
                    }
                } else if (min_dist == 1) {
                    // 第2层：使用中心2阶差分
                    df[kl][jl][il] = (f[k+1][j][i] - f[k-1][j][i]) / (2.0 * hz);
                } else if (min_dist == 2) {
                    // 第3层：使用4阶中心差分
                    df[kl][jl][il] = (8.0 * (f[k+1][j][i] - f[k-1][j][i]) 
                                  - (f[k+2][j][i] - f[k-2][j][i])) / (12.0 * hz);
                } else {
                    // 第4层及以上：使用高阶中心差分
                    if (Scheme_Vis == OCFD_Scheme_CD6) {
                        df[kl][jl][il] = a1 * (f[k+3][j][i] - f[k-3][j][i]) 
                                    + a2 * (f[k+2][j][i] - f[k-2][j][i]) 
                                    + a3 * (f[k+1][j][i] - f[k-1][j][i]);
                    } else if (Scheme_Vis == OCFD_Scheme_CD8) {
                        df[kl][jl][il] = c1 * (f[k+1][j][i] - f[k-1][j][i]) 
                                    + c2 * (f[k+2][j][i] - f[k-2][j][i]) 
                                    + c3 * (f[k+3][j][i] - f[k-3][j][i]) 
                                    + c4 * (f[k+4][j][i] - f[k-4][j][i]);
                    }
                }
            }
        }
    }
}
//计算jacobi矩阵
PetscErrorCode Mesh::comput_Jacobian3d()
{
    PetscErrorCode ierr;

    // 1. 创建雅可比系数向量（全局向量，无 ghost）
    ierr = DMCreateGlobalVector(da, &Akx); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aky); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Akz); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aix); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aiy); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aiz); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Asx); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Asy); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Asz); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Ajac); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Akx1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aky1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Akz1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aix1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aiy1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Aiz1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Asx1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Asy1); CHKERRQ(ierr);
    ierr = VecDuplicate(Akx, &Asz1); CHKERRQ(ierr);

    // 2. 创建局部向量并填充 ghost（用来安全地做差分）
    Vec Axx_local, Ayy_local, Azz_local;
    ierr = DMCreateLocalVector(da, &Axx_local); CHKERRQ(ierr);
    ierr = DMCreateLocalVector(da, &Ayy_local); CHKERRQ(ierr);
    ierr = DMCreateLocalVector(da, &Azz_local); CHKERRQ(ierr);

    ierr = DMGlobalToLocalBegin(da, Axx, INSERT_VALUES, Axx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Axx, INSERT_VALUES, Axx_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Ayy, INSERT_VALUES, Ayy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Ayy, INSERT_VALUES, Ayy_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalBegin(da, Azz, INSERT_VALUES, Azz_local); CHKERRQ(ierr);
    ierr = DMGlobalToLocalEnd(da, Azz, INSERT_VALUES, Azz_local); CHKERRQ(ierr);

    // 3. 获取局部数组指针（包含 ghost，可用全局索引访问，PETSc 的 C 接口保证这一点）
    PetscReal ***axx, ***ayy, ***azz;
    ierr = DMDAVecGetArray(da, Axx_local, &axx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy_local, &ayy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz_local, &azz); CHKERRQ(ierr);

    // 4. 获取全局系数数组指针（只写物理点）
    PetscReal ***akx, ***aky, ***akz;
    PetscReal ***aix, ***aiy, ***aiz;
    PetscReal ***asx, ***asy, ***asz;
    PetscReal ***akx1, ***aky1, ***akz1;
    PetscReal ***aix1, ***aiy1, ***aiz1;
    PetscReal ***asx1, ***asy1, ***asz1;
    PetscReal ***ajac;

    ierr = DMDAVecGetArray(da, Akx, &akx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aky, &aky); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akz, &akz); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aix, &aix); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiy, &aiy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiz, &aiz); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asx, &asx); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asy, &asy); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asz, &asz); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akx1, &akx1); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aky1, &aky1); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Akz1, &akz1); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aix1, &aix1); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiy1, &aiy1); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Aiz1, &aiz1); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asx1, &asx1); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asy1, &asy1); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Asz1, &asz1); CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ajac, &ajac); CHKERRQ(ierr);

    // 5. 获取局部网格范围（全局索引的起止点）
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info); CHKERRQ(ierr);
    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    // 6. 分配临时导数数组，大小 = zm × ym × xm，全部用局部索引 [kl][jl][il] 访问
    //   xi[kl][jl][il] = ∂x/∂ξ 在 (i=xs+il, j=ys+jl, k=zs+kl) 处的值
    //   f 数组（来自 DMDAVecGetArray）用全局索引不变
    PetscReal ***xi, ***xj, ***xk;
    PetscReal ***yi, ***yj, ***yk;
    PetscReal ***zi, ***zj, ***zk;

    ierr = PetscMalloc1(zm, &xi); CHKERRQ(ierr);
    ierr = PetscMalloc1(zm, &xj); CHKERRQ(ierr);
    ierr = PetscMalloc1(zm, &xk); CHKERRQ(ierr);
    ierr = PetscMalloc1(zm, &yi); CHKERRQ(ierr);
    ierr = PetscMalloc1(zm, &yj); CHKERRQ(ierr);
    ierr = PetscMalloc1(zm, &yk); CHKERRQ(ierr);
    ierr = PetscMalloc1(zm, &zi); CHKERRQ(ierr);
    ierr = PetscMalloc1(zm, &zj); CHKERRQ(ierr);
    ierr = PetscMalloc1(zm, &zk); CHKERRQ(ierr);

    for (PetscInt kl = 0; kl < zm; kl++) {
        ierr = PetscMalloc1(ym, &xi[kl]); CHKERRQ(ierr);
        ierr = PetscMalloc1(ym, &xj[kl]); CHKERRQ(ierr);
        ierr = PetscMalloc1(ym, &xk[kl]); CHKERRQ(ierr);
        ierr = PetscMalloc1(ym, &yi[kl]); CHKERRQ(ierr);
        ierr = PetscMalloc1(ym, &yj[kl]); CHKERRQ(ierr);
        ierr = PetscMalloc1(ym, &yk[kl]); CHKERRQ(ierr);
        ierr = PetscMalloc1(ym, &zi[kl]); CHKERRQ(ierr);
        ierr = PetscMalloc1(ym, &zj[kl]); CHKERRQ(ierr);
        ierr = PetscMalloc1(ym, &zk[kl]); CHKERRQ(ierr);

        for (PetscInt jl = 0; jl < ym; jl++) {
            ierr = PetscMalloc1(xm, &xi[kl][jl]); CHKERRQ(ierr);
            ierr = PetscMalloc1(xm, &xj[kl][jl]); CHKERRQ(ierr);
            ierr = PetscMalloc1(xm, &xk[kl][jl]); CHKERRQ(ierr);
            ierr = PetscMalloc1(xm, &yi[kl][jl]); CHKERRQ(ierr);
            ierr = PetscMalloc1(xm, &yj[kl][jl]); CHKERRQ(ierr);
            ierr = PetscMalloc1(xm, &yk[kl][jl]); CHKERRQ(ierr);
            ierr = PetscMalloc1(xm, &zi[kl][jl]); CHKERRQ(ierr);
            ierr = PetscMalloc1(xm, &zj[kl][jl]); CHKERRQ(ierr);
            ierr = PetscMalloc1(xm, &zk[kl][jl]); CHKERRQ(ierr);
        }
    }

    // 7. 计算导数
    //   f 参数：来自 DMDAVecGetArray，用全局索引（含 ghost）
    //   df 参数：手动分配的临时数组，用 0-based 局部索引
    //   compute_derivative_* 写入时自动使用局部索引
    compute_derivative_x(axx, xi);
    compute_derivative_x(ayy, yi);
    compute_derivative_x(azz, zi);

    compute_derivative_y(axx, xj);
    compute_derivative_y(ayy, yj);
    compute_derivative_y(azz, zj);

    compute_derivative_z(axx, xk);
    compute_derivative_z(ayy, yk);
    compute_derivative_z(azz, zk);

    // 8. 计算雅可比系数
    //   ajac/akx/... 来自 DMDAVecGetArray → 全局索引
    //   xi/xj/... 是局部临时数组 → 局部索引 [kl][jl][il]
    for (PetscInt k = zs; k < zs + zm; k++) {
        PetscInt kl = k - zs;
        for (PetscInt j = ys; j < ys + ym; j++) {
            PetscInt jl = j - ys;
            for (PetscInt i = xs; i < xs + xm; i++) {
                PetscInt il = i - xs;

                PetscReal xi1 = xi[kl][jl][il];
                PetscReal xj1 = xj[kl][jl][il];
                PetscReal xk1 = xk[kl][jl][il];
                PetscReal yi1 = yi[kl][jl][il];
                PetscReal yj1 = yj[kl][jl][il];
                PetscReal yk1 = yk[kl][jl][il];
                PetscReal zi1 = zi[kl][jl][il];
                PetscReal zj1 = zj[kl][jl][il];
                PetscReal zk1 = zk[kl][jl][il];

                PetscReal Jac1 = 1.0 / (xi1*yj1*zk1 + yi1*zj1*xk1 + zi1*xj1*yk1
                                      - zi1*yj1*xk1 - yi1*xj1*zk1 - xi1*zj1*yk1);

                ajac[k][j][i] = Jac1;

                akx[k][j][i] = Jac1 * (yj1*zk1 - zj1*yk1);
                aky[k][j][i] = Jac1 * (zj1*xk1 - xj1*zk1);
                akz[k][j][i] = Jac1 * (xj1*yk1 - yj1*xk1);
                aix[k][j][i] = Jac1 * (yk1*zi1 - zk1*yi1);
                aiy[k][j][i] = Jac1 * (zk1*xi1 - xk1*zi1);
                aiz[k][j][i] = Jac1 * (xk1*yi1 - yk1*xi1);
                asx[k][j][i] = Jac1 * (yi1*zj1 - zi1*yj1);
                asy[k][j][i] = Jac1 * (zi1*xj1 - xi1*zj1);
                asz[k][j][i] = Jac1 * (xi1*yj1 - yi1*xj1);

                akx1[k][j][i] = akx[k][j][i] / ajac[k][j][i];
                aky1[k][j][i] = aky[k][j][i] / ajac[k][j][i];
                akz1[k][j][i] = akz[k][j][i] / ajac[k][j][i];
                aix1[k][j][i] = aix[k][j][i] / ajac[k][j][i];
                aiy1[k][j][i] = aiy[k][j][i] / ajac[k][j][i];
                aiz1[k][j][i] = aiz[k][j][i] / ajac[k][j][i];
                asx1[k][j][i] = asx[k][j][i] / ajac[k][j][i];
                asy1[k][j][i] = asy[k][j][i] / ajac[k][j][i];
                asz1[k][j][i] = asz[k][j][i] / ajac[k][j][i];

                if (Jac1 < 0) {
                    PetscPrintf(comm,
                               " Jacobian < 0 !!! , Jac=%f\n", Jac1);
                }
            }
        }
    }

    // 9. 释放临时导数数组（顺序与分配相反）
    for (PetscInt kl = 0; kl < zm; kl++) {
        for (PetscInt jl = 0; jl < ym; jl++) {
            PetscFree(xi[kl][jl]);
            PetscFree(xj[kl][jl]);
            PetscFree(xk[kl][jl]);
            PetscFree(yi[kl][jl]);
            PetscFree(yj[kl][jl]);
            PetscFree(yk[kl][jl]);
            PetscFree(zi[kl][jl]);
            PetscFree(zj[kl][jl]);
            PetscFree(zk[kl][jl]);
        }
        PetscFree(xi[kl]);
        PetscFree(xj[kl]);
        PetscFree(xk[kl]);
        PetscFree(yi[kl]);
        PetscFree(yj[kl]);
        PetscFree(yk[kl]);
        PetscFree(zi[kl]);
        PetscFree(zj[kl]);
        PetscFree(zk[kl]);
    }
    PetscFree(xi);
    PetscFree(xj);
    PetscFree(xk);
    PetscFree(yi);
    PetscFree(yj);
    PetscFree(yk);
    PetscFree(zi);
    PetscFree(zj);
    PetscFree(zk);

    // 10. 释放数组指针
    ierr = DMDAVecRestoreArray(da, Axx_local, &axx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy_local, &ayy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz_local, &azz); CHKERRQ(ierr);

    ierr = DMDAVecRestoreArray(da, Akx, &akx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aky, &aky); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Akz, &akz); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aix, &aix); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiy, &aiy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiz, &aiz); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asx, &asx); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asy, &asy); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asz, &asz); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Akx1, &akx1); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aky1, &aky1); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Akz1, &akz1); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aix1, &aix1); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiy1, &aiy1); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Aiz1, &aiz1); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asx1, &asx1); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asy1, &asy1); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Asz1, &asz1); CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ajac, &ajac); CHKERRQ(ierr);

    // 11. 销毁局部向量
    VecDestroy(&Axx_local);
    VecDestroy(&Ayy_local);
    VecDestroy(&Azz_local);

    return 0;
}
