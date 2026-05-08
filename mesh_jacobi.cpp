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
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
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
                        df[k][j][i] = (-3.0 * f[k][j][i] + 4.0 * f[k][j][i+1] - f[k][j][i+2]) / (2.0 * hx);
                    } else {
                        df[k][j][i] = (f[k][j][i-2] - 4.0 * f[k][j][i-1] + 3.0 * f[k][j][i]) / (2.0 * hx);
                    }
                } else if (min_dist == 1) {
                    // 第2层：使用中心2阶差分
                    df[k][j][i] = (f[k][j][i+1] - f[k][j][i-1]) / (2.0 * hx);
                } else if (min_dist == 2) {
                    // 第3层：使用4阶中心差分
                    df[k][j][i] = (8.0 * (f[k][j][i+1] - f[k][j][i-1]) 
                                  - (f[k][j][i+2] - f[k][j][i-2])) / (12.0 * hx);
                } else {
                    // 第4层及以上：使用高阶中心差分
                    if (Scheme_Vis == OCFD_Scheme_CD6) {
                        df[k][j][i] = a1 * (f[k][j][i+3] - f[k][j][i-3]) 
                                    + a2 * (f[k][j][i+2] - f[k][j][i-2]) 
                                    + a3 * (f[k][j][i+1] - f[k][j][i-1]);
                    } else if (Scheme_Vis == OCFD_Scheme_CD8) {
                        df[k][j][i] = c1 * (f[k][j][i+1] - f[k][j][i-1]) 
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
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
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
                        df[k][j][i] = (-3.0 * f[k][j][i] + 4.0 * f[k][j+1][i] - f[k][j+2][i]) / (2.0 * hy);
                    } else {
                        df[k][j][i] = (f[k][j-2][i] - 4.0 * f[k][j-1][i] + 3.0 * f[k][j][i]) / (2.0 * hy);
                    }
                } else if (min_dist == 1) {
                    // 第2层：使用中心2阶差分
                    df[k][j][i] = (f[k][j+1][i] - f[k][j-1][i]) / (2.0 * hy);
                } else if (min_dist == 2) {
                    // 第3层：使用4阶中心差分
                    df[k][j][i] = (8.0 * (f[k][j+1][i] - f[k][j-1][i]) 
                                  - (f[k][j+2][i] - f[k][j-2][i])) / (12.0 * hy);
                } else {
                    // 第4层及以上：使用高阶中心差分
                    if (Scheme_Vis == OCFD_Scheme_CD6) {
                        df[k][j][i] = a1 * (f[k][j+3][i] - f[k][j-3][i]) 
                                    + a2 * (f[k][j+2][i] - f[k][j-2][i]) 
                                    + a3 * (f[k][j+1][i] - f[k][j-1][i]);
                    } else if (Scheme_Vis == OCFD_Scheme_CD8) {
                        df[k][j][i] = c1 * (f[k][j+1][i] - f[k][j-1][i]) 
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
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
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
                        df[k][j][i] = (-3.0 * f[k][j][i] + 4.0 * f[k+1][j][i] - f[k+2][j][i]) / (2.0 * hz);
                    } else {
                        df[k][j][i] = (f[k-2][j][i] - 4.0 * f[k-1][j][i] + 3.0 * f[k][j][i]) / (2.0 * hz);
                    }
                } else if (min_dist == 1) {
                    // 第2层：使用中心2阶差分
                    df[k][j][i] = (f[k+1][j][i] - f[k-1][j][i]) / (2.0 * hz);
                } else if (min_dist == 2) {
                    // 第3层：使用4阶中心差分
                    df[k][j][i] = (8.0 * (f[k+1][j][i] - f[k-1][j][i]) 
                                  - (f[k+2][j][i] - f[k-2][j][i])) / (12.0 * hz);
                } else {
                    // 第4层及以上：使用高阶中心差分
                    if (Scheme_Vis == OCFD_Scheme_CD6) {
                        df[k][j][i] = a1 * (f[k+3][j][i] - f[k-3][j][i]) 
                                    + a2 * (f[k+2][j][i] - f[k-2][j][i]) 
                                    + a3 * (f[k+1][j][i] - f[k-1][j][i]);
                    } else if (Scheme_Vis == OCFD_Scheme_CD8) {
                        df[k][j][i] = c1 * (f[k+1][j][i] - f[k-1][j][i]) 
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

    // 3. 获取局部数组指针（包含 ghost，可用全局索引访问邻居点）
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

    // 6. 分配临时导数数组
    // 要点：数组每一维的大小必须能容纳使用全局索引的直接访问
    // 即 xi[k][j][i], k∈[zs, zs+zm-1], j∈[ys, ys+ym-1], i∈[xs, xs+xm-1]
    PetscReal ***xi, ***xj, ***xk;
    PetscReal ***yi, ***yj, ***yk;
    PetscReal ***zi, ***zj, ***zk;

    PetscInt total_z = zs + zm;   // 第一维大小
    PetscInt total_y = ys + ym;   // 第二维大小
    PetscInt total_x = xs + xm;   // 第三维大小

    // 分配第一维（k 方向）
    ierr = PetscMalloc(total_z * sizeof(PetscReal**), &xi); CHKERRQ(ierr);
    ierr = PetscMalloc(total_z * sizeof(PetscReal**), &xj); CHKERRQ(ierr);
    ierr = PetscMalloc(total_z * sizeof(PetscReal**), &xk); CHKERRQ(ierr);
    ierr = PetscMalloc(total_z * sizeof(PetscReal**), &yi); CHKERRQ(ierr);
    ierr = PetscMalloc(total_z * sizeof(PetscReal**), &yj); CHKERRQ(ierr);
    ierr = PetscMalloc(total_z * sizeof(PetscReal**), &yk); CHKERRQ(ierr);
    ierr = PetscMalloc(total_z * sizeof(PetscReal**), &zi); CHKERRQ(ierr);
    ierr = PetscMalloc(total_z * sizeof(PetscReal**), &zj); CHKERRQ(ierr);
    ierr = PetscMalloc(total_z * sizeof(PetscReal**), &zk); CHKERRQ(ierr);

    // 前 zs 层不会被访问，设为 NULL 防止误用
    for (PetscInt k = 0; k < zs; k++) {
        xi[k] = NULL; xj[k] = NULL; xk[k] = NULL;
        yi[k] = NULL; yj[k] = NULL; yk[k] = NULL;
        zi[k] = NULL; zj[k] = NULL; zk[k] = NULL;
    }

    // 为真正使用的 z 层分配 j 和 i 方向
    for (PetscInt k = zs; k < zs + zm; k++) {
        // 分配第二维（j 方向），大小为 total_y
        PetscMalloc(total_y * sizeof(PetscReal*), &xi[k]);
        PetscMalloc(total_y * sizeof(PetscReal*), &xj[k]);
        PetscMalloc(total_y * sizeof(PetscReal*), &xk[k]);
        PetscMalloc(total_y * sizeof(PetscReal*), &yi[k]);
        PetscMalloc(total_y * sizeof(PetscReal*), &yj[k]);
        PetscMalloc(total_y * sizeof(PetscReal*), &yk[k]);
        PetscMalloc(total_y * sizeof(PetscReal*), &zi[k]);
        PetscMalloc(total_y * sizeof(PetscReal*), &zj[k]);
        PetscMalloc(total_y * sizeof(PetscReal*), &zk[k]);

        // 前 ys 行未使用，设为 NULL
        for (PetscInt j = 0; j < ys; j++) {
            xi[k][j] = NULL; xj[k][j] = NULL; xk[k][j] = NULL;
            yi[k][j] = NULL; yj[k][j] = NULL; yk[k][j] = NULL;
            zi[k][j] = NULL; zj[k][j] = NULL; zk[k][j] = NULL;
        }

        // 为实际使用的 y 行分配第三维（i 方向）
        for (PetscInt j = ys; j < ys + ym; j++) {
            PetscMalloc(total_x * sizeof(PetscReal), &xi[k][j]);
            PetscMalloc(total_x * sizeof(PetscReal), &xj[k][j]);
            PetscMalloc(total_x * sizeof(PetscReal), &xk[k][j]);
            PetscMalloc(total_x * sizeof(PetscReal), &yi[k][j]);
            PetscMalloc(total_x * sizeof(PetscReal), &yj[k][j]);
            PetscMalloc(total_x * sizeof(PetscReal), &yk[k][j]);
            PetscMalloc(total_x * sizeof(PetscReal), &zi[k][j]);
            PetscMalloc(total_x * sizeof(PetscReal), &zj[k][j]);
            PetscMalloc(total_x * sizeof(PetscReal), &zk[k][j]);
        }
    }

    // 7. 计算导数（传入含 ghost 的局部数组，写入到我们刚分配的大数组）
    compute_derivative_x(axx, xi);
    compute_derivative_x(ayy, yi);
    compute_derivative_x(azz, zi);

    compute_derivative_y(axx, xj);
    compute_derivative_y(ayy, yj);
    compute_derivative_y(azz, zj);

    compute_derivative_z(axx, xk);
    compute_derivative_z(ayy, yk);
    compute_derivative_z(azz, zk);

    // 8. 计算雅可比系数（只遍历本进程的物理点）
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                PetscReal xi1 = xi[k][j][i];
                PetscReal xj1 = xj[k][j][i];
                PetscReal xk1 = xk[k][j][i];
                PetscReal yi1 = yi[k][j][i];
                PetscReal yj1 = yj[k][j][i];
                PetscReal yk1 = yk[k][j][i];
                PetscReal zi1 = zi[k][j][i];
                PetscReal zj1 = zj[k][j][i];
                PetscReal zk1 = zk[k][j][i];

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
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            PetscFree(xi[k][j]);
            PetscFree(xj[k][j]);
            PetscFree(xk[k][j]);
            PetscFree(yi[k][j]);
            PetscFree(yj[k][j]);
            PetscFree(yk[k][j]);
            PetscFree(zi[k][j]);
            PetscFree(zj[k][j]);
            PetscFree(zk[k][j]);
        }
        PetscFree(xi[k]);
        PetscFree(xj[k]);
        PetscFree(xk[k]);
        PetscFree(yi[k]);
        PetscFree(yj[k]);
        PetscFree(yk[k]);
        PetscFree(zi[k]);
        PetscFree(zj[k]);
        PetscFree(zk[k]);
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