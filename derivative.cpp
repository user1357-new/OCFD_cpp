#include "mesh.h"
#include <petscdmda.h>
#include <cmath>

// x方向导数计算（对应Fortran的OCFD_dx0）
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

// y方向导数计算（对应Fortran的OCFD_dy0）
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

// z方向导数计算（对应Fortran的OCFD_dz0）
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