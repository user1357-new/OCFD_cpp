#include "flux_derivative.h"
#include <cmath>

// ====================================================================
// CD 系数（与 mesh_jacobi.cpp 公式一致，独立存放）
// coeff[m] 已含 1/(2h)：  df/dx = Σ coeff[m] * (f[i+m+1] - f[i-m-1])
// ====================================================================
static inline void getCD2_coeff(PetscReal h, PetscReal *c) {
    c[0] = 1.0 / (2.0 * h);
}
static inline void getCD4_coeff(PetscReal h, PetscReal *c) {
    c[0] =  8.0 / (12.0 * h);
    c[1] = -1.0 / (12.0 * h);
}
static inline void getCD6_coeff(PetscReal h, PetscReal *c) {
    c[0] =  45.0 / (60.0 * h);
    c[1] =  -9.0 / (60.0 * h);
    c[2] =   1.0 / (60.0 * h);
}
static inline void getCD8_coeff(PetscReal h, PetscReal *c) {
    c[0] =  0.8 / h;
    c[1] = -0.2 / h;
    c[2] =  3.80952380952380952e-2 / h;
    c[3] = -3.571428571428571428e-3 / h;
}

// ====================================================================
// 辅助函数：从 DM 获取全局维度
// ====================================================================
static void getGlobalDims(DM da, PetscInt &nx, PetscInt &ny, PetscInt &nz)
{
    PetscInt dim;
    DMDAGetInfo(da, &dim, &nx, &ny, &nz,
                NULL,NULL,NULL, NULL,NULL, NULL,NULL,NULL, NULL);
}

// ====================================================================
// CD6 导数实现
//
// f  : DMDA local Vec 数组，global 索引 [k][j][i]（含 ghost）
// df : DMDA local Vec 数组，global 索引 [k][j][i]
//      仅填充内部格心 (xs..xs+xm-1 等)，ghost 区域不写
// ====================================================================
void CD6FluxDerivative::deriv_xi(DM da, PetscReal ***f, PetscReal ***df)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    PetscInt nx_g, ny_g, nz_g;
    getGlobalDims(da, nx_g, ny_g, nz_g);
    PetscReal hx = 1.0 / (PetscReal)nx_g;

    PetscReal c[3];
    getCD6_coeff(hx, c);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                df[k][j][i] = c[0] * (f[k][j][i+1] - f[k][j][i-1])
                            + c[1] * (f[k][j][i+2] - f[k][j][i-2])
                            + c[2] * (f[k][j][i+3] - f[k][j][i-3]);
            }
        }
    }
}

void CD6FluxDerivative::deriv_eta(DM da, PetscReal ***f, PetscReal ***df)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    PetscInt nx_g, ny_g, nz_g;
    getGlobalDims(da, nx_g, ny_g, nz_g);
    PetscReal hy = 1.0 / (PetscReal)ny_g;

    PetscReal c[3];
    getCD6_coeff(hy, c);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                df[k][j][i] = c[0] * (f[k][j+1][i] - f[k][j-1][i])
                            + c[1] * (f[k][j+2][i] - f[k][j-2][i])
                            + c[2] * (f[k][j+3][i] - f[k][j-3][i]);
            }
        }
    }
}

void CD6FluxDerivative::deriv_zeta(DM da, PetscReal ***f, PetscReal ***df)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    PetscInt nx_g, ny_g, nz_g;
    getGlobalDims(da, nx_g, ny_g, nz_g);
    PetscReal hz = 1.0 / (PetscReal)nz_g;

    PetscReal c[3];
    getCD6_coeff(hz, c);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                df[k][j][i] = c[0] * (f[k+1][j][i] - f[k-1][j][i])
                            + c[1] * (f[k+2][j][i] - f[k-2][j][i])
                            + c[2] * (f[k+3][j][i] - f[k-3][j][i]);
            }
        }
    }
}

// ====================================================================
// CD8 导数实现
// ====================================================================
void CD8FluxDerivative::deriv_xi(DM da, PetscReal ***f, PetscReal ***df)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    PetscInt nx_g, ny_g, nz_g;
    getGlobalDims(da, nx_g, ny_g, nz_g);
    PetscReal hx = 1.0 / (PetscReal)nx_g;

    PetscReal c[4];
    getCD8_coeff(hx, c);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                df[k][j][i] = c[0] * (f[k][j][i+1] - f[k][j][i-1])
                            + c[1] * (f[k][j][i+2] - f[k][j][i-2])
                            + c[2] * (f[k][j][i+3] - f[k][j][i-3])
                            + c[3] * (f[k][j][i+4] - f[k][j][i-4]);
            }
        }
    }
}

void CD8FluxDerivative::deriv_eta(DM da, PetscReal ***f, PetscReal ***df)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    PetscInt nx_g, ny_g, nz_g;
    getGlobalDims(da, nx_g, ny_g, nz_g);
    PetscReal hy = 1.0 / (PetscReal)ny_g;

    PetscReal c[4];
    getCD8_coeff(hy, c);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                df[k][j][i] = c[0] * (f[k][j+1][i] - f[k][j-1][i])
                            + c[1] * (f[k][j+2][i] - f[k][j-2][i])
                            + c[2] * (f[k][j+3][i] - f[k][j-3][i])
                            + c[3] * (f[k][j+4][i] - f[k][j-4][i]);
            }
        }
    }
}

void CD8FluxDerivative::deriv_zeta(DM da, PetscReal ***f, PetscReal ***df)
{
    DMDALocalInfo info;
    DMDAGetLocalInfo(da, &info);

    PetscInt nx_g, ny_g, nz_g;
    getGlobalDims(da, nx_g, ny_g, nz_g);
    PetscReal hz = 1.0 / (PetscReal)nz_g;

    PetscReal c[4];
    getCD8_coeff(hz, c);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                df[k][j][i] = c[0] * (f[k+1][j][i] - f[k-1][j][i])
                            + c[1] * (f[k+2][j][i] - f[k-2][j][i])
                            + c[2] * (f[k+3][j][i] - f[k-3][j][i])
                            + c[3] * (f[k+4][j][i] - f[k-4][j][i]);
            }
        }
    }
}

// ====================================================================
// DerivativeRegistry
// ====================================================================
DerivativeRegistry& DerivativeRegistry::instance()
{
    static DerivativeRegistry reg;
    return reg;
}

void DerivativeRegistry::add(const std::string& type, std::unique_ptr<FluxDerivative> d)
{
    registry_[type] = std::move(d);
}

FluxDerivative* DerivativeRegistry::get(const std::string& type)
{
    auto it = registry_.find(type);
    if (it != registry_.end())
        return it->second.get();
    return nullptr;
}

bool DerivativeRegistry::has(const std::string& type) const
{
    return registry_.find(type) != registry_.end();
}

// ====================================================================
// createDerivative — 简单工厂
// ====================================================================
FluxDerivative* createDerivative(const std::string& name)
{
    if      (name == "CD6") return new CD6FluxDerivative();
    else if (name == "CD8") return new CD8FluxDerivative();
    return nullptr;
}
