#include "BC_filler.h"
#include "mesh.h"
#include <cmath>
#include <algorithm>

// ====================================================================
// 工具函数
// ====================================================================
void BCFaceFiller::faceNormal(int face, PetscReal n[3])
{
    n[0] = n[1] = n[2] = 0.0;
    switch (face) {
    case 0: n[0] = -1.0; break;  // LEFT
    case 1: n[0] = +1.0; break;  // RIGHT
    case 2: n[1] = -1.0; break;  // BOTTOM
    case 3: n[1] = +1.0; break;  // TOP
    case 4: n[2] = -1.0; break;  // BACK
    case 5: n[2] = +1.0; break;  // FRONT
    }
}

void BCFaceFiller::ghostRange(int face, PetscInt LAP,
                               PetscInt NX, PetscInt NY, PetscInt NZ,
                               const DMDALocalInfo& info,
                               int& i0, int& i1, int& j0, int& j1, int& k0, int& k1)
{
    i0 = -LAP;   i1 = NX + LAP - 1;
    j0 = -LAP;   j1 = NY + LAP - 1;
    k0 = -LAP;   k1 = NZ + LAP - 1;

    switch (face) {
    case 0: i0 = -LAP;   i1 = -1;        j0 = 0;  j1 = NY - 1;    k0 = 0;  k1 = NZ - 1; break;
    case 1: i0 = NX;     i1 = NX+LAP-1;  j0 = 0;  j1 = NY - 1;    k0 = 0;  k1 = NZ - 1; break;
    case 2: i0 = 0;      i1 = NX - 1;    j0 = -LAP; j1 = -1;      k0 = 0;  k1 = NZ - 1; break;
    case 3: i0 = 0;      i1 = NX - 1;    j0 = NY;   j1 = NY+LAP-1; k0 = 0;  k1 = NZ - 1; break;
    case 4: i0 = 0;      i1 = NX - 1;    j0 = 0;    j1 = NY - 1;   k0 = -LAP; k1 = -1; break;
    case 5: i0 = 0;      i1 = NX - 1;    j0 = 0;    j1 = NY - 1;   k0 = NZ;   k1 = NZ+LAP-1; break;
    }

    // clamp to local range
    int i_lo = info.gxs,        i_hi = info.gxs + info.gxm - 1;
    int j_lo = info.gys,        j_hi = info.gys + info.gym - 1;
    int k_lo = info.gzs,        k_hi = info.gzs + info.gzm - 1;

    if (i0 < i_lo) i0 = i_lo;  if (i1 > i_hi) i1 = i_hi;
    if (j0 < j_lo) j0 = j_lo;  if (j1 > j_hi) j1 = j_hi;
    if (k0 < k_lo) k0 = k_lo;  if (k1 > k_hi) k1 = k_hi;
}

void BCFaceFiller::consToPrim(PetscReal U[5], PetscReal gamma,
                               PetscReal& rho, PetscReal& u, PetscReal& v,
                               PetscReal& w, PetscReal& p)
{
    rho = U[0];
    u   = U[1] / rho;
    v   = U[2] / rho;
    w   = U[3] / rho;
    PetscReal v2 = u*u + v*v + w*w;
    p   = (gamma - 1.0) * (U[4] - 0.5 * rho * v2);
}

void BCFaceFiller::primToCons(PetscReal rho, PetscReal u, PetscReal v,
                               PetscReal w, PetscReal p, PetscReal gamma,
                               PetscReal U[5])
{
    U[0] = rho;
    U[1] = rho * u;
    U[2] = rho * v;
    U[3] = rho * w;
    U[4] = p / (gamma - 1.0) + 0.5 * rho * (u*u + v*v + w*w);
}

// ====================================================================
// BCFillerRegistry
// ====================================================================
BCFillerRegistry& BCFillerRegistry::instance()
{
    static BCFillerRegistry reg;
    return reg;
}

void BCFillerRegistry::add(const std::string& type, std::unique_ptr<BCFaceFiller> filler)
{
    registry_[type] = std::move(filler);
}

BCFaceFiller* BCFillerRegistry::get(const std::string& type)
{
    auto it = registry_.find(type);
    return (it != registry_.end()) ? it->second.get() : nullptr;
}

bool BCFillerRegistry::has(const std::string& type) const
{
    return registry_.find(type) != registry_.end();
}

// ====================================================================
// BCWallFiller — 等温无滑移壁面
//
// 速度全部反向 (无滑移: (v_g + v_m)/2 = 0 at wall)
// 压力偶对称 (dp/dn = 0)
// 密度由等温条件确定: T_g = 2*Tw - T_m (线性插值到壁温)
//   若 Tw 未指定 (==0), 回退到绝热壁 ρ_g = ρ_m
// ghost_cell_value = f(mirror_interior_cell_value)
// ====================================================================
void BCWallFiller::fillGhostFace(Mesh* mesh, int face,
                                  PetscReal*** u[5],
                                  const DMDALocalInfo& info)
{
    PetscInt NX = mesh->getNxGlobal();
    PetscInt NY = mesh->getNyGlobal();
    PetscInt NZ = mesh->getNzGlobal();
    PetscInt LAP = mesh->getLAP();

    int i0, i1, j0, j1, k0, k1;
    ghostRange(face, LAP, NX, NY, NZ, info, i0, i1, j0, j1, k0, k1);

    for (int k = k0; k <= k1; ++k) {
        for (int j = j0; j <= j1; ++j) {
            for (int i = i0; i <= i1; ++i) {
                if (i >= 0 && i < NX && j >= 0 && j < NY && k >= 0 && k < NZ)
                    continue;

                // 镜像 interior 索引 (二阶镜像)
                int im = i, jm = j, km = k;
                if      (face == 0) im = -1 - i;
                else if (face == 1) im = 2*(NX-1) - i;
                else if (face == 2) jm = -1 - j;
                else if (face == 3) jm = 2*(NY-1) - j;
                else if (face == 4) km = -1 - k;
                else if (face == 5) km = 2*(NZ-1) - k;

                // 读取镜像点守恒量
                PetscReal U_m[5];
                for (int c = 0; c < 5; ++c)
                    U_m[c] = u[c][km][jm][im];

                // 守恒量 → 原始变量
                PetscReal rho_m, u_m, v_m, w_m, p_m;
                consToPrim(U_m, gamma_, rho_m, u_m, v_m, w_m, p_m);

                // 无滑移: 所有速度分量反向
                // (v_g + v_m)/2 = 0 → v_g = -v_m
                PetscReal u_g = -u_m;
                PetscReal v_g = -v_m;
                PetscReal w_g = -w_m;

                // 压力偶对称: dp/dn = 0
                PetscReal p_g = p_m;

                // 密度: 等温壁面条件
                PetscReal rho_g;
                if (Tw_ > 0.0) {
                    // T = p/ρ (无量纲, R=1)
                    PetscReal T_m = p_m / rho_m;
                    // 线性插值: (T_g + T_m)/2 = Tw
                    PetscReal T_g = 2.0 * Tw_ - T_m;
                    rho_g = p_g / T_g;
                    // 安全保护: 防止负密度
                    if (rho_g <= 0.0) rho_g = rho_m;
                } else {
                    // 未指定壁温 → 绝热壁 fallback
                    rho_g = rho_m;
                }

                // 密度限制器 [0.5, 1.5]×ρ_m
                PetscReal rho_lo = 0.5 * rho_m;
                PetscReal rho_hi = 1.5 * rho_m;
                if (rho_g < rho_lo) rho_g = rho_lo;
                if (rho_g > rho_hi) rho_g = rho_hi;

                // 写回守恒量
                PetscReal U_g[5];
                primToCons(rho_g, u_g, v_g, w_g, p_g, gamma_, U_g);
                for (int c = 0; c < 5; ++c)
                    u[c][k][j][i] = U_g[c];
            }
        }
    }
}

// ====================================================================
// BCInletFiller — 超音速入口 (5 守恒量 Dirichlet)
//
// U_face = U_inlet → (U_ghost + U_interior)/2 = U_inlet
// → U_ghost = 2 * U_inlet - U_interior
// ====================================================================
BCInletFiller::BCInletFiller(PetscReal gamma,
                             PetscReal rho_in, PetscReal u_in,
                             PetscReal v_in, PetscReal w_in, PetscReal p_in)
    : gamma_(gamma)
{
    primToCons(rho_in, u_in, v_in, w_in, p_in, gamma_, U_in_);
}

void BCInletFiller::fillGhostFace(Mesh* mesh, int face,
                                  PetscReal*** u[5],
                                  const DMDALocalInfo& info)
{
    PetscInt NX = mesh->getNxGlobal();
    PetscInt NY = mesh->getNyGlobal();
    PetscInt NZ = mesh->getNzGlobal();
    PetscInt LAP = mesh->getLAP();

    int i0, i1, j0, j1, k0, k1;
    ghostRange(face, LAP, NX, NY, NZ, info, i0, i1, j0, j1, k0, k1);

    for (int k = k0; k <= k1; ++k) {
        for (int j = j0; j <= j1; ++j) {
            for (int i = i0; i <= i1; ++i) {
                if (i >= 0 && i < NX && j >= 0 && j < NY && k >= 0 && k < NZ)
                    continue;

                // 最近内点索引
                int ie = i, je = j, ke = k;
                if      (i < 0)   ie = 0;
                else if (i >= NX) ie = NX - 1;
                if      (j < 0)   je = 0;
                else if (j >= NY) je = NY - 1;
                if      (k < 0)   ke = 0;
                else if (k >= NZ) ke = NZ - 1;

                // Dirichlet 镜像: U_ghost = 2*U_inlet - U_interior
                for (int c = 0; c < 5; ++c) {
                    PetscReal U_interior = u[c][ke][je][ie];
                    u[c][k][j][i] = 2.0 * U_in_[c] - U_interior;
                }
            }
        }
    }
}

// ====================================================================
// BCFarfieldFiller — 基于 Riemann 不变量的无反射远场
//
// Step 1: 计算法向马赫数 Mn_∞ = Vn_∞ / c_∞
// Step 2: 根据 Mn_∞ 分四种流态
//   Mn_∞ ≤ -1:  超音速流入 → 全取来流
//   -1 < Mn_∞ < 0: 亚音速流入 → J⁺来流, J⁻内点, s来流, Vt来流
//    0 < Mn_∞ < 1: 亚音速流出 → J⁺内点, J⁻来流, s内点, Vt内点
//    Mn_∞ ≥ 1:  超音速流出 → 全取内点
// Step 3: 亚音速时 Riemann 不变量求解边界面原始变量
// Step 4: Ghost 零阶外推 = 边界面值
// ====================================================================
BCFarfieldFiller::BCFarfieldFiller(PetscReal gamma,
                                   PetscReal rho_inf, PetscReal u_inf,
                                   PetscReal v_inf, PetscReal w_inf, PetscReal p_inf)
    : gamma_(gamma),
      rho_inf_(rho_inf), u_inf_(u_inf), v_inf_(v_inf), w_inf_(w_inf), p_inf_(p_inf)
{
    c_inf_ = std::sqrt(gamma_ * p_inf_ / rho_inf_);
}

void BCFarfieldFiller::fillGhostFace(Mesh* mesh, int face,
                                     PetscReal*** u[5],
                                     const DMDALocalInfo& info)
{
    PetscInt NX = mesh->getNxGlobal();
    PetscInt NY = mesh->getNyGlobal();
    PetscInt NZ = mesh->getNzGlobal();
    PetscInt LAP = mesh->getLAP();
    PetscReal gm1 = gamma_ - 1.0;

    PetscReal n[3];
    faceNormal(face, n);

    int i0, i1, j0, j1, k0, k1;
    ghostRange(face, LAP, NX, NY, NZ, info, i0, i1, j0, j1, k0, k1);

    for (int k = k0; k <= k1; ++k) {
        for (int j = j0; j <= j1; ++j) {
            for (int i = i0; i <= i1; ++i) {
                if (i >= 0 && i < NX && j >= 0 && j < NY && k >= 0 && k < NZ)
                    continue;

                // ---- 最近内点 ----
                int ie = i, je = j, ke = k;
                if      (i < 0)   ie = 0;
                else if (i >= NX) ie = NX - 1;
                if      (j < 0)   je = 0;
                else if (j >= NY) je = NY - 1;
                if      (k < 0)   ke = 0;
                else if (k >= NZ) ke = NZ - 1;

                // 读取内点守恒量 → 原始变量
                PetscReal U_int[5];
                for (int c = 0; c < 5; ++c)
                    U_int[c] = u[c][ke][je][ie];
                PetscReal rho_int, u_int, v_int, w_int, p_int;
                consToPrim(U_int, gamma_, rho_int, u_int, v_int, w_int, p_int);
                PetscReal c_int = std::sqrt(gamma_ * p_int / rho_int);

                // ---- 法向马赫数 ----
                PetscReal Vn_inf = u_inf_ * n[0] + v_inf_ * n[1] + w_inf_ * n[2];
                PetscReal Vn_int = u_int * n[0] + v_int * n[1] + w_int * n[2];
                PetscReal Mn_inf = Vn_inf / c_inf_;
                PetscReal Mn_int = Vn_int / c_int;

                PetscReal rho_b, u_b, v_b, w_b, p_b;

                if (Vn_int <= 0.0) {
                    // ======== 流入 (内点法向速度指向域内) ========
                    if (Mn_inf <= -1.0) {
                        // 超音速流入 → 全取来流 (所有特征线指向域内)
                        rho_b = rho_inf_;
                        u_b   = u_inf_;
                        v_b   = v_inf_;
                        w_b   = w_inf_;
                        p_b   = p_inf_;
                    } else {
                        // 亚音速流入 → J⁺ 从来流, J⁻ 从内点
                        PetscReal Jp = Vn_inf + 2.0 * c_inf_ / gm1;
                        PetscReal Jm = Vn_int - 2.0 * c_int  / gm1;

                        PetscReal Vn_b = 0.5 * (Jp + Jm);
                        PetscReal c_b  = 0.25 * gm1 * (Jp - Jm);

                        // 熵和切向速度从来流
                        PetscReal s_b = p_inf_ / std::pow(rho_inf_, gamma_);
                        PetscReal Vt_x = u_inf_ - Vn_inf * n[0];
                        PetscReal Vt_y = v_inf_ - Vn_inf * n[1];
                        PetscReal Vt_z = w_inf_ - Vn_inf * n[2];

                        rho_b = std::pow(c_b * c_b / (gamma_ * s_b), 1.0 / gm1);
                        p_b   = rho_b * c_b * c_b / gamma_;
                        u_b   = Vt_x + Vn_b * n[0];
                        v_b   = Vt_y + Vn_b * n[1];
                        w_b   = Vt_z + Vn_b * n[2];
                    }
                } else {
                    // ======== 流出 (内点法向速度指向域外) ========
                    if (Mn_int >= 1.0) {
                        // 超音速流出 → 全取内点 (所有特征线指向域外)
                        rho_b = rho_int;
                        u_b   = u_int;
                        v_b   = v_int;
                        w_b   = w_int;
                        p_b   = p_int;
                    } else {
                        // 亚音速流出 → J⁺ 从内点, J⁻ 从来流
                        PetscReal Jp = Vn_int + 2.0 * c_int  / gm1;
                        PetscReal Jm = Vn_inf - 2.0 * c_inf_ / gm1;

                        PetscReal Vn_b = 0.5 * (Jp + Jm);
                        PetscReal c_b  = 0.25 * gm1 * (Jp - Jm);

                        // 熵和切向速度从内点
                        PetscReal s_b = p_int / std::pow(rho_int, gamma_);
                        PetscReal Vt_x = u_int - Vn_int * n[0];
                        PetscReal Vt_y = v_int - Vn_int * n[1];
                        PetscReal Vt_z = w_int - Vn_int * n[2];

                        rho_b = std::pow(c_b * c_b / (gamma_ * s_b), 1.0 / gm1);
                        p_b   = rho_b * c_b * c_b / gamma_;
                        u_b   = Vt_x + Vn_b * n[0];
                        v_b   = Vt_y + Vn_b * n[1];
                        w_b   = Vt_z + Vn_b * n[2];
                    }
                }

                // 镜像: U_ghost = 2*U_b - U_int, 使面平均 (U_ghost+U_int)/2 = U_b
                PetscReal U_b[5];
                primToCons(rho_b, u_b, v_b, w_b, p_b, gamma_, U_b);
                for (int c = 0; c < 5; ++c)
                    u[c][k][j][i] = 2.0 * U_b[c] - U_int[c];
            }
        }
    }
}

// ====================================================================
// BCExtrapolateFiller — 零阶外推
// ====================================================================
void BCExtrapolateFiller::fillGhostFace(Mesh* mesh, int face,
                                         PetscReal*** u[5],
                                         const DMDALocalInfo& info)
{
    PetscInt NX = mesh->getNxGlobal();
    PetscInt NY = mesh->getNyGlobal();
    PetscInt NZ = mesh->getNzGlobal();
    PetscInt LAP = mesh->getLAP();

    int i0, i1, j0, j1, k0, k1;
    ghostRange(face, LAP, NX, NY, NZ, info, i0, i1, j0, j1, k0, k1);

    for (int k = k0; k <= k1; ++k) {
        for (int j = j0; j <= j1; ++j) {
            for (int i = i0; i <= i1; ++i) {
                if (i >= 0 && i < NX && j >= 0 && j < NY && k >= 0 && k < NZ)
                    continue;

                // 最近边界内点
                int ie = i, je = j, ke = k;
                if      (i < 0)   ie = 0;
                else if (i >= NX) ie = NX - 1;
                if      (j < 0)   je = 0;
                else if (j >= NY) je = NY - 1;
                if      (k < 0)   ke = 0;
                else if (k >= NZ) ke = NZ - 1;

                for (int c = 0; c < 5; ++c)
                    u[c][k][j][i] = u[c][ke][je][ie];
            }
        }
    }
}
