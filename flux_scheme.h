#ifndef FLUX_SCHEME_H
#define FLUX_SCHEME_H

#include <petscsys.h>
#include <cmath>

// ====================================================================
// LinearEulerFlux — 三维线性化 Euler 方程无粘通量（静止基底）
//
// 守恒变量 U' = [ρ',  ρ₀u',  ρ₀v',  ρ₀w',  p']
//   ρ'  = 密度扰动
//   u',v',w' = 速度扰动
//   p'  = 压力扰动
//   ρ₀, p₀ = 静止基态密度和压力
//   c₀² = γ p₀ / ρ₀
//
// 线性化通量（u₀=v₀=w₀=0 处一阶 Taylor 展开）：
//   F' = [ρ₀u',  p',  0,   0,  γp₀u']   = [U[1], U[4], 0, 0, c0sq * U[1]]
//   G' = [ρ₀v',  0,   p',  0,  γp₀v']   = [U[2], 0, U[4], 0, c0sq * U[2]]
//   H' = [ρ₀w',  0,   0,   p', γp₀w']   = [U[3], 0, 0, U[4], c0sq * U[3]]
//
// 通量是 U' 的线性函数，不需要 consToPrim / primToCons 转换。
// ====================================================================
class LinearEulerFlux {
public:
    /// @param c0sq  基态声速平方 c₀² = γ p₀ / ρ₀
    LinearEulerFlux(PetscReal c0sq) : c0sq_(c0sq) {}

    /// @brief 计算单点线性化 Euler 物理通量（静止基底）
    /// @param U[5]  扰动守恒变量
    /// @param F[5]  输出: x 方向通量
    /// @param G[5]  输出: y 方向通量
    /// @param H[5]  输出: z 方向通量
    inline void computeFlux(const PetscReal U[5],
                            PetscReal F[5], PetscReal G[5], PetscReal H[5]) const
    {
        // F = [ρ₀u', p', 0, 0, c₀² ρ₀u']
        F[0] = U[1];                // ρ₀u'
        F[1] = U[4];                // p'
        F[2] = 0.0;
        F[3] = 0.0;
        F[4] = c0sq_ * U[1];        // γp₀ u' = c₀² ρ₀u'

        // G = [ρ₀v', 0, p', 0, c₀² ρ₀v']
        G[0] = U[2];                // ρ₀v'
        G[1] = 0.0;
        G[2] = U[4];                // p'
        G[3] = 0.0;
        G[4] = c0sq_ * U[2];        // c₀² ρ₀v'

        // H = [ρ₀w', 0, 0, p', c₀² ρ₀w']
        H[0] = U[3];                // ρ₀w'
        H[1] = 0.0;
        H[2] = 0.0;
        H[3] = U[4];                // p'
        H[4] = c0sq_ * U[3];        // c₀² ρ₀w'
    }

    /// @brief 守恒量 → 原始扰动变量
    /// @param U[5]   扰动守恒量 [ρ', ρ₀u', ρ₀v', ρ₀w', p']
    /// @param base_rho  基态密度 ρ₀
    /// @param gamma     比热比
    /// @param rho   输出: ρ'
    /// @param u     输出: u'
    /// @param v     输出: v'
    /// @param w     输出: w'
    /// @param p     输出: p'
    static inline void consToPrim(const PetscReal U[5], PetscReal base_rho, PetscReal gamma,
                                   PetscReal &rho, PetscReal &u, PetscReal &v,
                                   PetscReal &w, PetscReal &p)
    {
        rho = U[0];
        PetscReal inv_base_rho = 1.0 / base_rho;
        u = U[1] * inv_base_rho;
        v = U[2] * inv_base_rho;
        w = U[3] * inv_base_rho;
        p = U[4];
    }

    /// @brief 原始扰动变量 → 守恒量
    static inline void primToCons(PetscReal rho, PetscReal u, PetscReal v,
                                   PetscReal w, PetscReal p, PetscReal base_rho,
                                   PetscReal /*gamma*/,
                                   PetscReal U[5])
    {
        U[0] = rho;              // ρ'
        U[1] = base_rho * u;     // ρ₀u'
        U[2] = base_rho * v;     // ρ₀v'
        U[3] = base_rho * w;     // ρ₀w'
        U[4] = p;                // p'
    }

    /// @brief 从原始变量计算基态声速
    static PetscReal soundSpeed(PetscReal rho, PetscReal p, PetscReal gamma)
    {
        return std::sqrt(gamma * p / rho);
    }

    PetscReal c0sq() const { return c0sq_; }

private:
    PetscReal c0sq_;   // c₀² = γ p₀ / ρ₀
};

#endif
