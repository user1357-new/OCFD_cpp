#ifndef BC_FILLER_H
#define BC_FILLER_H

#include <petscdmda.h>
#include <string>
#include <map>
#include <memory>

class Mesh;

// ====================================================================
// BCFaceFiller — 物理边界 ghost 填充基类
// ====================================================================
class BCFaceFiller {
public:
    virtual ~BCFaceFiller() = default;

    /// @brief 填一个面上所有 ghost 层的守恒量 U
    /// @param mesh  网格块
    /// @param face  面编号 (0..5)
    /// @param u     5 分量守恒量 3D 数组 (来自 local DMDA vector)
    /// @param info  DMDA local info (含 ghost 范围)
    virtual void fillGhostFace(Mesh* mesh, int face,
                               PetscReal*** u[5],
                               const DMDALocalInfo& info) = 0;

    virtual const char* name() const = 0;

protected:
    /// @brief 面外法向量 (向外)
    static void faceNormal(int face, PetscReal n[3]);

    /// @brief 该面 ghost 层索引范围
    static void ghostRange(int face, PetscInt LAP,
                           PetscInt NX, PetscInt NY, PetscInt NZ,
                           const DMDALocalInfo& info,
                           int& i0, int& i1, int& j0, int& j1, int& k0, int& k1);

    /// @brief 从守恒量计算原始变量 (rho, u, v, w, p)
    static void consToPrim(PetscReal U[5], PetscReal gamma,
                           PetscReal& rho, PetscReal& u, PetscReal& v,
                           PetscReal& w, PetscReal& p);

    /// @brief 从原始变量计算守恒量
    static void primToCons(PetscReal rho, PetscReal u, PetscReal v,
                           PetscReal w, PetscReal p, PetscReal gamma,
                           PetscReal U[5]);
};

// ====================================================================
// BCFillerRegistry — 工厂/注册表单例
// ====================================================================
class BCFillerRegistry {
public:
    static BCFillerRegistry& instance();

    void add(const std::string& type, std::unique_ptr<BCFaceFiller> filler);
    BCFaceFiller* get(const std::string& type);
    bool has(const std::string& type) const;

private:
    BCFillerRegistry() = default;
    std::map<std::string, std::unique_ptr<BCFaceFiller>> registry_;
};

// ====================================================================
// BCWallFiller — 等温无滑移壁面
// ====================================================================
class BCWallFiller : public BCFaceFiller {
public:
    BCWallFiller(PetscReal gamma, PetscReal Tw = 0.0)
        : gamma_(gamma), Tw_(Tw) {}
    const char* name() const override { return "BCWall"; }
    void fillGhostFace(Mesh* mesh, int face,
                       PetscReal*** u[5],
                       const DMDALocalInfo& info) override;
private:
    PetscReal gamma_;
    PetscReal Tw_;     // 壁温 (暂未用，保留)
};

// ====================================================================
// BCInletFiller — 超音速入口 (5 守恒量 Dirichlet)
//
// U_ghost = 2 * U_inlet - U_interior  (镜像绕入口值)
// ====================================================================
class BCInletFiller : public BCFaceFiller {
public:
    BCInletFiller(PetscReal gamma,
                  PetscReal rho_in, PetscReal u_in,
                  PetscReal v_in, PetscReal w_in, PetscReal p_in);
    const char* name() const override { return "BCInlet"; }
    void fillGhostFace(Mesh* mesh, int face,
                       PetscReal*** u[5],
                       const DMDALocalInfo& info) override;
private:
    PetscReal gamma_;
    PetscReal U_in_[5];  // 预存的入口守恒量
};

// ====================================================================
// BCFarfieldFiller — 基于 Riemann 不变量的无反射远场
//
// 对每个 ghost 格点:
//   1. 一维特征线理论判断流态 (Mn_∞ = Vn_∞ / c_∞)
//   2. Riemann 不变量 J⁺/J⁻ 求解边界面原始变量
//   3. Ghost 零阶外推 = 边界面值
// ====================================================================
class BCFarfieldFiller : public BCFaceFiller {
public:
    BCFarfieldFiller(PetscReal gamma,
                     PetscReal rho_inf, PetscReal u_inf,
                     PetscReal v_inf, PetscReal w_inf, PetscReal p_inf);
    const char* name() const override { return "BCFarfield"; }
    void fillGhostFace(Mesh* mesh, int face,
                       PetscReal*** u[5],
                       const DMDALocalInfo& info) override;
private:
    PetscReal gamma_;
    PetscReal rho_inf_, u_inf_, v_inf_, w_inf_, p_inf_;
    PetscReal c_inf_;   // 来流声速 (预存)
};

// ====================================================================
// BCExtrapolateFiller — 零阶外推 (回退用)
// ====================================================================
class BCExtrapolateFiller : public BCFaceFiller {
public:
    const char* name() const override { return "BCExtrapolate"; }
    void fillGhostFace(Mesh* mesh, int face,
                       PetscReal*** u[5],
                       const DMDALocalInfo& info) override;
};

#endif
