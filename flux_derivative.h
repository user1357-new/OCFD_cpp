#ifndef FLUX_DERIVATIVE_H
#define FLUX_DERIVATIVE_H

#include <petscdmda.h>
#include <string>
#include <map>
#include <memory>

// ====================================================================
// FluxDerivative — 通量导数计算抽象基类
//
// 对填好 ghost 的三维通量数组沿计算空间方向 (ξ/η/ζ) 求导。
//
// 约定（与 mesh_jacobi.cpp 中的 compute_derivative_x 一致）：
//   f  : 输入，global 索引 [k][j][i]（含 ghost，来自 DMDA local Vec）
//   df : 输出，global 索引 [k][j][i]（仅填充内部格心，来自 DMDA local Vec）
//
// 步长 hξ/hη/hζ = 1/N（N 为该方向格心数），从 DM 获取。
// ====================================================================
class FluxDerivative {
public:
    virtual ~FluxDerivative() = default;

    /// @brief ξ 方向导数（沿 i 索引）
    virtual void deriv_xi(DM da, PetscReal ***f, PetscReal ***df) = 0;

    /// @brief η 方向导数（沿 j 索引）
    virtual void deriv_eta(DM da, PetscReal ***f, PetscReal ***df) = 0;

    /// @brief ζ 方向导数（沿 k 索引）
    virtual void deriv_zeta(DM da, PetscReal ***f, PetscReal ***df) = 0;

    virtual const char* name() const = 0;
};

// ====================================================================
// CD6FluxDerivative — 全场统一 6 阶中心差分
// ====================================================================
class CD6FluxDerivative : public FluxDerivative {
public:
    const char* name() const override { return "CD6"; }
    void deriv_xi(DM da, PetscReal ***f, PetscReal ***df) override;
    void deriv_eta(DM da, PetscReal ***f, PetscReal ***df) override;
    void deriv_zeta(DM da, PetscReal ***f, PetscReal ***df) override;
};

// ====================================================================
// CD8FluxDerivative — 全场统一 8 阶中心差分
// ====================================================================
class CD8FluxDerivative : public FluxDerivative {
public:
    const char* name() const override { return "CD8"; }
    void deriv_xi(DM da, PetscReal ***f, PetscReal ***df) override;
    void deriv_eta(DM da, PetscReal ***f, PetscReal ***df) override;
    void deriv_zeta(DM da, PetscReal ***f, PetscReal ***df) override;
};

// ====================================================================
// DerivativeRegistry — 导数方案注册表（单例）
// ====================================================================
class DerivativeRegistry {
public:
    static DerivativeRegistry& instance();

    void add(const std::string& type, std::unique_ptr<FluxDerivative> d);
    FluxDerivative* get(const std::string& type);
    bool has(const std::string& type) const;

private:
    DerivativeRegistry() = default;
    std::map<std::string, std::unique_ptr<FluxDerivative>> registry_;
};

/// @brief 简单工厂：从字符串创建空间导数算子
FluxDerivative* createDerivative(const std::string& name);

#endif
