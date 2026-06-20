#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

#include <petscsys.h>
#include <vector>
#include <array>
#include <string>
#include <map>

struct BCOverride {
    int block;
    int face;
    std::string bc_type;
};

/// @brief 周期性边界对（文本输入）
struct PeriodicPair {
    int block_a, face_a;
    int block_b, face_b;
    PetscReal translate[3] = {0,0,0};  // MIN面→MAX面物理平移向量
};

/// @brief 从 key=value 格式的 txt 读取仿真参数
///
/// 文件格式（# 到行尾为注释，空行跳过）：
///   grid_file    = 1.dat
///   scheme_vis   = CD8        （CD6 或 CD8）
///   face_gtype   = 2          （1=一阶线性, 2=二阶镜像）
///   edge_gtype   = 2
///   metric_gtype = 2          （缺省 = face_gtype）
///   lap          = 4          （缺省由 scheme_vis 推导：CD8→4, CD6→3）
///   procs        = 2,2,1 | 2,1,1 | ...
class SimConfig {
public:
    SimConfig(const std::string &input_file);

    const std::string& getGridFile()      const { return grid_file_; }
    PetscInt           getGridFormat()    const { return grid_format_; }
    PetscInt           getSchemeVis()     const { return scheme_vis_; }
    PetscInt           getLAP()           const { return LAP_; }
    PetscInt           getFaceGtype()     const { return face_gtype_; }
    PetscInt           getEdgeGtype()     const { return edge_gtype_; }
    PetscInt           getMetricGtype()   const { return metric_gtype_; }
    PetscInt           getMetricDiffType()const { return metric_diff_type_; }
    const std::string& getInitType()      const { return init_type_; }
    const std::string& getInitFile()      const { return init_file_; }
    const std::string& getRestartFile()   const { return restart_file_; }
    PetscReal          getMach()          const { return mach_; }
    PetscReal          getGamma()         const { return gamma_; }
    PetscReal          getAttack()        const { return attack_; }
    PetscReal          getSideslip()      const { return sideslip_; }

    // Inlet BC
    PetscReal          getInletRho()      const { return inlet_rho_; }
    PetscReal          getInletU()        const { return inlet_u_; }
    PetscReal          getInletV()        const { return inlet_v_; }
    PetscReal          getInletW()        const { return inlet_w_; }
    PetscReal          getInletP()        const { return inlet_p_; }

    // Sinusoidal init wave numbers (0 = auto from domain size)
    PetscReal          getKx()            const { return k_x_; }
    PetscReal          getKy()            const { return k_y_; }
    PetscReal          getKz()            const { return k_z_; }

    const std::vector<std::array<PetscInt,3>>& getProcs() const { return procs_; }
    bool isProcsAuto() const { return procs_auto_; }

    const std::vector<BCOverride>& getBCOverrides() const { return bc_overrides_; }
    const std::vector<PeriodicPair>& getPeriodicPairs() const { return periodic_pairs_; }
    const PetscReal* getPeriodicSpan() const { return periodic_span_; }

    void print() const;

private:
    PetscInt    grid_format_;
    std::string grid_file_;
    PetscInt    scheme_vis_;
    PetscInt    LAP_;
    PetscInt    face_gtype_;
    PetscInt    edge_gtype_;
    PetscInt    metric_gtype_;
    PetscInt    metric_diff_type_;
    std::string init_type_;
    std::string init_file_;
    std::string restart_file_;
    PetscReal   mach_;
    PetscReal   gamma_;
    PetscReal   attack_;
    PetscReal   sideslip_;

    // Inlet BC
    PetscReal   inlet_rho_;
    PetscReal   inlet_u_;
    PetscReal   inlet_v_;
    PetscReal   inlet_w_;
    PetscReal   inlet_p_;

    // Sinusoidal init
    PetscReal   k_x_ = 0.0;
    PetscReal   k_y_ = 0.0;
    PetscReal   k_z_ = 0.0;

    std::vector<std::array<PetscInt,3>> procs_;
    bool procs_auto_;

    // BC override
    std::vector<BCOverride> bc_overrides_;
    std::vector<PeriodicPair> periodic_pairs_;
    PetscReal periodic_span_[3] = {0,0,0};

    void parse(const std::string &filename);
    void applyDefaults();

    static std::string trim(const std::string &s);
    static PetscInt  toInt(const std::string &s);
    static PetscReal toReal(const std::string &s);
    static PetscInt  parseScheme(const std::string &s);
    static PetscInt  parseMetricDiff(const std::string &s);
    static std::vector<std::array<PetscInt,3>> parseProcs(const std::string &value);
    static std::vector<BCOverride> parseBCOverrides(const std::string &value);
    static std::vector<PeriodicPair> parsePeriodicPairs(const std::string &value);
    static int parseFaceName(const std::string &s);
};

#endif
