#ifndef SIM_CONFIG_H
#define SIM_CONFIG_H

#include <petscsys.h>
#include <vector>
#include <array>
#include <string>

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

    const std::string& getGridFile()   const { return grid_file_; }
    PetscInt           getGridFormat() const { return grid_format_; }
    PetscInt           getSchemeVis()  const { return scheme_vis_; }
    PetscInt           getLAP()        const { return LAP_; }
    PetscInt           getFaceGtype()  const { return face_gtype_; }
    PetscInt           getEdgeGtype()  const { return edge_gtype_; }
    PetscInt           getMetricGtype()const { return metric_gtype_; }
    const std::vector<std::array<PetscInt,3>>& getProcs() const { return procs_; }
    bool               isProcsAuto()   const { return procs_auto_; }

    void print() const;

private:
    PetscInt grid_format_ = 1;
    std::string grid_file_   = "1.dat";
    PetscInt    scheme_vis_  = 8;
    PetscInt    LAP_         = 0;
    PetscInt    face_gtype_  = 2;
    PetscInt    edge_gtype_  = 2;
    PetscInt    metric_gtype_= 0;
    std::vector<std::array<PetscInt,3>> procs_;
    bool procs_auto_ = false;

    void parse(const std::string &filename);
    void applyDefaults();

    static std::string trim(const std::string &s);
    static PetscInt  toInt(const std::string &s);
    static PetscInt  parseScheme(const std::string &s);
    static std::vector<std::array<PetscInt,3>> parseProcs(const std::string &value);
};

#endif
