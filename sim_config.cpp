#include "sim_config.h"
#include "mesh.h"       
#include <petscsys.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

SimConfig::SimConfig(const std::string &input_file)
{
    parse(input_file);
    applyDefaults();
}

// ========== 解析 ==========
void SimConfig::parse(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open input file: " + filename);

    std::string line;
    while (std::getline(file, line)) {
        // 去掉行内注释
        size_t comment = line.find('#');
        if (comment != std::string::npos)
            line = line.substr(0, comment);

        line = trim(line);
        if (line.empty()) continue;

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key   = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));

        if      (key == "grid_file")    grid_file_    = value;
        else if (key == "grid_format")  grid_format_ = toInt(value);
        else if (key == "scheme_vis")   scheme_vis_   = parseScheme(value);
        else if (key == "lap")          LAP_          = toInt(value);
        else if (key == "face_gtype")   face_gtype_   = toInt(value);
        else if (key == "edge_gtype")   edge_gtype_   = toInt(value);
        else if (key == "metric_gtype") metric_gtype_ = toInt(value);
        else if (key == "procs")        procs_        = parseProcs(value);
    }

    if (procs_.empty())
        procs_auto_ = true;
}

// ========== 自动填充默认值 ==========
void SimConfig::applyDefaults()
{
    if (LAP_ == 0)
        LAP_ = (scheme_vis_ == OCFD_Scheme_CD8) ? 4 : 3;

    if (metric_gtype_ == 0)
        metric_gtype_ = face_gtype_;
}

// ========== 解析 scheme ==========
PetscInt SimConfig::parseScheme(const std::string &s)
{
    std::string upper = s;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (upper == "CD6") return OCFD_Scheme_CD6;
    if (upper == "CD8") return OCFD_Scheme_CD8;
    int val = toInt(s);
    if (val == 6 || val == 8) return val;
    throw std::runtime_error("Unknown scheme_vis: " + s + " (expected CD6 or CD8)");
}

// ========== 解析 procs ==========
std::vector<std::array<PetscInt,3>> SimConfig::parseProcs(const std::string &value)
{
    std::vector<std::array<PetscInt,3>> result;
    std::istringstream iss(value);
    std::string block;

    while (std::getline(iss, block, '|')) {
        block = trim(block);
        if (block.empty()) continue;

        std::istringstream bs(block);
        std::string token;
        std::array<PetscInt,3> arr = {1, 1, 1};
        int idx = 0;

        while (std::getline(bs, token, ',') && idx < 3) {
            arr[idx++] = toInt(trim(token));
        }
        if (idx != 3)
            throw std::runtime_error("Invalid procs entry: " + block + " (need 3 integers)");
        result.push_back(arr);
    }
    return result;
}

// ========== 工具函数 ==========
PetscInt SimConfig::toInt(const std::string &s)
{
    return std::stoi(trim(s));
}

std::string SimConfig::trim(const std::string &s)
{
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

// ========== 打印 ==========
void SimConfig::print() const
{
    std::string procs_str;
    if (procs_auto_) {
        procs_str = "auto";
    } else {
        for (size_t i = 0; i < procs_.size(); ++i) {
            if (i > 0) procs_str += " | ";
            procs_str += std::to_string(procs_[i][0]) + ","
                       + std::to_string(procs_[i][1]) + ","
                       + std::to_string(procs_[i][2]);
        }
    }

    PetscPrintf(PETSC_COMM_WORLD,
        "===== Simulation Config =====\n"
        "  grid_file    = %s\n"
        "  grid_format  = %d (%s)\n"
        "  scheme_vis   = %d (CD%d)\n"
        "  LAP          = %d\n"
        "  face_gtype   = %d\n"
        "  edge_gtype   = %d\n"
        "  metric_gtype = %d\n"
        "  procs (%d blocks) = %s\n"
        "==============================\n",
        grid_file_.c_str(),
        (int)grid_format_, (grid_format_ == 0 ? "Plot3D" : (grid_format_ == 2 ? "CGNS" : "Tecplot")),
        (int)scheme_vis_, (int)scheme_vis_,
        (int)LAP_,
        (int)face_gtype_,
        (int)edge_gtype_,
        (int)metric_gtype_,
        (int)(procs_auto_ ? 0 : procs_.size()),
        procs_str.c_str());
}
