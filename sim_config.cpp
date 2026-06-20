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

void SimConfig::parse(const std::string &filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("Cannot open input file: " + filename);

    std::string line;
    while (std::getline(file, line)) {
        size_t comment = line.find('#');
        if (comment != std::string::npos)
            line = line.substr(0, comment);

        line = trim(line);
        if (line.empty()) continue;

        size_t eq = line.find('=');
        if (eq == std::string::npos) continue;

        std::string key   = trim(line.substr(0, eq));
        std::string value = trim(line.substr(eq + 1));

        if      (key == "grid_file")        grid_file_        = value;
        else if (key == "grid_format")      grid_format_      = toInt(value);
        else if (key == "scheme_vis")       scheme_vis_       = parseScheme(value);
        else if (key == "lap")              LAP_              = toInt(value);
        else if (key == "face_gtype")       face_gtype_       = toInt(value);
        else if (key == "edge_gtype")       edge_gtype_       = toInt(value);
        else if (key == "metric_gtype")     metric_gtype_     = toInt(value);
        else if (key == "metric_diff")      metric_diff_type_ = parseMetricDiff(value);
        else if (key == "init_type")        init_type_        = value;
        else if (key == "init_file")        init_file_        = value;
        else if (key == "restart_file")     restart_file_     = value;
        else if (key == "mach")             mach_             = toReal(value);
        else if (key == "gamma")            gamma_            = toReal(value);
        else if (key == "attack")           attack_           = toReal(value);
        else if (key == "sideslip")         sideslip_         = toReal(value);
        else if (key == "periodic_span") {
            std::istringstream ps(value);
            std::string tok;
            for (int i = 0; i < 3; ++i) {
                if (!std::getline(ps, tok, ',')) break;
                periodic_span_[i] = toReal(trim(tok));
            }
        }
        else if (key == "inlet_rho")      inlet_rho_        = toReal(value);
        else if (key == "inlet_u")        inlet_u_          = toReal(value);
        else if (key == "inlet_v")        inlet_v_          = toReal(value);
        else if (key == "inlet_w")        inlet_w_          = toReal(value);
        else if (key == "inlet_p")        inlet_p_          = toReal(value);
        else if (key == "k_x")           k_x_              = toReal(value);
        else if (key == "k_y")           k_y_              = toReal(value);
        else if (key == "k_z")           k_z_              = toReal(value);
        else if (key == "bc") {
            bc_overrides_  = parseBCOverrides(value);
            periodic_pairs_ = parsePeriodicPairs(value);
        }
        else if (key == "procs")            procs_            = parseProcs(value);
        else if (key == "base_rho")          base_rho_         = toReal(value);
        else if (key == "base_p")            base_p_           = toReal(value);
        else if (key == "amp")               amp_              = toReal(value);
        else if (key == "spatial_derivative") spatial_derivative_ = value;
        else if (key == "time_integrator")    time_integrator_   = value;
        else if (key == "cfl")              cfl_              = toReal(value);
        else if (key == "max_steps")        max_steps_        = toInt(value);
        else if (key == "output_interval")  output_interval_  = toInt(value);
    }

    if (procs_.empty())
        procs_auto_ = true;
}

void SimConfig::applyDefaults()
{
    if (LAP_ == 0)
        LAP_ = (scheme_vis_ == OCFD_Scheme_CD8) ? 4 : 3;

    if (metric_gtype_ == 0)
        metric_gtype_ = face_gtype_;
    // metric_diff_type_ 默认为 0（由 parse 中 toInt 保证）

    if (restart_file_.empty())
        restart_file_ = "restart.dat";

    // Inlet BC 默认值
    if (inlet_rho_ == 0.0) inlet_rho_ = 1.0;
    if (inlet_p_   == 0.0) inlet_p_   = 1.0;

    // Linearized Euler base state 默认值
    if (base_rho_ == 0.0) base_rho_ = 1.0;
    if (base_p_   == 0.0) base_p_   = 1.0 / (gamma_ * mach_ * mach_);  // 从 Ma 反算
    base_c0_ = std::sqrt(gamma_ * base_p_ / base_rho_);
    if (amp_ == 0.0) amp_ = 1e-4;  // 线性声波小振幅

    // k_x/y/z 默认为 0 → sinusoidalInit 内自动按域尺寸算

    // Solver 默认值
    if (spatial_derivative_.empty()) spatial_derivative_  = "CD8";
    if (time_integrator_.empty())    time_integrator_    = "RK3";
    if (cfl_ == 0.0)                 cfl_                = 0.5;
    if (max_steps_ == 0)            max_steps_          = 1000;
    if (output_interval_ == 0)      output_interval_    = 100;

    // 将 periodic_span 填入每个 PeriodicPair
    for (auto& pp : periodic_pairs_) {
        pp.translate[0] = periodic_span_[0];
        pp.translate[1] = periodic_span_[1];
        pp.translate[2] = periodic_span_[2];
    }
}

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

PetscInt SimConfig::parseMetricDiff(const std::string &s)
{
    std::string lower = s;
    std::transform(lower.begin(), lower.end(), lower.begin(), ::tolower);
    if (lower == "uniform") return METRIC_DIFF_UNIFORM;
    if (lower == "reduced") return METRIC_DIFF_REDUCED;
    throw std::runtime_error("Unknown metric_diff: " + s + " (expected uniform or reduced)");
}

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

PetscInt SimConfig::toInt(const std::string &s)
{
    return std::stoi(trim(s));
}

PetscReal SimConfig::toReal(const std::string &s)
{
    return std::stod(trim(s));
}

std::string SimConfig::trim(const std::string &s)
{
    size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

int SimConfig::parseFaceName(const std::string &s)
{
    std::string upper = s;
    std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);
    if (upper == "LEFT")   return 0;
    if (upper == "RIGHT")  return 1;
    if (upper == "BOTTOM") return 2;
    if (upper == "TOP")    return 3;
    if (upper == "BACK")   return 4;
    if (upper == "FRONT")  return 5;
    return -1;
}

std::vector<BCOverride> SimConfig::parseBCOverrides(const std::string &value)
{
    std::vector<BCOverride> result;
    std::istringstream iss(value);
    std::string token;

    while (std::getline(iss, token, '|')) {
        token = trim(token);
        if (token.empty()) continue;

        std::istringstream ts(token);
        std::string part;
        std::vector<std::string> parts;
        while (std::getline(ts, part, ','))
            parts.push_back(trim(part));

        if (parts.size() < 3)
            throw std::runtime_error("Invalid bc override: " + token + " (need block,face,type)");

        // ★ BCPeriodic 条目由 parsePeriodicPairs 单独处理
        if (parts[2].find("BCPeriodic") == 0)
            continue;

        BCOverride ov;
        ov.block   = toInt(parts[0]);
        ov.face    = parseFaceName(parts[1]);
        ov.bc_type = parts[2];

        if (ov.face < 0)
            throw std::runtime_error("Unknown face name: " + parts[1]);

        result.push_back(ov);
    }
    return result;
}

std::vector<PeriodicPair> SimConfig::parsePeriodicPairs(const std::string &value)
{
    std::vector<PeriodicPair> result;
    std::istringstream iss(value);
    std::string token;

    while (std::getline(iss, token, '|')) {
        token = trim(token);
        if (token.empty()) continue;

        std::istringstream ts(token);
        std::string part;
        std::vector<std::string> parts;
        while (std::getline(ts, part, ','))
            parts.push_back(trim(part));

        if (parts.size() < 3) continue;
        if (parts[2].find("BCPeriodic") != 0) continue;

        // BCPeriodic 或 BCPeriodic:block,face
        // 注意：冒号后的逗号会被外层 split 切开，所以 parts[2]="BCPeriodic:target_block", parts[3]="target_face"
        std::string bc_spec = parts[2];
        size_t colon = bc_spec.find(':');
        if (colon == std::string::npos || colon + 1 >= bc_spec.size())
            throw std::runtime_error("BCPeriodic requires target: 'BCPeriodic:block,face' in: " + token);

        if (parts.size() < 4)
            throw std::runtime_error("BCPeriodic requires 'block,face' after colon in: " + token);

        int target_block = toInt(trim(bc_spec.substr(colon + 1)));
        int target_face  = parseFaceName(trim(parts[3]));

        if (target_face < 0)
            throw std::runtime_error("Unknown target face in BCPeriodic: " + parts[3]);

        PeriodicPair pp;
        pp.block_a = toInt(parts[0]);
        pp.face_a  = parseFaceName(parts[1]);
        pp.block_b = target_block;
        pp.face_b  = target_face;

        if (pp.face_a < 0)
            throw std::runtime_error("Unknown face name: " + parts[1]);

        // 不自连: (b,f) → (b,f)
        if (pp.block_a == pp.block_b && pp.face_a == pp.face_b)
            throw std::runtime_error("BCPeriodic cannot connect a face to itself: " + token);

        result.push_back(pp);
    }
    return result;
}

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
        "  grid_file     = %s\n"
        "  grid_format   = %d (%s)\n"
        "  scheme_vis    = %d (CD%d)\n"
        "  LAP           = %d\n"
        "  face_gtype    = %d\n"
        "  edge_gtype    = %d\n"
        "  metric_gtype  = %d\n"
        "  metric_diff   = %s\n"
        "  init_type     = %s\n"
        "  init_file     = %s\n"
        "  restart_file  = %s\n"
        "  mach          = %.4f\n"
        "  gamma         = %.4f\n"
        "  attack        = %.4f\n"
        "  sideslip      = %.4f\n"
        "  inlet_rho     = %.4f\n"
        "  inlet_u       = %.4f\n"
        "  inlet_v       = %.4f\n"
        "  inlet_w       = %.4f\n"
        "  inlet_p       = %.4f\n"
        "  base_rho      = %.6f\n"
        "  base_p        = %.6f\n"
        "  base_c0       = %.6f\n"
        "  amp           = %.2e\n"
        "  k_x            = %.4f\n"
        "  k_y            = %.4f\n"
        "  k_z            = %.4f\n"
        "  periodic_span  = %.4f, %.4f, %.4f\n"
        "  procs (%d blocks) = %s\n"
        "  --- Solver ---\n"
        "  spatial_derivative = %s\n"
        "  time_integrator    = %s\n"
        "  cfl                = %.4f\n"
        "  max_steps          = %d\n"
        "  output_interval    = %d\n"
        "=============================\n",
        grid_file_.c_str(),
        (int)grid_format_, (grid_format_ == 0 ? "Plot3D" : (grid_format_ == 2 ? "CGNS" : "Tecplot")),
        (int)scheme_vis_, (int)scheme_vis_,
        (int)LAP_,
        (int)face_gtype_,
        (int)edge_gtype_,
        (int)metric_gtype_,
        (metric_diff_type_ == METRIC_DIFF_UNIFORM ? "uniform" : "reduced"),
        init_type_.c_str(),
        init_file_.c_str(),
        restart_file_.c_str(),
        (double)mach_, (double)gamma_, (double)attack_, (double)sideslip_,
        (double)inlet_rho_, (double)inlet_u_, (double)inlet_v_,
        (double)inlet_w_, (double)inlet_p_,
        (double)base_rho_, (double)base_p_, (double)base_c0_, (double)amp_,
        (double)k_x_, (double)k_y_, (double)k_z_,
        (double)periodic_span_[0], (double)periodic_span_[1], (double)periodic_span_[2],
        (int)(procs_auto_ ? 0 : procs_.size()),
        procs_str.c_str(),
        spatial_derivative_.c_str(),
        time_integrator_.c_str(),
        (double)cfl_,
        (int)max_steps_,
        (int)output_interval_);

    // BC overrides
    if (!bc_overrides_.empty()) {
        PetscPrintf(PETSC_COMM_WORLD, "  bc_overrides  = %d entries:\n", (int)bc_overrides_.size());
        const char* face_names[6] = {"LEFT","RIGHT","BOTTOM","TOP","BACK","FRONT"};
        for (const auto& ov : bc_overrides_)
            PetscPrintf(PETSC_COMM_WORLD, "    block %d  %s  → %s\n",
                       ov.block, face_names[ov.face], ov.bc_type.c_str());
    }
    // Periodic pairs
    if (!periodic_pairs_.empty()) {
        PetscPrintf(PETSC_COMM_WORLD, "  periodic      = %d pairs:\n", (int)periodic_pairs_.size());
        const char* face_names[6] = {"LEFT","RIGHT","BOTTOM","TOP","BACK","FRONT"};
        for (const auto& pp : periodic_pairs_)
            PetscPrintf(PETSC_COMM_WORLD, "    block %d %s  ↔  block %d %s\n",
                       pp.block_a, face_names[pp.face_a],
                       pp.block_b, face_names[pp.face_b]);
    }
}
