#include "mesh.h"
#include <petscdmda.h>
#include <petscvec.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <string>
#include "mesh_mutiblock.h"
#include <sstream>
#include <algorithm>
#include <cctype>
#include <petsc.h>
#include <stdexcept>
#include <memory>
#include <cstdint>    // int32_t, uint8_t, uint16_t
#include <cgnslib.h>  // CGNS library
#include <map>


//找到 "T=" 跳过空格和等号，读取引号内内容
std::string MultiBlockMesh::extract_string(const std::string &line, const std::string &key) {
    size_t pos = line.find(key);
    if (pos == std::string::npos) return "";
    pos += key.size();
    while (pos < line.size() && (line[pos] == ' ' || line[pos] == '=')) pos++;
    if (pos >= line.size()) return "";
    if (line[pos] == '\"') {
        size_t end = line.find('\"', pos+1);
        if (end != std::string::npos)
            return line.substr(pos+1, end-pos-1);
        else
            return line.substr(pos+1);
    } else {
        std::istringstream iss(line.substr(pos));
        std::string token;
        iss >> token;
        return token;
    }
}
//找到 "I=/J=/K=" 跳过等号，读取后面内容
PetscInt MultiBlockMesh::extract_int(const std::string &line, const std::string &key) {
    size_t pos = line.find(key);
    if (pos == std::string::npos) return 0;
    pos += key.size();
    while (pos < line.size() && line[pos] == ' ') pos++;
    std::string num;
    while (pos < line.size() && std::isdigit(line[pos])) {
        num += line[pos++];
    }
    return num.empty() ? 0 : std::stoi(num);
}
//读取数据，每个zone为一个块
void MultiBlockMesh::ParseTecplotFile(const std::string &filename,
                                      std::vector<BlockInfo> &infos) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open multi-block file: " + filename);
    }

    std::string line;
    BlockInfo cur;
    bool in_data = false;      // 正在读数据
    bool need_ijk = false;     // 正在等尺寸行
    int total = 0, count = 0;

    while (std::getline(file, line)) {
        // 去首尾空格
        size_t s = line.find_first_not_of(" \t\r\n");
        if (s == std::string::npos) continue;
        size_t e = line.find_last_not_of(" \t\r\n");
        line = line.substr(s, e - s + 1);

        if (line.empty() || line[0] == '#') continue;

        std::string upper = line;
        std::transform(upper.begin(), upper.end(), upper.begin(), ::toupper);

        // 跳过全局头行
        if (upper.find("TITLE") == 0 ||
            upper.find("VARIABLES") == 0 ||
            upper.find("DATASETAUXDATA") == 0 ||
            upper.find("DT=") != std::string::npos) {
            continue;
        }

        // 如果正在等尺寸
        if (need_ijk) {
            if (upper.find("I=") != std::string::npos &&
                upper.find("J=") != std::string::npos) {
                cur.ni = extract_int(line, "I=");
                cur.nj = extract_int(line, "J=");
                cur.nk = extract_int(line, "K=");
                if (cur.ni > 0 && cur.nj > 0 && cur.nk > 0) {
                    total = cur.ni * cur.nj * cur.nk;
                    cur.x.reserve(total);
                    cur.y.reserve(total);
                    cur.z.reserve(total);
                    need_ijk = false;
                    in_data = true;
                    count = 0;
                }
            }
            continue;
        }

        // 发现新块：ZONE T=
        if (upper.find("ZONE T=") != std::string::npos) {
            // 先把上一个块入库
            if (in_data && count == total) {
                infos.push_back(cur);
            }
            cur = BlockInfo();
            cur.name = extract_string(line, "T=");
            // 尝试从本行直接拿尺寸
            cur.ni = extract_int(line, "I=");
            cur.nj = extract_int(line, "J=");
            cur.nk = extract_int(line, "K=");
            if (cur.ni > 0 && cur.nj > 0 && cur.nk > 0) {
                total = cur.ni * cur.nj * cur.nk;
                cur.x.reserve(total);
                cur.y.reserve(total);
                cur.z.reserve(total);
                in_data = true;
                count = 0;
            } else {
                // 尺寸不在本行，等下一行
                need_ijk = true;
                in_data = false;
            }
            continue;
        }

        // 读数据
        if (in_data) {
            std::istringstream iss(line);
            double x, y, z;
            if (iss >> x >> y >> z) {
                cur.x.push_back(x);
                cur.y.push_back(y);
                cur.z.push_back(z);
                count++;
                if (count == total) {
                    infos.push_back(cur);
                    in_data = false;
                }
            }
        }
    }

    // 最后一组数据可能没关闭
    if (in_data && count == total) {
        infos.push_back(cur);
    }
}

void MultiBlockMesh::ParsePlot3DFile(const std::string &filename,
                                     std::vector<BlockInfo> &infos)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("Cannot open Plot3D file: " + filename);

    // ========== 字节序检测 ==========
    auto swap_int32 = [](int32_t v) -> int32_t {
        return ((v & 0xFF) << 24) | ((v & 0xFF00) << 8) |
               ((v >> 8) & 0xFF00) | ((v >> 24) & 0xFF);
    };

    int32_t raw[3];
    file.read(reinterpret_cast<char*>(raw), 12);
    file.seekg(0);

    bool file_le = true;
    {
        int32_t w1_be = swap_int32(raw[0]);
        if (w1_be == 4 && raw[0] != 4) file_le = false;
        else if (w1_be > 0 && w1_be < 100000 && (raw[0] <= 0 || raw[0] > 100000))
            file_le = false;
    }

    auto to_i32 = [&](int32_t v) -> int32_t {
        return file_le ? v : swap_int32(v);
    };

    int32_t marker0 = to_i32(raw[0]);
    int32_t nblocks  = to_i32(raw[1]);
    int32_t tail0    = to_i32(raw[2]);

    bool has_markers = (marker0 == 4 && nblocks > 0 && nblocks < 100000 && tail0 == 4);

    if (nblocks <= 0 || nblocks > 100000)
        throw std::runtime_error("Plot3D: bad nblocks=" + std::to_string(nblocks));

    // ========== 读取一条 Fortran record ==========
    auto read_fortran_record = [&](std::vector<char> &buf) -> bool {
        int32_t rl_raw;
        if (!file.read(reinterpret_cast<char*>(&rl_raw), 4)) return false;
        int32_t rl = to_i32(rl_raw);
        if (rl < 0 || rl > 500 * 1024 * 1024) {
            std::cerr << "BAD rec_len=" << rl << std::endl;
            return false;
        }
        buf.resize(rl);
        if (!file.read(buf.data(), rl)) return false;
        int32_t tr;
        file.read(reinterpret_cast<char*>(&tr), 4);
        return true;
    };

    std::vector<char> rec;

    // ========== 跳过头部的 nblocks record ==========
    if (has_markers) {
        if (!read_fortran_record(rec))
            throw std::runtime_error("Plot3D: failed on nblocks record");
        // nblocks 已经从头部分析得到，这里校验一下
        int32_t nb2 = to_i32(*reinterpret_cast<int32_t*>(rec.data()));
        if (nb2 != nblocks)
            std::cerr << "WARNING: nblocks mismatch: " << nblocks << " vs " << nb2 << std::endl;
    }
    //  你的文件：所有 6 块的尺寸打包在 1 个 72 字节的 record 中
    //  标准文件：每个块的尺寸各占 1 个 12 字节的 record
    struct BlockDims { int32_t ni, nj, nk; };
    std::vector<BlockDims> all_dims;

    if (has_markers) {
        // 读第一条 dims record
        std::streampos dims_pos = file.tellg();
        if (!read_fortran_record(rec))
            throw std::runtime_error("Plot3D: failed to read dims");

        int32_t dims_bytes = static_cast<int32_t>(rec.size());
        int32_t nints = dims_bytes / 4;

        if (nints == 3 * nblocks) {
            // ★ 你的文件格式：所有块尺寸在一个 record 里
            int32_t *d = reinterpret_cast<int32_t*>(rec.data());
            for (int32_t b = 0; b < nblocks; ++b) {
                BlockDims bd;
                bd.ni = to_i32(d[3*b]);
                bd.nj = to_i32(d[3*b+1]);
                bd.nk = to_i32(d[3*b+2]);
                all_dims.push_back(bd);
            }
        } else if (nints == 3) {
            // 标准格式：一个块尺寸占一个 record
            int32_t *d = reinterpret_cast<int32_t*>(rec.data());
            all_dims.push_back({to_i32(d[0]), to_i32(d[1]), to_i32(d[2])});

            // 继续读剩余块
            for (int32_t b = 1; b < nblocks; ++b) {
                if (!read_fortran_record(rec))
                    throw std::runtime_error("Plot3D: dims for block " +
                                             std::to_string(b+1));
                d = reinterpret_cast<int32_t*>(rec.data());
                all_dims.push_back({to_i32(d[0]), to_i32(d[1]), to_i32(d[2])});
            }
        } else {
            throw std::runtime_error("Plot3D: unexpected dims record size=" +
                                     std::to_string(dims_bytes) +
                                     " (nblocks=" + std::to_string(nblocks) + ")");
        }
    } else {
        // 无 markers：直接读 12×nblocks 字节
        std::vector<int32_t> dims_buf(3 * nblocks);
        file.read(reinterpret_cast<char*>(dims_buf.data()), 3 * nblocks * 4);
        for (int32_t b = 0; b < nblocks; ++b) {
            all_dims.push_back({
                to_i32(dims_buf[3*b]),
                to_i32(dims_buf[3*b+1]),
                to_i32(dims_buf[3*b+2])
            });
        }
    }

    // 验证尺寸
    for (int32_t b = 0; b < nblocks; ++b) {
        auto &bd = all_dims[b];
        if (bd.ni <= 0 || bd.nj <= 0 || bd.nk <= 0)
            throw std::runtime_error("Plot3D: block " + std::to_string(b+1) +
                                     " bad dims: " + std::to_string(bd.ni) + "," +
                                     std::to_string(bd.nj) + "," + std::to_string(bd.nk));
    }

    // ========== 读取坐标 ==========
    infos.reserve(nblocks);

    // 计算所有块的总浮点数（用于判断是否打包）
    int64_t total_floats = 0;
    for (auto &bd : all_dims)
        total_floats += static_cast<int64_t>(bd.ni) * bd.nj * bd.nk * 3;

    for (int32_t b = 0; b < nblocks; ++b) {
        BlockInfo cur;
        cur.name = "block_" + std::to_string(b + 1);
        cur.ni = all_dims[b].ni;
        cur.nj = all_dims[b].nj;
        cur.nk = all_dims[b].nk;

        PetscInt N = static_cast<PetscInt>(cur.ni) * cur.nj * cur.nk;
        cur.x.resize(N);
        cur.y.resize(N);
        cur.z.resize(N);

        if (has_markers) {
            // 探测坐标精度和打包方式
            std::streampos before = file.tellg();
            int32_t peek_raw;
            file.read(reinterpret_cast<char*>(&peek_raw), 4);
            file.seekg(before);
            int32_t peek_len = to_i32(peek_raw);

            int32_t f32_1 = static_cast<int32_t>(N * 4);       // float, 1分量
            int32_t f32_3 = static_cast<int32_t>(N * 12);      // float, 3分量合一
            int32_t f64_1 = static_cast<int32_t>(N * 8);       // double, 1分量
            int32_t f64_3 = static_cast<int32_t>(N * 24);      // double, 3分量合一

            // 判断是否 3 分量合一
            if (peek_len == f32_3 || peek_len == f64_3) {
                // 三合一
                if (!read_fortran_record(rec))
                    throw std::runtime_error("Plot3D: coords block " + std::to_string(b+1));
                if (peek_len == f64_3) {
                    double *p = reinterpret_cast<double*>(rec.data());
                    for (PetscInt i = 0; i < N; ++i) {
                        cur.x[i] = p[i];
                        cur.y[i] = p[i + N];
                        cur.z[i] = p[i + 2*N];
                    }
                } else {
                    float *p = reinterpret_cast<float*>(rec.data());
                    for (PetscInt i = 0; i < N; ++i) {
                        cur.x[i] = p[i];
                        cur.y[i] = p[i + N];
                        cur.z[i] = p[i + 2*N];
                    }
                }
            } else if (peek_len == f64_1 || peek_len == f32_1) {
                // 三分量独立
                auto read_one = [&](std::vector<double> &dst) -> bool {
                    if (!read_fortran_record(rec)) return false;
                    if (peek_len == f64_1) {
                        double *p = reinterpret_cast<double*>(rec.data());
                        for (PetscInt i = 0; i < N; ++i) dst[i] = p[i];
                    } else {
                        float *p = reinterpret_cast<float*>(rec.data());
                        for (PetscInt i = 0; i < N; ++i) dst[i] = p[i];
                    }
                    return true;
                };
                if (!read_one(cur.x) || !read_one(cur.y) || !read_one(cur.z))
                    throw std::runtime_error("Plot3D: coords block " + std::to_string(b+1));
            } else {
                // peek_len 不匹配单块 → 可能是多块打包
                // 多块 float 三合一
                int64_t remaining_floats = 0;
                for (int32_t bb = b; bb < nblocks; ++bb) {
                    auto &bd2 = all_dims[bb];
                    remaining_floats += static_cast<int64_t>(bd2.ni) * bd2.nj * bd2.nk * 3;
                }
                int32_t remaining_bytes = static_cast<int32_t>(remaining_floats * 4);

                if (peek_len == remaining_bytes) {
                    // 剩下的所有块打包在一起
                    if (!read_fortran_record(rec))
                        throw std::runtime_error("Plot3D: packed coords");
                    float *p = reinterpret_cast<float*>(rec.data());
                    PetscInt offset = 0;
                    for (int32_t bb = b; bb < nblocks; ++bb) {
                        auto &bd2 = all_dims[bb];
                        PetscInt Nb = static_cast<PetscInt>(bd2.ni) * bd2.nj * bd2.nk;
                        // 更新当前块（或新建）
                        BlockInfo &target = (bb == b) ? cur : infos[bb];
                        if (bb != b) {
                            target.x.resize(Nb);
                            target.y.resize(Nb);
                            target.z.resize(Nb);
                        }
                        for (PetscInt i = 0; i < Nb; ++i) {
                            target.x[i] = p[offset + i];
                            target.y[i] = p[offset + Nb + i];
                            target.z[i] = p[offset + 2*Nb + i];
                        }
                        offset += 3 * Nb;
                    }
                    // 后面块已经读完，直接跳出
                    b = nblocks;
                } else {
                    // 尝试按独立 float 分量读（多块分别读）
                    // 直接按单块 3 分量独立处理
                    auto read_one_float = [&](std::vector<double> &dst) -> bool {
                        if (!read_fortran_record(rec)) return false;
                        float *p = reinterpret_cast<float*>(rec.data());
                        int32_t expected = static_cast<int32_t>(dst.size() * 4);
                        if (static_cast<int32_t>(rec.size()) != expected) {
                            // 实际读到的可能包含多个块
                            int32_t actual_n = static_cast<int32_t>(rec.size()) / 4;
                            for (PetscInt i = 0; i < std::min((PetscInt)actual_n, (PetscInt)dst.size()); ++i)
                                dst[i] = p[i];
                            // 剩余数据可能是后续块的
                            // 简化处理：仅取所需的 N 个值
                            return true;
                        }
                        for (PetscInt i = 0; i < (PetscInt)dst.size(); ++i)
                            dst[i] = p[i];
                        return true;
                    };
                    if (!read_one_float(cur.x) || !read_one_float(cur.y) || !read_one_float(cur.z))
                        throw std::runtime_error("Plot3D: coords fallback block " + std::to_string(b+1));
                }
            }
        } else {
            // 无 markers
            std::vector<float> buf(N);
            file.read(reinterpret_cast<char*>(buf.data()), N * 4);
            for (PetscInt i = 0; i < N; ++i) cur.x[i] = buf[i];
            file.read(reinterpret_cast<char*>(buf.data()), N * 4);
            for (PetscInt i = 0; i < N; ++i) cur.y[i] = buf[i];
            file.read(reinterpret_cast<char*>(buf.data()), N * 4);
            for (PetscInt i = 0; i < N; ++i) cur.z[i] = buf[i];
        }

        infos.push_back(std::move(cur));
    }
}
/// ====================================================================
// ParseCGNSFile - 读取 CGNS 结构网格
// ====================================================================
void MultiBlockMesh::ParseCGNSFile(const std::string &filename,
                                   std::vector<BlockInfo> &infos)
{
    int fn;
    if (cg_open(filename.c_str(), CG_MODE_READ, &fn) != CG_OK)
        throw std::runtime_error("CGNS: cannot open " + filename);

    // ---------- 读 base ----------
    int nbases;
    if (cg_nbases(fn, &nbases) != CG_OK)
        throw std::runtime_error("CGNS: no bases");
    if (nbases != 1)
        PetscPrintf(PETSC_COMM_WORLD, "CGNS WARNING: %d bases, using first\n", nbases);

    int base = 1;
    char basename[64];
    int celldim, physdim;
    cg_base_read(fn, base, basename, &celldim, &physdim);

    // ---------- 读 zone 数 ----------
    int nzones;
    cg_nzones(fn, base, &nzones);
    infos.clear();
    infos.reserve(nzones);

    for (int z = 1; z <= nzones; ++z) {
        ZoneType_t ztype;
        cg_zone_type(fn, base, z, &ztype);
        if (ztype != Structured) {
            PetscPrintf(PETSC_COMM_WORLD, "CGNS: skipping zone %d (not Structured)\n", z);
            continue;
        }

        cgsize_t size[9];
        char zonename[64];
        cg_zone_read(fn, base, z, zonename, size);

        cgsize_t ni = size[0];
        cgsize_t nj = size[1];
        cgsize_t nk = size[2];

        BlockInfo cur;
        cur.name = zonename;
        cur.ni = (PetscInt)ni;
        cur.nj = (PetscInt)nj;
        cur.nk = (PetscInt)nk;

        cgsize_t N = ni * nj * nk;
        cur.x.resize(N);
        cur.y.resize(N);
        cur.z.resize(N);

        // 读取坐标
        {
            int ncoords;
            cg_ncoords(fn, base, z, &ncoords);
            for (int c = 1; c <= ncoords; ++c) {
                char cname[64];
                DataType_t dt;
                cg_coord_info(fn, base, z, c, &dt, cname);
                std::string cn(cname);

                std::vector<double> buf(N);
                cgsize_t rmin[3] = {1, 1, 1};
                cgsize_t rmax[3] = {ni, nj, nk};
                cg_coord_read(fn, base, z, cname, RealDouble, rmin, rmax, buf.data());

                if (cn == "CoordinateX" || cn.find("CoordinateX") == 0) {
                    for (cgsize_t i = 0; i < N; ++i) cur.x[i] = buf[i];
                } else if (cn == "CoordinateY" || cn.find("CoordinateY") == 0) {
                    for (cgsize_t i = 0; i < N; ++i) cur.y[i] = buf[i];
                } else if (cn == "CoordinateZ" || cn.find("CoordinateZ") == 0) {
                    for (cgsize_t i = 0; i < N; ++i) cur.z[i] = buf[i];
                }
            }
        }

        infos.push_back(std::move(cur));
    }

    // ========== 第二遍：读取 BC 和 1-to-1 连接 ==========
    {
        std::map<std::string, int> name_to_idx;
        for (size_t i = 0; i < infos.size(); ++i)
            name_to_idx[infos[i].name] = (int)i;

        for (int z = 1; z <= nzones; ++z) {
            if (cg_goto(fn, base, "Zone_t", z, NULL) != CG_OK) continue;

            char zonename2[64];
            cgsize_t dummy[9];
            cg_zone_read(fn, base, z, zonename2, dummy);
            auto it = name_to_idx.find(zonename2);
            if (it == name_to_idx.end()) continue;
            int block_idx = it->second;

            // ----- 读 BC -----
            {
                int nbocos;
                if (cg_nbocos(fn, base, z, &nbocos) == CG_OK) {
                    for (int boco = 1; boco <= nbocos; ++boco) {
                        char boconame[64];
                        BCType_t bocotype;
                        PointSetType_t ptsettype;
                        cgsize_t npnts;
                        int normalindex;
                        cgsize_t normallistflag;
                        DataType_t normaldatatype;
                        int ndataset;
                        cg_boco_info(fn, base, z, boco, boconame,
                                     &bocotype, &ptsettype, &npnts,
                                     &normalindex, &normallistflag,
                                     &normaldatatype, &ndataset);
                        cgsize_t pointrange[6];
                        cg_boco_read(fn, base, z, boco, pointrange, NULL);

                        // ★ CGNS C API 返回: [imin, jmin, kmin, imax, jmax, kmax]
                        cgsize_t imin = pointrange[0], jmin = pointrange[1], kmin = pointrange[2];
                        cgsize_t imax = pointrange[3], jmax = pointrange[4], kmax = pointrange[5];

                        int face = -1;
                        if (imin == imax)
                            face = (imin == 1) ? 0 : 1;
                        else if (jmin == jmax)
                            face = (jmin == 1) ? 2 : 3;
                        else if (kmin == kmax)
                            face = (kmin == 1) ? 4 : 5;
                        if (face >= 0) {
                            FaceBC fbc;
                            fbc.block = block_idx;
                            fbc.face  = face;
                            fbc.bc_type = cg_BCTypeName(bocotype);
                            fbc.family  = "";
                            char famname[64];
                            if (cg_famname_read(famname) == CG_OK)
                                fbc.family = famname;
                            face_bcs_.push_back(fbc);
                        }
                    }
                }
            }

            // ----- 读 1-to-1 连接 -----
            {
                int n1to1;
                if (cg_n1to1(fn, base, z, &n1to1) == CG_OK) {
                    for (int c = 1; c <= n1to1; ++c) {
                        char connectname[64], donorname[64];
                        cgsize_t pointrange[6], donorrange[6];
                        int transform[3] = {0};
                        cg_1to1_read(fn, base, z, c, connectname,
                                     donorname, pointrange, donorrange, transform);

                        auto dit = name_to_idx.find(donorname);
                        if (dit == name_to_idx.end()) continue;
                        int donor_block = dit->second;

                        // ========================================================
                        // CGNS C API 返回格式: [imin, jmin, kmin, imax, jmax, kmax]
                        // （Fortran 二维数组 (3,2) 被 flatten 成 C 一维数组）
                        // ========================================================
                        auto detectFace = [](cgsize_t r[6],
                                             cgsize_t ni, cgsize_t nj, cgsize_t nk,
                                             int& face, cgsize_t out6[6]) -> bool
                        {
                            // 格式A（CGNS C API）: [imin, jmin, kmin, imax, jmax, kmax]
                            bool cgns_ok = (r[0]>=1 && r[0]<=ni && r[1]>=1 && r[1]<=nj &&
                                            r[2]>=1 && r[2]<=nk && r[3]>=1 && r[3]<=ni &&
                                            r[4]>=1 && r[4]<=nj && r[5]>=1 && r[5]<=nk);
                            // 格式B（SIDS 文档序）: [imin, imax, jmin, jmax, kmin, kmax]
                            bool sids_ok = (r[0]>=1 && r[0]<=ni && r[1]>=1 && r[1]<=ni &&
                                            r[2]>=1 && r[2]<=nj && r[3]>=1 && r[3]<=nj &&
                                            r[4]>=1 && r[4]<=nk && r[5]>=1 && r[5]<=nk);

                            // ★ 优先 CGNS C API 格式（这是 cg_1to1_read 实际返回的）
                            cgsize_t imin, imax, jmin, jmax, kmin, kmax;
                            if (cgns_ok) {
                                imin=r[0]; imax=r[3]; jmin=r[1]; jmax=r[4]; kmin=r[2]; kmax=r[5];
                            } else if (sids_ok) {
                                imin=r[0]; imax=r[1]; jmin=r[2]; jmax=r[3]; kmin=r[4]; kmax=r[5];
                            } else {
                                return false;
                            }

                            if (imin>imax) { auto t=imin; imin=imax; imax=t; }
                            if (jmin>jmax) { auto t=jmin; jmin=jmax; jmax=t; }
                            if (kmin>kmax) { auto t=kmin; kmin=kmax; kmax=t; }

                            out6[0]=imin-1; out6[1]=imax-1;
                            out6[2]=jmin-1; out6[3]=jmax-1;
                            out6[4]=kmin-1; out6[5]=kmax-1;

                            if      (imin == imax) face = (imin == 1) ? 0 : 1;
                            else if (jmin == jmax) face = (jmin == 1) ? 2 : 3;
                            else if (kmin == kmax) face = (kmin == 1) ? 4 : 5;
                            else                  face = -1;
                            return true;
                        };

                        int face_a = -1, face_b = -1;
                        cgsize_t normPR[6], normDR[6];
                        cgsize_t cur_ni = (cgsize_t)infos[block_idx].ni;
                        cgsize_t cur_nj = (cgsize_t)infos[block_idx].nj;
                        cgsize_t cur_nk = (cgsize_t)infos[block_idx].nk;

                        bool ok_a = detectFace(pointrange,
                                               cur_ni, cur_nj, cur_nk,
                                               face_a, normPR);
                        bool ok_b = detectFace(donorrange,
                                               (cgsize_t)infos[donor_block].ni,
                                               (cgsize_t)infos[donor_block].nj,
                                               (cgsize_t)infos[donor_block].nk,
                                               face_b, normDR);

                        if (ok_a && ok_b && face_a >= 0 && face_b >= 0) {
                            FaceConnectivity fc;
                            fc.block_a = block_idx;
                            fc.face_a  = face_a;
                            fc.block_b = donor_block;
                            fc.face_b  = face_b;
                            fc.transform[0] = transform[0];
                            fc.transform[1] = transform[1];
                            fc.transform[2] = transform[2];
                            for (int i = 0; i < 6; ++i) {
                                fc.pointrange[i] = normPR[i];
                                fc.donorrange[i] = normDR[i];
                            }
                            face_connections_.push_back(fc);
                        }
                    }
                }
            }
        }
    }

    cg_close(fn);
}


// 构造函数：初始化成员变量
MultiBlockMesh::MultiBlockMesh(const std::vector<std::array<PetscInt,3>> &procs,
                               PetscInt lap, PetscInt scheme_vis)
    : block_procs_(procs), LAP(lap), scheme_vis(scheme_vis)
{}
// ====================================================================
// autoAllocateProcs — 两级自动进程分配
// ====================================================================
std::vector<std::array<PetscInt,3>> MultiBlockMesh::autoAllocateProcs(
    const std::vector<BlockInfo>& infos, PetscInt P_total)
{
    PetscInt nBlocks = (PetscInt)infos.size();
    std::vector<std::array<PetscInt,3>> result(nBlocks, {1,1,1});
    if (nBlocks == 0 || P_total < nBlocks) return result;

    // ── 第1级：按点数比例分配 ──
    std::vector<PetscReal> pts(nBlocks);
    PetscReal total_pts = 0.0;
    for (PetscInt b = 0; b < nBlocks; ++b) {
        pts[b] = (PetscReal)(infos[b].ni * infos[b].nj * infos[b].nk);
        total_pts += pts[b];
    }

    std::vector<PetscInt> alloc(nBlocks, 1);
    PetscInt remaining = P_total - nBlocks;
    if (remaining > 0) {
        struct Frac { PetscInt idx; PetscReal frac; };
        std::vector<Frac> fracs(nBlocks);
        for (PetscInt b = 0; b < nBlocks; ++b) {
            PetscReal raw = pts[b] / total_pts * (PetscReal)remaining;
            PetscInt fv = (PetscInt)raw;
            alloc[b] += fv;
            fracs[b] = {b, raw - (PetscReal)fv};
        }
        PetscInt used = 0;
        for (PetscInt b = 0; b < nBlocks; ++b) used += (alloc[b] - 1);
        PetscInt leftover = remaining - used;
        std::sort(fracs.begin(), fracs.end(),
            [](const Frac& a, const Frac& b) { return a.frac > b.frac; });
        for (PetscInt i = 0; i < leftover && i < nBlocks; ++i)
            alloc[fracs[i].idx]++;
    }

    // ── 第2级：块内最优分解 ──
    for (PetscInt b = 0; b < nBlocks; ++b) {
        PetscInt P = alloc[b], ni = infos[b].ni, nj = infos[b].nj, nk = infos[b].nk;
        PetscInt bp = 1, bq = 1, br = 1;
        PetscReal best = 1e30;
        for (PetscInt p = 1; p <= P; ++p) {
            if (P % p) continue;
            PetscInt r1 = P / p;
            for (PetscInt q = 1; q <= r1; ++q) {
                if (r1 % q) continue;
                PetscInt r = r1 / q;
                PetscReal sx = (PetscReal)ni/p, sy = (PetscReal)nj/q, sz = (PetscReal)nk/r;
                PetscReal smax = std::max({sx, sy, sz});
                PetscReal smin = std::min({sx, sy, sz});
                PetscReal score = (smin > 0) ? smax/smin : 1e30;
                int dirs = (p>1)+(q>1)+(r>1);
                if (dirs == 1 && P > 2) score *= 1.2;
                if (score < best) { best = score; bp = p; bq = q; br = r; }
            }
        }
        result[b] = {bp, bq, br};
    }
    return result;
}


// 析构函数：释放所有子通信域
MultiBlockMesh::~MultiBlockMesh() {
    for (auto &comm : block_comms_) {
        if (comm != MPI_COMM_NULL && comm != PETSC_COMM_WORLD) {
            MPI_Comm_free(&comm);
        }
    }
}
// Initialize 核心逻辑
void MultiBlockMesh::Initialize(const std::string &tecplot_filename,
                                PetscInt grid_format)
{
    PetscMPIInt rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    // ---------- 1. 解析网格文件（仅 rank 0）----------
    std::vector<BlockInfo> infos;
    if (rank == 0) {
        try {
            if (grid_format == 0)
                ParsePlot3DFile(tecplot_filename, infos);
            else if (grid_format == 2)
                ParseCGNSFile(tecplot_filename, infos);
            else
                ParseTecplotFile(tecplot_filename, infos);
        } catch (const std::exception &e) {
            PetscPrintf(PETSC_COMM_WORLD, "Error parsing file: %s\n", e.what());
            MPI_Abort(PETSC_COMM_WORLD, 1);
        }
    }

    // 广播块数
    PetscInt num_blocks = static_cast<PetscInt>(infos.size());
    MPI_Bcast(&num_blocks, 1, MPIU_INT, 0, PETSC_COMM_WORLD);

    // 自动分配：如果 block_procs_ 为空，则根据网格尺寸自动计算进程分布
    if (block_procs_.empty()) {
        std::vector<PetscInt> bdims(num_blocks * 3);
        if (rank == 0) {
            for (PetscInt i = 0; i < num_blocks; ++i) {
                bdims[3*i]   = infos[i].ni;
                bdims[3*i+1] = infos[i].nj;
                bdims[3*i+2] = infos[i].nk;
            }
        }
        MPI_Bcast(bdims.data(), 3 * num_blocks, MPIU_INT, 0, PETSC_COMM_WORLD);
        
        std::vector<BlockInfo> tmp(num_blocks);
        for (PetscInt i = 0; i < num_blocks; ++i) {
            tmp[i].ni = bdims[3*i]; 
            tmp[i].nj = bdims[3*i+1]; 
            tmp[i].nk = bdims[3*i+2];
        }
        
        block_procs_ = autoAllocateProcs(tmp, (PetscInt)size);
        
        if (rank == 0) {
            PetscPrintf(PETSC_COMM_WORLD, "Auto procs:\n");
            for (PetscInt b = 0; b < num_blocks; ++b)
                PetscPrintf(PETSC_COMM_WORLD, "  Block %d [%d x %d x %d]: %d x %d x %d\n",
                    (int)b, (int)tmp[b].ni, (int)tmp[b].nj, (int)tmp[b].nk,
                    (int)block_procs_[b][0], (int)block_procs_[b][1], (int)block_procs_[b][2]);
        }
    }

    if (static_cast<PetscInt>(block_procs_.size()) != num_blocks) {
        throw std::runtime_error("Block process layout size does not match number of blocks in file");
    }

    // ---------- 2. 校验总进程需求 ----------
    std::vector<PetscInt> block_nprocs(num_blocks);
    PetscInt total_needed = 0;
    for (PetscInt b = 0; b < num_blocks; ++b) {
        block_nprocs[b] = block_procs_[b][0] * block_procs_[b][1] * block_procs_[b][2];
        total_needed += block_nprocs[b];
    }
    if (total_needed != size) {
        throw std::runtime_error("Total required processes (" + std::to_string(total_needed) +
                                 ") != available (" + std::to_string(size) + ")");
    }

    // ---------- 3. 确定本进程所属块 ----------
    PetscInt my_block = -1;
    PetscInt offset = 0;
    std::vector<PetscInt> block_start_rank(num_blocks);
    for (PetscInt b = 0; b < num_blocks; ++b) {
        block_start_rank[b] = offset;
        if (rank >= offset && rank < offset + block_nprocs[b]) {
            my_block = b;
        }
        offset += block_nprocs[b];
    }

    // ---------- 4. 创建子通信域 ----------
    block_comms_.resize(num_blocks, MPI_COMM_NULL);
    MPI_Comm split_comm;
    // 如果 my_block 为 -1 (理论上不会发生，因为 total_needed == size)，需要保护
    int color = (my_block >= 0) ? my_block : MPI_UNDEFINED;
    int key   = (my_block >= 0) ? (rank - block_start_rank[my_block]) : 0;
    MPI_Comm_split(PETSC_COMM_WORLD, color, key, &split_comm);
    if (my_block >= 0) {
        block_comms_[my_block] = split_comm;
    } else {
        // 不应该到达这里，但为了安全释放无效 communicator
        if (split_comm != MPI_COMM_NULL) MPI_Comm_free(&split_comm);
    }

    // ---------- 5. 广播尺寸 ----------
    std::vector<PetscInt> dims(num_blocks * 3);
    if (rank == 0) {
        for (PetscInt i = 0; i < num_blocks; i++) {
            dims[3*i]   = infos[i].ni;
            dims[3*i+1] = infos[i].nj;
            dims[3*i+2] = infos[i].nk;
        }
    }
    MPI_Bcast(dims.data(), 3 * num_blocks, MPIU_INT, 0, PETSC_COMM_WORLD);

    // 广播名称
    block_names_.resize(num_blocks);
    if (rank == 0) {
        for (PetscInt i = 0; i < num_blocks; i++)
            block_names_[i] = infos[i].name;
    }
    for (PetscInt i = 0; i < num_blocks; i++) {
        int len = (rank == 0) ? static_cast<int>(block_names_[i].size()) : 0;
        MPI_Bcast(&len, 1, MPI_INT, 0, PETSC_COMM_WORLD);
        if (rank != 0) block_names_[i].resize(len);
        if (len > 0)
            MPI_Bcast(&block_names_[i][0], len, MPI_CHAR, 0, PETSC_COMM_WORLD);
    }

    // ---------- 广播 face_bcs_ ----------
    {
    int nbc = (rank == 0) ? static_cast<int>(face_bcs_.size()) : 0;
    MPI_Bcast(&nbc, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    if (rank != 0) face_bcs_.resize(nbc);
    for (int i = 0; i < nbc; ++i) {
        int blk = (rank == 0) ? face_bcs_[i].block : 0;
        int fc  = (rank == 0) ? face_bcs_[i].face  : 0;
        MPI_Bcast(&blk, 1, MPI_INT, 0, PETSC_COMM_WORLD);
        MPI_Bcast(&fc,  1, MPI_INT, 0, PETSC_COMM_WORLD);
        if (rank != 0) { face_bcs_[i].block = blk; face_bcs_[i].face = fc; }

        int type_len = (rank == 0) ? static_cast<int>(face_bcs_[i].bc_type.size()) : 0;
        MPI_Bcast(&type_len, 1, MPI_INT, 0, PETSC_COMM_WORLD);
        if (rank != 0) face_bcs_[i].bc_type.resize(type_len);
        if (type_len > 0) MPI_Bcast(&face_bcs_[i].bc_type[0], type_len, MPI_CHAR, 0, PETSC_COMM_WORLD);

        int fam_len = (rank == 0) ? static_cast<int>(face_bcs_[i].family.size()) : 0;
        MPI_Bcast(&fam_len, 1, MPI_INT, 0, PETSC_COMM_WORLD);
        if (rank != 0) face_bcs_[i].family.resize(fam_len);
        if (fam_len > 0) MPI_Bcast(&face_bcs_[i].family[0], fam_len, MPI_CHAR, 0, PETSC_COMM_WORLD);
    }
    }

    // ---------- 广播 face_connections_（含 pointrange / donorrange）----------
    {
    int nc = (rank == 0) ? static_cast<int>(face_connections_.size()) : 0;
    MPI_Bcast(&nc, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    if (rank != 0) face_connections_.resize(nc);
    for (int i = 0; i < nc; ++i) {
        int buf[19];
        if (rank == 0) {
            buf[0]  = face_connections_[i].block_a;
            buf[1]  = face_connections_[i].face_a;
            buf[2]  = face_connections_[i].block_b;
            buf[3]  = face_connections_[i].face_b;
            buf[4]  = face_connections_[i].transform[0];
            buf[5]  = face_connections_[i].transform[1];
            buf[6]  = face_connections_[i].transform[2];
            buf[7]  = (int)face_connections_[i].pointrange[0];
            buf[8]  = (int)face_connections_[i].pointrange[1];
            buf[9]  = (int)face_connections_[i].pointrange[2];
            buf[10] = (int)face_connections_[i].pointrange[3];
            buf[11] = (int)face_connections_[i].pointrange[4];
            buf[12] = (int)face_connections_[i].pointrange[5];
            buf[13] = (int)face_connections_[i].donorrange[0];
            buf[14] = (int)face_connections_[i].donorrange[1];
            buf[15] = (int)face_connections_[i].donorrange[2];
            buf[16] = (int)face_connections_[i].donorrange[3];
            buf[17] = (int)face_connections_[i].donorrange[4];
            buf[18] = (int)face_connections_[i].donorrange[5];
        }
        MPI_Bcast(buf, 19, MPI_INT, 0, PETSC_COMM_WORLD);
        if (rank != 0) {
            face_connections_[i].block_a    = buf[0];
            face_connections_[i].face_a     = buf[1];
            face_connections_[i].block_b    = buf[2];
            face_connections_[i].face_b     = buf[3];
            face_connections_[i].transform[0] = buf[4];
            face_connections_[i].transform[1] = buf[5];
            face_connections_[i].transform[2] = buf[6];
            face_connections_[i].pointrange[0] = buf[7];
            face_connections_[i].pointrange[1] = buf[8];
            face_connections_[i].pointrange[2] = buf[9];
            face_connections_[i].pointrange[3] = buf[10];
            face_connections_[i].pointrange[4] = buf[11];
            face_connections_[i].pointrange[5] = buf[12];
            face_connections_[i].donorrange[0] = buf[13];
            face_connections_[i].donorrange[1] = buf[14];
            face_connections_[i].donorrange[2] = buf[15];
            face_connections_[i].donorrange[3] = buf[16];
            face_connections_[i].donorrange[4] = buf[17];
            face_connections_[i].donorrange[5] = buf[18];
        }
    }
    }

    // ---------- 6. 发送坐标 ----------
    const int tag = 12345;
    if (rank == 0) {
        for (PetscInt b = 0; b < num_blocks; ++b) {
            PetscInt npts = infos[b].ni * infos[b].nj * infos[b].nk;
            if (block_start_rank[b] == 0) continue;
            MPI_Send(infos[b].x.data(), npts, MPI_DOUBLE, block_start_rank[b], tag,   PETSC_COMM_WORLD);
            MPI_Send(infos[b].y.data(), npts, MPI_DOUBLE, block_start_rank[b], tag+1, PETSC_COMM_WORLD);
            MPI_Send(infos[b].z.data(), npts, MPI_DOUBLE, block_start_rank[b], tag+2, PETSC_COMM_WORLD);
        }
    }

    // ---------- 7. 创建 Mesh ----------
    blocks_.clear();
    blocks_.reserve(num_blocks);
    full_coords_.resize(num_blocks);          // ★ 方案B: 预分配

    for (PetscInt b = 0; b < num_blocks; ++b) {
        PetscInt ni = dims[3*b];
        PetscInt nj = dims[3*b+1];
        PetscInt nk = dims[3*b+2];
        PetscInt npts = ni * nj * nk;

        if (b == my_block) {
            std::vector<PetscReal> x(npts), y(npts), z(npts);

            if (rank == block_start_rank[b]) {
                if (rank == 0) {
                    x = infos[b].x;
                    y = infos[b].y;
                    z = infos[b].z;
                } else {
                    MPI_Recv(x.data(), npts, MPI_DOUBLE, 0, tag,   PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(y.data(), npts, MPI_DOUBLE, 0, tag+1, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(z.data(), npts, MPI_DOUBLE, 0, tag+2, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            MPI_Bcast(x.data(), npts, MPI_DOUBLE, 0, block_comms_[b]);
            MPI_Bcast(y.data(), npts, MPI_DOUBLE, 0, block_comms_[b]);
            MPI_Bcast(z.data(), npts, MPI_DOUBLE, 0, block_comms_[b]);

            PetscInt my_rank_block;
            MPI_Comm_rank(block_comms_[b], &my_rank_block);

            auto mesh_ptr = std::make_unique<Mesh>(
                ni, nj, nk,
                my_rank_block,
                block_procs_[b][0], block_procs_[b][1], block_procs_[b][2],
                LAP, GRID3D, scheme_vis,
                block_comms_[b]
            );
            mesh_ptr->InitializeFromCoordinates(x, y, z);
            blocks_.push_back(std::move(mesh_ptr));

            // ★ 方案B: 保存完整坐标（供 slab 打包使用）
            full_coords_[b].valid = true;
            full_coords_[b].ni = ni;
            full_coords_[b].nj = nj;
            full_coords_[b].nk = nk;
            full_coords_[b].x = x;
            full_coords_[b].y = y;
            full_coords_[b].z = z;
        } else {
            blocks_.push_back(nullptr);
        }
    }
    // ---------- 8. ★ 方案B: 交换块间连接面 slab ----------
    donor_slabs_.resize(num_blocks);
    exchangeDonorSlabs(block_start_rank);
    MPI_Barrier(PETSC_COMM_WORLD);
}
// 导出到 Tecplot(dat格式)
PetscErrorCode Mesh::ExportToTecplot(const std::string &filename)
{
    PetscErrorCode ierr;
    PetscMPIInt rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    DMDALocalInfo info;
    PetscReal ***axx, ***ayy, ***azz;
    PetscReal ***akx, ***aky, ***akz;
    PetscReal ***aix, ***aiy, ***aiz;
    PetscReal ***asx, ***asy, ***asz;
    PetscReal ***ajac;
    PetscReal ***akx1, ***aky1, ***akz1;
    PetscReal ***aix1, ***aiy1, ***aiz1;
    PetscReal ***asx1, ***asy1, ***asz1;

    ierr = GetLocalArrays(axx, ayy, azz, akx, aky, akz,
                          aix, aiy, aiz, asx, asy, asz, ajac,
                          akx1, aky1, akz1, aix1, aiy1, aiz1,
                          asx1, asy1, asz1, info);
    CHKERRQ(ierr);

    PetscInt xs = info.xs, ys = info.ys, zs = info.zs;
    PetscInt xm = info.xm, ym = info.ym, zm = info.zm;

    // 全局尺寸
    PetscInt nx = nx_global, ny = ny_global, nz = nz_global;
    PetscInt global_size = nx * ny * nz;   // 总网格点数

    // 每个进程分配一个全局大小的数组（存放 6 个变量），初始为 0.0
    std::vector<PetscReal> X_global(global_size, 0.0);
    std::vector<PetscReal> Y_global(global_size, 0.0);
    std::vector<PetscReal> Z_global(global_size, 0.0);
    std::vector<PetscReal> Akx_global(global_size, 0.0);
    std::vector<PetscReal> Akx1_global(global_size, 0.0);
    std::vector<PetscReal> Axx_global(global_size, 0.0);

    // 填充本进程拥有的点
    // Tecplot 顺序：I 变化最快，J 次之，K 最慢，
    // 线性索引：i + j*nx + k*nx*ny
    for (PetscInt k = zs; k < zs + zm; k++) {
        for (PetscInt j = ys; j < ys + ym; j++) {
            for (PetscInt i = xs; i < xs + xm; i++) {
                PetscInt idx = i + j * nx + k * nx * ny;   // IJK 顺序
                X_global[idx]    = axx[k][j][i];
                Y_global[idx]    = ayy[k][j][i];
                Z_global[idx]    = azz[k][j][i];
                Akx_global[idx]  = akx[k][j][i];
                Akx1_global[idx] = akx1[k][j][i];
                Axx_global[idx]  = axx[k][j][i];
            }
        }
    }

    // 汇总到 rank 0（求和，因为每个点只有一个进程有非零值）
    std::vector<PetscReal> X_recv, Y_recv, Z_recv, Akx_recv, Akx1_recv, Axx_recv;
    if (rank == 0) {
        X_recv.resize(global_size);
        Y_recv.resize(global_size);
        Z_recv.resize(global_size);
        Akx_recv.resize(global_size);
        Akx1_recv.resize(global_size);
        Axx_recv.resize(global_size);
    }

    MPI_Reduce(X_global.data(), X_recv.data(), global_size, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(Y_global.data(), Y_recv.data(), global_size, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(Z_global.data(), Z_recv.data(), global_size, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(Akx_global.data(), Akx_recv.data(), global_size, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(Akx1_global.data(), Akx1_recv.data(), global_size, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(Axx_global.data(), Axx_recv.data(), global_size, MPI_DOUBLE, MPI_SUM, 0, comm);

    // rank 0 按 Tecplot 顺序写出
    if (rank == 0) {
        std::string outname = filename + ".dat";
        std::ofstream outfile(outname);
        if (!outfile.is_open()) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN, "Cannot open output file");
        }

        outfile << "TITLE = \"Mesh and Jacobian Data\"" << std::endl;
        outfile << "VARIABLES = \"X\", \"Y\", \"Z\", \"Akx\", \"Akx1\", \"Axx\"" << std::endl;
        outfile << "ZONE T=\"Full Grid\", I=" << nx
                << ", J=" << ny << ", K=" << nz << ", F=POINT" << std::endl;

        outfile.precision(8);
        outfile << std::scientific;
        for (PetscInt idx = 0; idx < global_size; idx++) {
            outfile << X_recv[idx]    << " "
                    << Y_recv[idx]    << " "
                    << Z_recv[idx]    << " "
                    << Akx_recv[idx]  << " "
                    << Akx1_recv[idx] << " "
                    << Axx_recv[idx]  << std::endl;
        }
        outfile.close();
        PetscPrintf(comm, "Data exported to %s\n", outname.c_str());
    }

    ierr = RestoreLocalArrays(axx, ayy, azz, akx, aky, akz,
                              aix, aiy, aiz, asx, asy, asz, ajac,
                              akx1, aky1, akz1, aix1, aiy1, aiz1,
                              asx1, asy1, asz1);
    CHKERRQ(ierr);

    return 0;
}
//输出一个文件
void MultiBlockMesh::ExportAllToTecplot(const std::string &filename) {
    // 每个块输出独立的临时文件
    for (PetscInt b = 0; b < static_cast<PetscInt>(blocks_.size()); ++b) {
        if (blocks_[b]) {
            blocks_[b]->ExportToTecplot("_temp_block_" + std::to_string(b));
        }
    }

    MPI_Barrier(PETSC_COMM_WORLD); // 等待所有块写完

    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank == 0) {
        std::string outname = filename + ".dat";
        std::ofstream outfile(outname);
        outfile << "TITLE = \"Multi-Block Grid\"\n";
        outfile << "VARIABLES = \"X\", \"Y\", \"Z\", \"Akx\", \"Akx1\", \"Axx\"\n";

        for (PetscInt b = 0; b < static_cast<PetscInt>(blocks_.size()); ++b) {
            std::string tempname = "_temp_block_" + std::to_string(b) + ".dat";
            std::ifstream tempfile(tempname);
            if (!tempfile.is_open()) {
                PetscPrintf(PETSC_COMM_WORLD, "Warning: cannot open %s\n", tempname.c_str());
                continue;
            }
            std::string line;
            bool in_zone = false;
            while (std::getline(tempfile, line)) {
                if (line.find("ZONE") != std::string::npos) {
                    size_t pos = line.find("T=\"Full Grid\"");
                    if (pos != std::string::npos && b < static_cast<PetscInt>(block_names_.size())) {
                        // 只替换 "Full Grid" 这9个字符，保留前后的双引号
                        line.replace(pos + 3, 9, block_names_[b]);
                    }
                    outfile << line << "\n";
                    in_zone = true;
                } else if (in_zone) {
                    outfile << line << "\n";
                }
            }
            tempfile.close();
            ::remove(tempname.c_str());  // 删除临时文件
        }
        outfile.close();
        PetscPrintf(PETSC_COMM_WORLD, "Multi-block data merged into %s\n", outname.c_str());
    }
}
// 打印网格信息
void Mesh::printInfo() const
{
    PetscPrintf(comm,
                "Mesh Info:\n"
                "  Global size: %ld x %ld x %ld\n"
                "  Local size:  %ld x %ld x %ld\n"
                "  Grid type:   %d\n"
                "  My ID:       %d, Process division: (%d,%d,%d)\n",
                (long)nx_global, (long)ny_global, (long)nz_global,
                (long)nx, (long)ny, (long)nz,
                (int)Iflag_Gridtype,
                (int)my_id, (int)npx0, (int)npy0, (int)npz0);
}
//查询连接信息
std::pair<int,int> MultiBlockMesh::findConnectedFace(int block, int face) const
{
    for (const auto &fc : face_connections_) {
        if (fc.block_a == block && fc.face_a == face)
            return {fc.block_b, fc.face_b};
        if (fc.block_b == block && fc.face_b == face)
            return {fc.block_a, fc.face_a};
    }
    return {-1, -1};
}
// ====================================================================
// printTopology - 打印拓扑信息
// ====================================================================
void MultiBlockMesh::printTopology() const
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    if (rank != 0) return;

    const char* face_names[6] = {"LEFT", "RIGHT", "BOTTOM", "TOP", "BACK", "FRONT"};

    PetscPrintf(PETSC_COMM_WORLD, "\n========== CGNS Topology ==========\n");

    // ---- 块列表 ----
    PetscPrintf(PETSC_COMM_WORLD, "Blocks: %d\n", (int)blocks_.size());
    for (size_t b = 0; b < blocks_.size(); ++b) {
        if (blocks_[b]) {
            PetscPrintf(PETSC_COMM_WORLD, "  Block %d: \"%s\"  [%d x %d x %d]\n",
                        (int)b, block_names_[b].c_str(),
                        (int)blocks_[b]->getNxGlobal(),
                        (int)blocks_[b]->getNyGlobal(),
                        (int)blocks_[b]->getNzGlobal());
        }
    }

    // ---- 每个块的面状态汇总 ----
    PetscPrintf(PETSC_COMM_WORLD, "\n--- Per-Block Face Summary ---\n");
    for (size_t b = 0; b < blocks_.size(); ++b) {
        PetscPrintf(PETSC_COMM_WORLD, "  Block %d \"%s\":\n", (int)b, block_names_[b].c_str());
        for (int f = 0; f < 6; ++f) {
            auto [db, df] = findConnectedFace((int)b, f);
            if (db >= 0) {
                PetscPrintf(PETSC_COMM_WORLD, "    %-7s ->  Block %d [%s]\n",
                            face_names[f], db, face_names[df]);
            } else {
                std::string bc_str = "(none)";
                for (const auto &bc : face_bcs_) {
                    if (bc.block == (int)b && bc.face == f) {
                        bc_str = bc.bc_type;
                        break;
                    }
                }
                PetscPrintf(PETSC_COMM_WORLD, "    %-7s ->  %s\n", face_names[f], bc_str.c_str());
            }
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, "====================================\n\n");
}
// ====================================================================
// exchangeDonorSlabs — 为每个连接面交换 LAP 层 slab
// ====================================================================
void MultiBlockMesh::exchangeDonorSlabs(const std::vector<PetscInt>& block_start_rank)
{
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

    PetscInt num_blocks = (PetscInt)full_coords_.size();

    PetscInt my_block = -1;
    for (PetscInt b = 0; b < num_blocks; ++b) {
        if (full_coords_[b].valid) { my_block = b; break; }
    }

    // ============================
    // ★ packSlab: 法向从 interior layer 1 开始打包
    // ============================
    auto packSlab = [&](PetscInt donor_block, PetscInt donor_face,
                        const cgsize_t* donorrange) -> DonorSlab
    {
        DonorSlab slab;
        const auto& coords = full_coords_[donor_block];
        if (!coords.valid) return slab;

        PetscInt NI = coords.ni, NJ = coords.nj, NK = coords.nk;

        PetscInt dr_imin = (PetscInt)donorrange[0];
        PetscInt dr_imax = (PetscInt)donorrange[1];
        PetscInt dr_jmin = (PetscInt)donorrange[2];
        PetscInt dr_jmax = (PetscInt)donorrange[3];
        PetscInt dr_kmin = (PetscInt)donorrange[4];
        PetscInt dr_kmax = (PetscInt)donorrange[5];

        if (dr_imin < 0)  dr_imin = 0;
        if (dr_jmin < 0)  dr_jmin = 0;
        if (dr_kmin < 0)  dr_kmin = 0;
        if (dr_imax >= NI) dr_imax = NI - 1;
        if (dr_jmax >= NJ) dr_jmax = NJ - 1;
        if (dr_kmax >= NK) dr_kmax = NK - 1;

        if (dr_imin > dr_imax || dr_jmin > dr_jmax || dr_kmin > dr_kmax)
            return slab;

        // 法向深度（去掉 boundary，只装 interior layers）
        PetscInt nd_i = (LAP < NI-1) ? LAP : NI-1;
        PetscInt nd_j = (LAP < NJ-1) ? LAP : NJ-1;
        PetscInt nd_k = (LAP < NK-1) ? LAP : NK-1;

        switch (donor_face) {
        case 0: // i-min: 法向=i, 从 interior layer 1 开始
            slab.i0 = 1;              slab.j0 = dr_jmin;  slab.k0 = dr_kmin;
            slab.ni = LAP;            slab.nj = dr_jmax - dr_jmin + 1;
                                      slab.nk = dr_kmax - dr_kmin + 1;
            break;
        case 1: // i-max: 法向=i, 从 NI-LAP-1 开始
            slab.i0 = NI - LAP - 1;   slab.j0 = dr_jmin;  slab.k0 = dr_kmin;
            slab.ni = LAP;            slab.nj = dr_jmax - dr_jmin + 1;
                                      slab.nk = dr_kmax - dr_kmin + 1;
            break;
        case 2: // j-min: 法向=j, 从 interior layer 1 开始
            slab.i0 = dr_imin;        slab.j0 = 1;        slab.k0 = dr_kmin;
            slab.ni = dr_imax - dr_imin + 1;
                                      slab.nj = LAP;
                                      slab.nk = dr_kmax - dr_kmin + 1;
            break;
        case 3: // j-max: 法向=j, 从 NJ-LAP-1 开始
            slab.i0 = dr_imin;        slab.j0 = NJ - LAP - 1;
                                      slab.k0 = dr_kmin;
            slab.ni = dr_imax - dr_imin + 1;
                                      slab.nj = LAP;
                                      slab.nk = dr_kmax - dr_kmin + 1;
            break;
        case 4: // k-min: 法向=k, 从 interior layer 1 开始
            slab.i0 = dr_imin;        slab.j0 = dr_jmin;  slab.k0 = 1;
            slab.ni = dr_imax - dr_imin + 1;
                                      slab.nj = dr_jmax - dr_jmin + 1;
                                      slab.nk = LAP;
            break;
        case 5: // k-max: 法向=k, 从 NK-LAP-1 开始
            slab.i0 = dr_imin;        slab.j0 = dr_jmin;
            slab.k0 = NK - LAP - 1;
            slab.ni = dr_imax - dr_imin + 1;
                                      slab.nj = dr_jmax - dr_jmin + 1;
                                      slab.nk = LAP;
            break;
        default:
            return slab;
        }

        PetscInt total = slab.ni * slab.nj * slab.nk;
        slab.x.resize((size_t)total);
        slab.y.resize((size_t)total);
        slab.z.resize((size_t)total);

        PetscInt idx = 0;
        for (PetscInt k = 0; k < slab.nk; ++k) {
            PetscInt kg = slab.k0 + k;
            for (PetscInt j = 0; j < slab.nj; ++j) {
                PetscInt jg = slab.j0 + j;
                for (PetscInt i = 0; i < slab.ni; ++i) {
                    PetscInt ig = slab.i0 + i;
                    if (ig < 0 || ig >= NI || jg < 0 || jg >= NJ || kg < 0 || kg >= NK) {
                        slab.x[idx] = slab.y[idx] = slab.z[idx] = 0.0;
                    } else {
                        PetscInt gi = (kg * NJ + jg) * NI + ig;
                        slab.x[idx] = coords.x[gi];
                        slab.y[idx] = coords.y[gi];
                        slab.z[idx] = coords.z[gi];
                    }
                    ++idx;
                }
            }
        }
        slab.valid = true;
        return slab;
    };

    // ======== 世界广播 ========
    PetscInt nConn = (PetscInt)face_connections_.size();
    for (PetscInt ci = 0; ci < nConn; ++ci) {
        auto& conn = face_connections_[ci];

        // ── 方向1: block_a ← block_b ──
        {
            PetscMPIInt root = (PetscMPIInt)block_start_rank[conn.block_b];
            PetscMPIInt dims[6];
            PetscMPIInt total = 0;
            std::vector<PetscReal> xb, yb, zb;

            if (rank == root) {
                DonorSlab s = packSlab((PetscInt)conn.block_b,
                                       (PetscInt)conn.face_b,
                                       conn.donorrange);
                if (s.valid) {
                    dims[0] = (PetscMPIInt)s.ni; dims[1] = (PetscMPIInt)s.nj;
                    dims[2] = (PetscMPIInt)s.nk; dims[3] = (PetscMPIInt)s.i0;
                    dims[4] = (PetscMPIInt)s.j0; dims[5] = (PetscMPIInt)s.k0;
                    total   = (PetscMPIInt)(s.ni * s.nj * s.nk);
                    xb = std::move(s.x);
                    yb = std::move(s.y);
                    zb = std::move(s.z);
                }
            }
            MPI_Bcast(dims,  6, MPI_INT,    root, PETSC_COMM_WORLD);
            MPI_Bcast(&total, 1, MPI_INT,    root, PETSC_COMM_WORLD);

            // ★★★ 广播数据（之前缺失） ★★★
            if (total > 0) {
                if (rank != root) {
                    xb.resize((size_t)total);
                    yb.resize((size_t)total);
                    zb.resize((size_t)total);
                }
                MPI_Bcast(xb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                MPI_Bcast(yb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                MPI_Bcast(zb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
            }

            PetscMPIInt recv_root = (PetscMPIInt)block_start_rank[conn.block_a];
            if (rank == recv_root) {
                DonorSlab s;
                s.ni = (PetscInt)dims[0]; s.nj = (PetscInt)dims[1];
                s.nk = (PetscInt)dims[2]; s.i0 = (PetscInt)dims[3];
                s.j0 = (PetscInt)dims[4]; s.k0 = (PetscInt)dims[5];
                s.valid = true;
                if (total > 0) {
                    s.x = std::move(xb);
                    s.y = std::move(yb);
                    s.z = std::move(zb);
                }
                donor_slabs_[conn.block_a][conn.face_a] = std::move(s);
            }
        }

        // ── 方向2: block_b ← block_a ──
        {
            PetscMPIInt root = (PetscMPIInt)block_start_rank[conn.block_a];
            PetscMPIInt dims[6];
            PetscMPIInt total = 0;
            std::vector<PetscReal> xb, yb, zb;

            if (rank == root) {
                DonorSlab s = packSlab((PetscInt)conn.block_a,
                                       (PetscInt)conn.face_a,
                                       conn.pointrange);
                if (s.valid) {
                    dims[0] = (PetscMPIInt)s.ni; dims[1] = (PetscMPIInt)s.nj;
                    dims[2] = (PetscMPIInt)s.nk; dims[3] = (PetscMPIInt)s.i0;
                    dims[4] = (PetscMPIInt)s.j0; dims[5] = (PetscMPIInt)s.k0;
                    total   = (PetscMPIInt)(s.ni * s.nj * s.nk);
                    xb = std::move(s.x);
                    yb = std::move(s.y);
                    zb = std::move(s.z);
                }
            }
            MPI_Bcast(dims,  6, MPI_INT,    root, PETSC_COMM_WORLD);
            MPI_Bcast(&total, 1, MPI_INT,    root, PETSC_COMM_WORLD);

            // ★★★ 广播数据（之前缺失） ★★★
            if (total > 0) {
                if (rank != root) {
                    xb.resize((size_t)total);
                    yb.resize((size_t)total);
                    zb.resize((size_t)total);
                }
                MPI_Bcast(xb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                MPI_Bcast(yb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
                MPI_Bcast(zb.data(), total, MPI_DOUBLE, root, PETSC_COMM_WORLD);
            }

            PetscMPIInt recv_root = (PetscMPIInt)block_start_rank[conn.block_b];
            if (rank == recv_root) {
                DonorSlab s;
                s.ni = (PetscInt)dims[0]; s.nj = (PetscInt)dims[1];
                s.nk = (PetscInt)dims[2]; s.i0 = (PetscInt)dims[3];
                s.j0 = (PetscInt)dims[4]; s.k0 = (PetscInt)dims[5];
                s.valid = true;
                if (total > 0) {
                    s.x = std::move(xb);
                    s.y = std::move(yb);
                    s.z = std::move(zb);
                }
                donor_slabs_[conn.block_b][conn.face_b] = std::move(s);
            }
        }
    }

    // ======== 各块内广播 ========
    for (PetscInt b = 0; b < num_blocks; ++b) {
        if ((PetscInt)my_block != b) continue;
        MPI_Comm comm = block_comms_[b];
        for (PetscMPIInt face = 0; face < 6; ++face) {
            DonorSlab& slab = donor_slabs_[b][face];
            PetscMPIInt vf = slab.valid ? 1 : 0;
            MPI_Bcast(&vf, 1, MPI_INT, 0, comm);
            slab.valid = (vf != 0);
            if (!slab.valid) continue;

            PetscMPIInt dims[6];
            PetscMPIInt total;
            if (rank == (PetscMPIInt)block_start_rank[b]) {
                dims[0] = (PetscMPIInt)slab.ni; dims[1] = (PetscMPIInt)slab.nj;
                dims[2] = (PetscMPIInt)slab.nk; dims[3] = (PetscMPIInt)slab.i0;
                dims[4] = (PetscMPIInt)slab.j0; dims[5] = (PetscMPIInt)slab.k0;
                total  = (PetscMPIInt)(slab.ni * slab.nj * slab.nk);
            }
            MPI_Bcast(dims, 6, MPI_INT, 0, comm);
            MPI_Bcast(&total, 1, MPI_INT, 0, comm);

            if (rank != (PetscMPIInt)block_start_rank[b]) {
                slab.ni = dims[0]; slab.nj = dims[1]; slab.nk = dims[2];
                slab.i0 = dims[3]; slab.j0 = dims[4]; slab.k0 = dims[5];
                if (total > 0) {
                    slab.x.resize((size_t)total);
                    slab.y.resize((size_t)total);
                    slab.z.resize((size_t)total);
                }
            }
            if (total > 0) {
                MPI_Bcast(slab.x.data(), total, MPI_DOUBLE, 0, comm);
                MPI_Bcast(slab.y.data(), total, MPI_DOUBLE, 0, comm);
                MPI_Bcast(slab.z.data(), total, MPI_DOUBLE, 0, comm);
            }
        }
    }
}
// ====================================================================
// getDonorSlab — 查询本地缓存的 donor slab
// ====================================================================
const DonorSlab* MultiBlockMesh::getDonorSlab(int block_id, int face) const
{
    if (block_id < 0 || block_id >= (int)donor_slabs_.size()) return nullptr;
    if (face < 0 || face > 5) return nullptr;
    if (!donor_slabs_[block_id][face].valid) return nullptr;
    return &donor_slabs_[block_id][face];
}

