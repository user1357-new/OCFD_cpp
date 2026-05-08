#include "mesh_mutiblock.h"
#include <mpi.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cctype>
#include "mesh.h"
#include <petscdmda.h>
#include <petscvec.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <petsc.h>
#include <stdexcept>
#include <memory>
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
// 构造函数：初始化成员变量
MultiBlockMesh::MultiBlockMesh(const std::vector<std::array<PetscInt,3>> &procs,
                               PetscInt lap, PetscInt scheme_vis)
    : block_procs_(procs), LAP(lap), scheme_vis(scheme_vis)
{}
// 析构函数：释放所有子通信域
MultiBlockMesh::~MultiBlockMesh() {
    for (auto &comm : block_comms_) {
        if (comm != MPI_COMM_NULL && comm != PETSC_COMM_WORLD) {
            MPI_Comm_free(&comm);
        }
    }
}
// Initialize 核心逻辑
void MultiBlockMesh::Initialize(const std::string &tecplot_filename) {
    PetscMPIInt rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    // ---------- 1. 解析 Tecplot 文件（仅 rank 0）----------
    std::vector<BlockInfo> infos;
    if (rank == 0) {
        try {
            ParseTecplotFile(tecplot_filename, infos);
        } catch (const std::exception &e) {
            PetscPrintf(PETSC_COMM_WORLD, "Error parsing file: %s\n", e.what());
            MPI_Abort(PETSC_COMM_WORLD, 1);
        }
    }

    // 广播块数
    PetscInt num_blocks = static_cast<PetscInt>(infos.size());
    MPI_Bcast(&num_blocks, 1, MPIU_INT, 0, PETSC_COMM_WORLD);

    // 检查进程布局与块数是否一致
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
    //5——7结束（mesh还没修改）
    // ---------- 3. 确定本进程所属块 ----------
    PetscInt my_block = -1;
    PetscInt offset = 0;
    std::vector<PetscInt> block_start_rank(num_blocks);  // 每个块在全局中的第一个进程 rank
    for (PetscInt b = 0; b < num_blocks; ++b) {
        block_start_rank[b] = offset;
        if (rank >= offset && rank < offset + block_nprocs[b]) {
            my_block = b;
        }
        offset += block_nprocs[b];
    }
    if (my_block < 0) {
        throw std::runtime_error("Internal error: rank not assigned to any block");
    }

    // ---------- 4. 创建子通信域 ----------
    block_comms_.resize(num_blocks, MPI_COMM_NULL);
    MPI_Comm split_comm;
    MPI_Comm_split(PETSC_COMM_WORLD, my_block, rank - block_start_rank[my_block], &split_comm);
    block_comms_[my_block] = split_comm;

    // ---------- 5. 广播每个块的尺寸（所有进程都需要，以便接收数据）----------
    std::vector<PetscInt> dims(num_blocks * 3);
    if (rank == 0) {
        for (PetscInt i = 0; i < num_blocks; i++) {
            dims[3*i]   = infos[i].ni;
            dims[3*i+1] = infos[i].nj;
            dims[3*i+2] = infos[i].nk;
        }
    }
    MPI_Bcast(dims.data(), 3 * num_blocks, MPIU_INT, 0, PETSC_COMM_WORLD);

    // ---------- 6. 为每个块创建 Mesh ----------
    blocks_.clear();
    blocks_.reserve(num_blocks);
    const int tag = 12345;

    for (PetscInt b = 0; b < num_blocks; ++b) {
        PetscInt ni = dims[3*b];
        PetscInt nj = dims[3*b+1];
        PetscInt nk = dims[3*b+2];
        PetscInt npts = ni * nj * nk;

        if (b == my_block) {
            // --- 本进程属于此块 ---
            std::vector<PetscReal> x(npts), y(npts), z(npts);

            // 块内 rank 0 负责获取完整坐标（可能从全局 rank 0 点对点接收）
            if (rank == block_start_rank[b]) {
                if (rank == 0) {
                    // 全局 rank 0 直接拥有所有数据
                    x = infos[b].x;
                    y = infos[b].y;
                    z = infos[b].z;
                } else {
                    // 从全局 rank 0 接收
                    MPI_Recv(x.data(), npts, MPI_DOUBLE, 0, tag, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(y.data(), npts, MPI_DOUBLE, 0, tag+1, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
                    MPI_Recv(z.data(), npts, MPI_DOUBLE, 0, tag+2, PETSC_COMM_WORLD, MPI_STATUS_IGNORE);
                }
            }

            // 在子通信域内广播坐标
            MPI_Bcast(x.data(), npts, MPI_DOUBLE, 0, block_comms_[b]);
            MPI_Bcast(y.data(), npts, MPI_DOUBLE, 0, block_comms_[b]);
            MPI_Bcast(z.data(), npts, MPI_DOUBLE, 0, block_comms_[b]);

            // 获取本进程在子域内的 rank
            PetscInt my_rank_block;
            MPI_Comm_rank(block_comms_[b], &my_rank_block);

            // 创建 Mesh（使用子通信域）
            auto mesh_ptr = std::make_unique<Mesh>(
                ni, nj, nk,
                my_rank_block,
                block_procs_[b][0], block_procs_[b][1], block_procs_[b][2],
                LAP, GRID3D, scheme_vis,
                block_comms_[b]
            );
            mesh_ptr->InitializeFromCoordinates(x, y, z);
            blocks_.push_back(std::move(mesh_ptr));
        } else {
            // --- 本进程不属于此块，存入空指针 ---
            blocks_.push_back(nullptr);
        }
    }

    // 全局 rank 0 向其它块的第一个进程发送坐标数据
    if (rank == 0) {
        for (PetscInt b = 0; b < num_blocks; ++b) {
            if (block_start_rank[b] != 0) {
                PetscInt npts = infos[b].ni * infos[b].nj * infos[b].nk;
                MPI_Send(infos[b].x.data(), npts, MPI_DOUBLE, block_start_rank[b], tag, PETSC_COMM_WORLD);
                MPI_Send(infos[b].y.data(), npts, MPI_DOUBLE, block_start_rank[b], tag+1, PETSC_COMM_WORLD);
                MPI_Send(infos[b].z.data(), npts, MPI_DOUBLE, block_start_rank[b], tag+2, PETSC_COMM_WORLD);
            }
        }
    }

    // 全局同步（非必须，但有助于干净输出）
    MPI_Barrier(PETSC_COMM_WORLD);
}