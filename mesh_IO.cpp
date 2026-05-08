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

    // ---------- 3. 确定本进程所属块、计算每个块起始 rank ----------
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
    MPI_Comm_split(PETSC_COMM_WORLD, my_block, rank - block_start_rank[my_block], &split_comm);
    block_comms_[my_block] = split_comm;

    // ---------- 5. 广播每个块的尺寸 ----------
    std::vector<PetscInt> dims(num_blocks * 3);
    if (rank == 0) {
        for (PetscInt i = 0; i < num_blocks; i++) {
            dims[3*i]   = infos[i].ni;
            dims[3*i+1] = infos[i].nj;
            dims[3*i+2] = infos[i].nk;
        }
    }
    MPI_Bcast(dims.data(), 3 * num_blocks, MPIU_INT, 0, PETSC_COMM_WORLD);

    // ---------- 6. 全局 rank 0 预先发送坐标给各块的第一个进程 ----------
    const int tag = 12345;
    if (rank == 0) {
        for (PetscInt b = 0; b < num_blocks; ++b) {
            PetscInt npts = infos[b].ni * infos[b].nj * infos[b].nk;
            if (block_start_rank[b] == 0) continue; // 自己就是该块第一个进程，无需发送
            MPI_Send(infos[b].x.data(), npts, MPI_DOUBLE, block_start_rank[b], tag,   PETSC_COMM_WORLD);
            MPI_Send(infos[b].y.data(), npts, MPI_DOUBLE, block_start_rank[b], tag+1, PETSC_COMM_WORLD);
            MPI_Send(infos[b].z.data(), npts, MPI_DOUBLE, block_start_rank[b], tag+2, PETSC_COMM_WORLD);
        }
    }

    // ---------- 7. 为每个块创建 Mesh ----------
    blocks_.clear();
    blocks_.reserve(num_blocks);

    for (PetscInt b = 0; b < num_blocks; ++b) {
        PetscInt ni = dims[3*b];
        PetscInt nj = dims[3*b+1];
        PetscInt nk = dims[3*b+2];
        PetscInt npts = ni * nj * nk;

        if (b == my_block) {
            std::vector<PetscReal> x(npts), y(npts), z(npts);

            // 块内 rank 0 获取完整坐标
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

            // 在子通信域内广播坐标
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
        } else {
            blocks_.push_back(nullptr);
        }
    }

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