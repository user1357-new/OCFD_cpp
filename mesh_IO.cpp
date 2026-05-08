#include "mesh.h"
#include <petscdmda.h>
#include <petscvec.h>
#include <iostream>
#include <fstream>
#include <cmath>
#include <mpi.h>
#include <vector>
#include <string>
// 公开初始化接口
PetscErrorCode Mesh::Initialize()
{
    return init_mesh();
} 
 // 初始化网格
PetscErrorCode Mesh::init_mesh()
{
    PetscErrorCode ierr;
    
    // Step 1: 根据网格类型读取网格
    if (Iflag_Gridtype == GRID1D) {
        ierr = read_mesh1d();
        CHKERRQ(ierr);
    } else if (Iflag_Gridtype == GRID2D_PLANE) {
        ierr = read_mesh2d_plane();
        CHKERRQ(ierr);
    } else if (Iflag_Gridtype == GRID2D_AXIAL_SYMM) {
        ierr = read_mesh2d_AxialSymm();
        CHKERRQ(ierr);
    } else if (Iflag_Gridtype == GRID3D || Iflag_Gridtype == GRID_AND_JACOBIAN3D) {
        ierr = read_mesh3d();
        CHKERRQ(ierr);
    }
    
    // Step 2: 交换边界数据（MPI部分，留空给您实现）
    exchange_boundary_xyz(Axx);
    exchange_boundary_xyz(Ayy);
    exchange_boundary_xyz(Azz);
    
    // Step 3: 调用用户注册的边界条件处理函数（包括周期性边界等）
    if (boundary_condition_handler) {
        boundary_condition_handler();
    }
    
    // Step 4: 如果不是预计算的雅可比，则计算
    if (Iflag_Gridtype != GRID_AND_JACOBIAN3D) {
        ierr = comput_Jacobian3d();
        CHKERRQ(ierr);
    }
    
    // Step 5: 交换雅可比系数边界（MPI部分，留空给您实现）
    exchange_boundary_xyz(Akx);
    exchange_boundary_xyz(Aky);
    exchange_boundary_xyz(Akz);
    exchange_boundary_xyz(Aix);
    exchange_boundary_xyz(Aiy);
    exchange_boundary_xyz(Aiz);
    exchange_boundary_xyz(Asx);
    exchange_boundary_xyz(Asy);
    exchange_boundary_xyz(Asz);
    exchange_boundary_xyz(Ajac);
    
    // Step 6: Ghost Cell边界处理（MPI部分，留空给您实现）
    ierr = Jac_Ghost_boundary();
    CHKERRQ(ierr);
    
    if (my_id == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "Initial Mesh OK\n");
    }
    
    return 0;
}
// 读取1D网格（对应原read_mesh1d）
PetscErrorCode Mesh::read_mesh1d()
{
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    // 分配临时数组存储全局网格
    PetscReal *x0 = new PetscReal[nx_global];
    PetscReal *y0 = new PetscReal[ny_global];
    PetscReal *z0 = new PetscReal[nz_global];
    
    // 进程0读取文件
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "read 1D mesh ...\n");
        
        std::ifstream file("OCFD-grid.dat", std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
                    "Cannot open OCFD-grid.dat");
        }
        
        file.read(reinterpret_cast<char*>(x0), nx_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(y0), ny_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(z0), nz_global * sizeof(PetscReal));
        
        file.close();
    }
    
    // 广播到所有进程
    MPI_Bcast(x0, nx_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(y0, ny_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(z0, nz_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    
    // 创建DMDA：使用您规定的 npx0, npy0, npz0 进行分块
    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,
                        nx_global, ny_global, nz_global,
                        npx0, npy0, npz0,  // ← 这里使用了您规定的参数
                        1, LAP,
                        NULL, NULL, NULL,
                        &da);
    CHKERRQ(ierr);
    
    ierr = DMSetUp(da);
    CHKERRQ(ierr);
    
    // 创建坐标向量
    ierr = DMCreateGlobalVector(da, &Axx);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Ayy);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Azz);
    CHKERRQ(ierr);
    
    // 获取局部信息
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info);
    CHKERRQ(ierr);
    
    nx = info.xm; ny = info.ym; nz = info.zm;
    
    PetscReal ***axx, ***ayy, ***azz;
    ierr = DMDAVecGetArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    // 简化逻辑：直接使用 info.xs/ys/zs 作为全局索引访问 x0/y0/z0
    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                axx[k][j][i] = x0[i];
                ayy[k][j][i] = y0[j];
                azz[k][j][i] = z0[k];
            }
        }
    }
    
    ierr = DMDAVecRestoreArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    delete[] x0;
    delete[] y0;
    delete[] z0;
    
    return 0;
}

// 读取2D平面网格
PetscErrorCode Mesh::read_mesh2d_plane()
{
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    PetscReal *x2 = new PetscReal[nx_global * ny_global];
    PetscReal *y2 = new PetscReal[nx_global * ny_global];
    PetscReal *z1 = new PetscReal[nz_global];
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "read 2D plane mesh ...\n");
        
        std::ifstream file("OCFD-grid.dat", std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
                    "Cannot open OCFD-grid.dat");
        }
        
        // 跳过 Fortran unformatted 记录标记（每个记录前后有4字节长度）
        int record_len;
        
        // 读取 x2d
        file.read(reinterpret_cast<char*>(&record_len), 4);
        file.read(reinterpret_cast<char*>(x2), nx_global * ny_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(&record_len), 4);
        
        // 读取 y2d
        file.read(reinterpret_cast<char*>(&record_len), 4);
        file.read(reinterpret_cast<char*>(y2), nx_global * ny_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(&record_len), 4);
        
        // 读取 z1d
        file.read(reinterpret_cast<char*>(&record_len), 4);
        file.read(reinterpret_cast<char*>(z1), nz_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(&record_len), 4);
        
        file.close();
    }
    
    MPI_Bcast(x2, nx_global * ny_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(y2, nx_global * ny_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(z1, nz_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    
    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,
                        nx_global, ny_global, nz_global,
                        npx0, npy0, npz0, // 使用规定分块
                        1, LAP,
                        NULL, NULL, NULL,
                        &da);
    CHKERRQ(ierr);
    
    ierr = DMSetUp(da);
    CHKERRQ(ierr);
    

    ierr = DMCreateGlobalVector(da, &Axx);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Ayy);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Azz);
    CHKERRQ(ierr);
    
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info);
    CHKERRQ(ierr);
    
    nx = info.xm; ny = info.ym; nz = info.zm;
    

    nx = info.xm; ny = info.ym; nz = info.zm;
    
    PetscPrintf(PETSC_COMM_WORLD, "Rank %d: xs=%d, xm=%d, ys=%d, ym=%d, zs=%d, zm=%d\n", 
                rank, info.xs, info.xm, info.ys, info.ym, info.zs, info.zm);
    



    PetscReal ***axx, ***ayy, ***azz;
    ierr = DMDAVecGetArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                
                PetscInt idx = j * nx_global + i;
                axx[k][j][i] = x2[idx];
                ayy[k][j][i] = y2[idx];
                azz[k][j][i] = z1[k];
             
            }
        }
    }
    
    ierr = DMDAVecRestoreArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    delete[] x2;
    delete[] y2;
    delete[] z1;
    
    return 0;
}

// 读取2D轴对称网格
PetscErrorCode Mesh::read_mesh2d_AxialSymm()
{
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    PetscReal *x2 = new PetscReal[nx_global * ny_global];
    PetscReal *R2 = new PetscReal[nx_global * ny_global];
    PetscReal *seta1 = new PetscReal[nz_global];
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "read 2D axial symm mesh ...\n");
        
        std::ifstream file("OCFD-grid.dat", std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
                    "Cannot open OCFD-grid.dat");
        }
        
        file.read(reinterpret_cast<char*>(x2), 
                  nx_global * ny_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(R2), 
                  nx_global * ny_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(seta1), 
                  nz_global * sizeof(PetscReal));
        
        file.close();
    }
    
    MPI_Bcast(x2, nx_global * ny_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(R2, nx_global * ny_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(seta1, nz_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    
    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,
                        nx_global, ny_global, nz_global,
                        npx0, npy0, npz0, // 使用规定分块
                        1, LAP,
                        NULL, NULL, NULL,
                        &da);
    CHKERRQ(ierr);
    
    ierr = DMSetUp(da);
    CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(da, &Axx);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Ayy);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Azz);
    CHKERRQ(ierr);
    
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info);
    CHKERRQ(ierr);
    
    nx = info.xm; ny = info.ym; nz = info.zm;
    
    PetscReal ***axx, ***ayy, ***azz;
    ierr = DMDAVecGetArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                PetscInt idx = j * nx_global + i;
                axx[k][j][i] = x2[idx];
                ayy[k][j][i] = R2[idx] * std::cos(seta1[k]);
                azz[k][j][i] = R2[idx] * std::sin(seta1[k]);
            }
        }
    }
    
    ierr = DMDAVecRestoreArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    delete[] x2;
    delete[] R2;
    delete[] seta1;
    
    return 0;
}

// 读取3D网格
PetscErrorCode Mesh::read_mesh3d()
{
    PetscErrorCode ierr;
    PetscMPIInt rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    
    // 分配临时数组存储全局网格
    PetscReal *x3 = new PetscReal[nx_global * ny_global * nz_global];
    PetscReal *y3 = new PetscReal[nx_global * ny_global * nz_global];
    PetscReal *z3 = new PetscReal[nx_global * ny_global * nz_global];
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "read 3D mesh ...\n");
        
        std::ifstream file("OCFD-grid.dat", std::ios::binary | std::ios::in);
        if (!file.is_open()) {
            SETERRQ(PETSC_COMM_SELF, PETSC_ERR_FILE_OPEN,
                    "Cannot open OCFD-grid.dat");
        }
        
        // 读取所有网格点的 x, y, z 坐标
        file.read(reinterpret_cast<char*>(x3), 
                  nx_global * ny_global * nz_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(y3), 
                  nx_global * ny_global * nz_global * sizeof(PetscReal));
        file.read(reinterpret_cast<char*>(z3), 
                  nx_global * ny_global * nz_global * sizeof(PetscReal));
        
        file.close();
    }
    
    // 广播到所有进程
    MPI_Bcast(x3, nx_global * ny_global * nz_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(y3, nx_global * ny_global * nz_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    MPI_Bcast(z3, nx_global * ny_global * nz_global, MPI_DOUBLE, 0, PETSC_COMM_WORLD);
    
    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_BOX,
                        nx_global, ny_global, nz_global,
                        npx0, npy0, npz0,
                        1, LAP,
                        NULL, NULL, NULL,
                        &da);
    CHKERRQ(ierr);
    
    ierr = DMSetUp(da);
    CHKERRQ(ierr);
    
    ierr = DMCreateGlobalVector(da, &Axx);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Ayy);
    CHKERRQ(ierr);
    ierr = VecDuplicate(Axx, &Azz);
    CHKERRQ(ierr);
    
    DMDALocalInfo info;
    ierr = DMDAGetLocalInfo(da, &info);
    CHKERRQ(ierr);
    
    nx = info.xm; ny = info.ym; nz = info.zm;
    
    // 获取局部数组并填充网格坐标
    PetscReal ***axx, ***ayy, ***azz;
    ierr = DMDAVecGetArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    // 使用全局索引填充当前进程的局部网格
    for (PetscInt k = info.zs; k < info.zs + info.zm; k++) {
        for (PetscInt j = info.ys; j < info.ys + info.ym; j++) {
            for (PetscInt i = info.xs; i < info.xs + info.xm; i++) {
                // 3D 网格的线性索引
                PetscInt idx = k * nx_global * ny_global + j * nx_global + i;
                axx[k][j][i] = x3[idx];
                ayy[k][j][i] = y3[idx];
                azz[k][j][i] = z3[idx];
            }
        }
    }
    
    ierr = DMDAVecRestoreArray(da, Axx, &axx);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Ayy, &ayy);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArray(da, Azz, &azz);
    CHKERRQ(ierr);
    
    // 释放临时数组
    delete[] x3;
    delete[] y3;
    delete[] z3;
    
    // 如果是预计算的雅可比类型，则创建雅可比向量（但不计算）
    if (Iflag_Gridtype == GRID_AND_JACOBIAN3D) {
        ierr = DMCreateGlobalVector(da, &Akx);
        CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Aky); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Akz); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Aix); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Aiy); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Aiz); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Asx); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Asy); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Asz); CHKERRQ(ierr);
        ierr = VecDuplicate(Akx, &Ajac); CHKERRQ(ierr);
    }
    
    if (rank == 0) {
        PetscPrintf(PETSC_COMM_WORLD, "read 3D mesh OK\n");
    }
    
    return 0;
}
// 导出到 Tecplot(dat格式)
PetscErrorCode Mesh::ExportToTecplot(const std::string &filename)
{
    PetscErrorCode ierr;
    PetscMPIInt rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

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

    MPI_Reduce(X_global.data(), X_recv.data(), global_size, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(Y_global.data(), Y_recv.data(), global_size, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(Z_global.data(), Z_recv.data(), global_size, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(Akx_global.data(), Akx_recv.data(), global_size, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(Akx1_global.data(), Akx1_recv.data(), global_size, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(Axx_global.data(), Axx_recv.data(), global_size, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);

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
        PetscPrintf(PETSC_COMM_WORLD, "Data exported to %s\n", outname.c_str());
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
    PetscPrintf(PETSC_COMM_WORLD,
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