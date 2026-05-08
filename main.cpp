
#include "mesh.h"
#include <petsc.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    
    PetscMPIInt rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);
    
    // 根据你的OCFD-grid.dat实际尺寸设置
    PetscInt nx_global = 181;   // 改成你的实际值
    PetscInt ny_global = 101;   // 改成你的实际值
    PetscInt nz_global = 16;    // 2D平面，Z方向为1
    
    PetscInt npx0 = 2;
    PetscInt npy0 = 2;
    PetscInt npz0 = 1;
    
    PetscInt scheme_vis = OCFD_Scheme_CD6;
    PetscInt LAP = (scheme_vis == OCFD_Scheme_CD8) ? 4 : 3;  // 自动根据差分格式设置stencil width
    PetscInt grid_type = GRID2D_PLANE;  // 2D平面网格
    
    Mesh mesh(nx_global, ny_global, nz_global,
              rank, npx0, npy0, npz0,
              LAP, grid_type, scheme_vis);
    
    mesh.RegisterBoundaryConditionHandler([&]() {
        // 边界条件处理
    });
    
    mesh.Initialize();  // 这里会读取OCFD-grid.dat并计算Axx,Akx等
    
    mesh.printInfo();
    
    mesh.ExportToTecplot("mesh_data");
    
    PetscFinalize();
    return 0;
}