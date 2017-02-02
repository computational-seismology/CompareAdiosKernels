## Comparing ADIOS kernels for SPECFEM3D and SPECFEM3D_GLOBE

Required:
* cmake
* C++ compiler
* MPI libraries
* boost with mpi enable
* ADIOS and MXML

To compare and run:
```bash
 mkdir build
 cd build/
 cmake ..
 make -j 8
 mpirun -np 4 ./CompareAdiosKernels --help 
 
```

