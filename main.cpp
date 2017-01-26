#include "kernel_comparator.h"
#include "parameters.h"

#include <iostream>
#include <vector>
#include <numeric>

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>


namespace mpi = boost::mpi;
namespace kv = ::kernel_validation;


int main(int argc, char* argv[])
{
    mpi::environment env;
    mpi::communicator world;

    kv::Params params;

    try {
      if (!world.rank()) {
        params.set_from_cmdline(argc, argv);
        params.print();
      }
      broadcast_params(world, params);
      //if (!world.rank())
        //std::cerr << "Running MPI with: " << world.size() << " processes." << std::endl;

      adios_read_init_method(ADIOS_READ_METHOD_BP, world, "");

      KernelComparator comparator(world, params.get_reference_file(), params.get_kernels_file());
      comparator.compare_multiple(100.f, params.get_kernel_names());

      adios_read_finalize_method(ADIOS_READ_METHOD_BP);

    } catch (std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
      env.abort(-1);
    } catch (std::exception& e) {
      std::cerr << e.what() << std::endl;
      env.abort(-1);
    }
    return 0;
}
