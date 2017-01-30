#include "kernel_comparator.h"
#include "Params.h"
#include "adios_reader.h"

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>

#include <iostream>
#include <vector>
#include <numeric>

namespace mpi = boost::mpi;
namespace kv = ::kernel_validation;

int main(int argc, char* argv[])
{
    mpi::environment env;
    mpi::communicator world;

    kv::Params params;

    try {
        // Get parameters to every process
        if (!world.rank()) {
            params.set_from_cmdline(argc, argv);
            params.print(std::cerr);
        }
        broadcast_params(world, params);

        adios_read_init_method(ADIOS_READ_METHOD_BP, world, "");  // make it RAII, at some point
        if (adios_errno) throw kv::adios_exception;

        // Actual work
        kv::KernelComparator comparator(world, params.get_reference_file(), params.get_kernels_file());
        comparator.compare_multiple(100.f, params.get_kernel_names());

        adios_read_finalize_method(ADIOS_READ_METHOD_BP);
        if (adios_errno) throw kv::adios_exception;
    }
    catch (std::exception& e) {
        // Everything that fails or is not validated should end here.
        std::cerr << e.what() << std::endl;
        env.abort(-1);
    }
    return 0;
}
