/******************************************************************************
 * Copyright 2017 Matthieu Lefebvre
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 *       limitations under the License.
 ******************************************************************************/

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
        comparator.compare_multiple(params.get_tolerance(), params.get_kernel_names());

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
