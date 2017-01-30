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

#ifndef COMPAREADIOSKERNELS_PARAMETERS_H
#define COMPAREADIOSKERNELS_PARAMETERS_H

#include <boost/mpi/communicator.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/serialization/string.hpp>

#include <streambuf>

namespace mpi= boost::mpi;
namespace po = boost::program_options;
namespace fs = boost::filesystem;

namespace kernel_validation {
    /**
     * Default list of kernel names
     */
    static const std::vector<std::string> default_kernels = {"rhonotprime_kl_crust_mantle", "kappa_kl_crust_mantle",
                                                             "beta_kl_crust_mantle", "bulk_c_kl_crust_mantle"};

    /**
     * Class to read cmd line parameters.
     *
     * Primarily designed to be used by mpi rank == 0, with later broadcasting to other processes.
     * Should also work with all processes reading the cmd line.
     */
    class Params {
    public:
        /**
         * Read the command line and assigne relevant values to class attributes
         * @param argc The number of arguments from main()
         * @param argv The list of arguments from main()
         */
        void set_from_cmdline(int argc, char* argv[]);

        /**
         * Print parameter information
         * @param buffer Where to write the arguments to. Typically std::cerr
         */
        void print(std::ostream& buffer);

        /**
         * Accessor
         * @return The name of the reference file
         */
        std::string get_reference_file() { return reference_file; }

        /**
         * Accessor
         * @return The name of the kernel file
         */
        std::string get_kernels_file() { return kernels_file; }

        /**
         * Accessor
         * @return The list of kernel names to be compared
         */
        std::vector<std::string> get_kernel_names() { return kernel_names; }

        /**
         * Accessor
         * @return The tolerance of comparisons
         */
        float get_tolerance() { return tolerance; }


        /**
         * Propagate parameters to process with rank != 0
         * @param comm The MPI communicator in which operations are ran
         * @param p Params instance for the calling process. Probably empty before the call, except for rank == 0
         */
        friend void broadcast_params(mpi::communicator& comm, Params& p);

    private:
        std::string reference_file;
        std::string kernels_file;
        float tolerance = 1.0e-2f;
        std::vector<std::string> kernel_names = default_kernels;
    };


}  // namespace kernel_validation

#endif  // COMPAREADIOSKERNELS_PARAMETERS_H
