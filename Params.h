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

    static const std::vector<std::string> default_kernels = {"rhonotprime_kl_crust_mantle", "kappa_kl_crust_mantle",
                                                             "beta_kl_crust_mantle", "bulk_c_kl_crust_mantle"};

    class Params {
    public:
        void set_from_cmdline(int argc, char* argv[]);

        void print(std::ostream& buffer);

        std::string get_reference_file() { return reference_file; }

        std::string get_kernels_file() { return kernels_file; }

        std::vector<std::string> get_kernel_names() { return kernel_names; }

        friend void broadcast_params(mpi::communicator& comm, Params& p);

    private:
        std::string reference_file;
        std::string kernels_file;
        float tolerance = 1.0e-2f;
        std::vector<std::string> kernel_names = default_kernels;
    };


}  // namespace kernel_validation

#endif  // COMPAREADIOSKERNELS_PARAMETERS_H
