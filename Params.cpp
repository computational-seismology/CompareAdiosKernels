#include "Params.h"

#include <boost/mpi/collectives.hpp>

namespace kernel_validation {

    void Params::set_from_cmdline(int argc, char* argv[])
    {
        po::options_description desc("Options");
        desc.add_options()
                ("help", "Produce this help message")
                ("reference", po::value<std::string>(), "Set reference file.")
                ("kernels", po::value<std::string>(), "Set kernel file to be tested.")
                ("tolerance", po::value<float>()->default_value(tolerance), "Set comparison tolerance.")
                ("kernel-names", po::value<std::vector<std::string>>()
                        ->multitoken()
                        ->zero_tokens()
                        ->composing(), "List of space separated kernel names");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        po::notify(vm);

        try {
            if (vm.count("help")) {
                throw std::runtime_error("Asked for help.");
            }

            if (vm.count("reference")) {
                reference_file = vm["reference"].as<std::string>();
                if (!fs::exists(reference_file) || !fs::is_regular_file(reference_file)) {
                    throw std::runtime_error("Reference file does not exists.");
                }
            }
            else {
                throw std::runtime_error("Should specify a reference file.");
            }

            if (vm.count("kernels")) {
                kernels_file = vm["kernels"].as<std::string>();
                if (!fs::exists(kernels_file) || !fs::is_regular_file(kernels_file)) {
                    throw std::runtime_error("Reference file does not exists.");
                }
            }
            else {
                throw std::runtime_error("Should specify a kernels file.");
            }
            if (vm.count("precision")) {
                tolerance = vm["tolerance"].as<float>();
            }
            if (vm.count("kernel-names")) {
                kernel_names = vm["kernel-names"].as<std::vector<std::string>>();
            }

        }
        catch (std::runtime_error& e) {
            std::cerr << desc << "\n";
            std::cerr << "Default kernels are: " << std::endl;
            for (auto& x: kernel_names) {
                std::cerr << "   *  " << x << std::endl;
            }
            throw e;
        }
    }

    void Params::print(std::ostream& buffer)
    {
        buffer << std::endl
               << "********************************************" << std::endl
               << "|   PARAMETERS                             |" << std::endl
               << "--------------------------------------------" << std::endl
               << "Reference file is: " << reference_file << std::endl
               << "Kernels file is: " << kernels_file << std::endl
               << "Using a tolerance of: " << tolerance << std::endl
               << "Kernels to be computed are: " << std::endl;
        for (auto& x: kernel_names) {
            buffer << "   *  " << x << std::endl;
        }
        buffer << "********************************************" << std::endl << std::endl;
    }

    void broadcast_params(mpi::communicator& comm, Params& params)
    {
        mpi::broadcast(comm, params.reference_file, 0);
        mpi::broadcast(comm, params.kernels_file, 0);
        mpi::broadcast(comm, params.tolerance, 0);

        int num_kernels = params.kernel_names.size();
        mpi::broadcast(comm, num_kernels, 0);
        if (comm.rank()) {
            params.kernel_names.clear();
            params.kernel_names.resize(num_kernels);
        }
        for (auto& k : params.kernel_names) {
            mpi::broadcast(comm, k, 0);
        }
    }


}  // namespace kernel_validation
